"""
ParkSense core: shared by the web app and the training scripts so that slot crops
are produced in exactly the same way at training time and in production.

  * order_quad / rectify  - perspective rectification of a slot quadrilateral (Week 2)
  * baseline_features     - classical edge-density + colour-variance features (Weeks 2-3)
  * OccupancyModel        - ONNX Runtime wrapper for the CNN (Weeks 6-7, Lecture D)
"""
from __future__ import annotations

import json
import os
import numpy as np
import cv2

CROP = 64  # rectified slot crop is CROP x CROP pixels


# ----------------------------------------------------------------------------- geometry
def to_quad(points) -> np.ndarray:
    """Accept a flat [x1,y1,...] list or [[x,y],...]; return a 4x2 float32 quad.
    Polygons with more than 4 points are reduced with the minimum-area rectangle."""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(p) != 4:
        p = cv2.boxPoints(cv2.minAreaRect(p)).astype(np.float32)
    return p


def order_quad(q: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting top-left, then rotate the order so that the
    LONG side of the slot becomes vertical in the rectified crop (canonical view:
    a car always appears 'nose up', whatever the camera angle)."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])
    q = q[np.argsort(ang)]  # clockwise in image coords (y down)
    # start with the point closest to the top-left of the bounding box
    start = np.argmin(q[:, 0] + q[:, 1])
    q = np.roll(q, -start, axis=0)
    top = np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
    side = np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
    if top > side:  # horizontal slot -> rotate by one corner so long side is vertical
        q = np.roll(q, -1, axis=0)
    return q.astype(np.float32)


def rectify(img: np.ndarray, points, size: int = CROP) -> np.ndarray:
    """Warp the slot quadrilateral to a size x size top-down crop (BGR uint8)."""
    q = order_quad(to_quad(points))
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    H = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------- baseline
BASELINE_FEATURES = ["edge_density", "lap_var", "gray_std", "sat_mean", "sat_std", "hue_std",
                     "grad_mean", "center_edge"]


def baseline_features(crop: np.ndarray) -> np.ndarray:
    """Classical features for one rectified crop. Cars add edges, texture and colour;
    empty asphalt is smooth and grey. Features are made robust to global brightness."""
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    med = float(np.median(g)) + 1e-6
    edges = cv2.Canny(g, int(0.66 * med), int(1.33 * med))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = g.shape
    center = edges[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return np.array([
        edges.mean() / 255.0,
        cv2.Laplacian(g, cv2.CV_32F).var() / (med ** 2),
        g.std() / med,
        hsv[..., 1].mean() / 255.0,
        hsv[..., 1].std() / 255.0,
        hsv[..., 0].std() / 180.0,
        mag.mean() / med,
        center.mean() / 255.0,
    ], dtype=np.float32)


class BaselineModel:
    """Logistic regression on baseline_features, stored as plain JSON (no sklearn at runtime)."""

    def __init__(self, path: str):
        p = json.load(open(path))
        self.mu = np.array(p["mean"], np.float32)
        self.sd = np.array(p["std"], np.float32)
        self.w = np.array(p["coef"], np.float32)
        self.b = float(p["intercept"])

    def predict_proba(self, crops: list[np.ndarray]) -> np.ndarray:
        X = np.stack([baseline_features(c) for c in crops])
        z = ((X - self.mu) / self.sd) @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------- CNN (ONNX)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(crops: list[np.ndarray]) -> np.ndarray:
    x = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class OccupancyModel:
    def __init__(self, path: str, threads: int = 1):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        meta = path.replace(".onnx", ".json")
        self.meta = json.load(open(meta)) if os.path.exists(meta) else {}
        self.temperature = float(self.meta.get("temperature", 1.0))

    def predict_proba(self, crops: list[np.ndarray], batch: int = 64) -> np.ndarray:
        """Return P(occupied) for each crop (temperature-calibrated)."""
        out = []
        for i in range(0, len(crops), batch):
            logits = self.sess.run(None, {self.inp: preprocess(crops[i:i + batch])})[0]
            z = logits / self.temperature
            z = z - z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            out.append(p[:, 1])
        return np.concatenate(out) if out else np.zeros(0, np.float32)


# ----------------------------------------------------------------------------- image QA
def image_quality(img: np.ndarray) -> dict:
    """Cheap checks that drive the 'not sure' state for the whole frame."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
    return {
        "brightness": float(small.mean()),
        "contrast": float(small.std()),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_32F).var()),
    }
