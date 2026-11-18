"""
ParkSense - FastAPI back end (Render free web service, 512 MB RAM).

Endpoints
  GET  /                 operator page (draw slots, analyse, occupancy-over-time)
  GET  /public           public occupancy page - counts only, never images
  GET  /help             short help page
  GET  /model-card       model card
  POST /api/predict      image + slot layout -> free / occupied / not sure per slot
  POST /api/publish      operator pushes the latest COUNTS (no image) to the public page
  GET  /api/lots         public counts (+ short history) for every lot
  GET  /api/samples      demo images and layouts bundled with the app
  GET  /api/health       liveness + uptime (used to measure cold starts)

Privacy by design: uploaded images are decoded in memory, never written to disk and
never returned; no licence-plate reading or vehicle identification is performed.
"""
from __future__ import annotations

import json
import os
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from parksense_core import BaselineModel, OccupancyModel, image_quality, rectify

HERE = Path(__file__).parent
MODELS = Path(os.environ.get("PARKSENSE_MODELS", HERE.parent / "models"))
CNN_PATH = MODELS / os.environ.get("PARKSENSE_CNN", "parksense_cnn_int8.onnx")
STARTED = time.time()

# Decision thresholds on the calibrated P(occupied). Inside the band -> "not sure".
UNSURE_LOW = float(os.environ.get("UNSURE_LOW", 0.25))
UNSURE_HIGH = float(os.environ.get("UNSURE_HIGH", 0.75))
MAX_SIDE = 2000          # larger uploads are downscaled (and the layout with them)
MAX_BYTES = 8 * 1024 * 1024
MAX_SLOTS = 400

app = FastAPI(title="ParkSense", docs_url="/api/docs", redoc_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
app.mount("/samples", StaticFiles(directory=HERE / "samples"), name="samples")

cnn = OccupancyModel(str(CNN_PATH), threads=1)
baseline = BaselineModel(str(MODELS / "baseline.json"))
_ = cnn.predict_proba([np.zeros((64, 64, 3), np.uint8)])  # warm-up

LOTS: dict[str, dict] = {}
LOCK = threading.Lock()


def _page(name: str):
    return FileResponse(HERE / "static" / name)


@app.get("/")
def index():
    return _page("index.html")


@app.get("/public")
def public():
    return _page("public.html")


@app.get("/help")
def help_page():
    return _page("help.html")


@app.get("/model-card")
def model_card():
    return _page("model_card.html")


@app.get("/api/health")
def health():
    return {"ok": True, "uptime_s": round(time.time() - STARTED, 1), "model": CNN_PATH.name}


@app.get("/api/samples")
def samples():
    return json.load(open(HERE / "samples" / "index.json"))


def _decode(data: bytes) -> np.ndarray:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 8 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not read the image (use JPG or PNG)")
    return img


def _parse_layout(layout: str, w: int, h: int):
    try:
        L = json.loads(layout)
    except json.JSONDecodeError:
        raise HTTPException(400, "Layout is not valid JSON")
    slots = L.get("slots", L if isinstance(L, list) else [])
    if not slots:
        raise HTTPException(400, "Layout has no slots - draw at least one slot")
    if len(slots) > MAX_SLOTS:
        raise HTTPException(400, f"At most {MAX_SLOTS} slots per image")
    sx = sy = 1.0
    if isinstance(L, dict) and L.get("image_size"):  # layout drawn on a different resolution
        lw, lh = L["image_size"]; sx, sy = w / lw, h / lh
    out = []
    for i, s in enumerate(slots):
        pts = np.asarray(s.get("points"), np.float32).reshape(-1, 2)
        if len(pts) < 4:
            raise HTTPException(400, f"Slot {s.get('id', i)} needs 4 corner points")
        pts[:, 0] *= sx; pts[:, 1] *= sy
        out.append((str(s.get("id", i + 1)), pts))
    return out


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), layout: str = Form(...), model: str = Form("cnn")):
    t0 = time.perf_counter()
    img = _decode(await image.read())
    h, w = img.shape[:2]
    slots = _parse_layout(layout, w, h)
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        slots = [(i, p * scale) for i, p in slots]
    t1 = time.perf_counter()
    crops = [rectify(img, p) for _, p in slots]
    if model == "baseline":
        probs = baseline.predict_proba(crops)
    elif model == "combined":  # average of CNN and baseline: more robust on a new, uncalibrated camera
        probs = 0.5 * cnn.predict_proba(crops) + 0.5 * baseline.predict_proba(crops)
    else:
        probs = cnn.predict_proba(crops)
    t2 = time.perf_counter()

    q = image_quality(img)
    warnings = []
    # Thresholds set on PKLot/CNRPark frames vs simulated night/fog/blur (results/quality_gate.json):
    # 2% false alarms on real frames; flags 100% of night, 92% of fog, 88% of blurred frames.
    if q["brightness"] < 50:
        warnings.append("Image is very dark (night or under-exposed). The model was not trained on night images: check counts by eye.")
    if q["contrast"] / max(q["brightness"], 1.0) < 0.2:
        warnings.append("Very low contrast (fog, glare or a dirty lens). Accuracy drops sharply in fog: check counts by eye.")
    if q["sharpness"] < 300:
        warnings.append("Image looks blurred or out of focus: counts may be unreliable.")

    res, n_free, n_occ, n_uns = [], 0, 0, 0
    for (sid, _), p in zip(slots, probs):
        p = float(p)
        if p >= UNSURE_HIGH:
            st = "occupied"; n_occ += 1
        elif p <= UNSURE_LOW:
            st = "free"; n_free += 1
        else:
            st = "unsure"; n_uns += 1
        conf = p if p >= 0.5 else 1 - p
        res.append({"id": sid, "p_occupied": round(p, 4), "state": st, "confidence": round(conf, 3)})
    total = len(res)
    # best estimate of occupied count: confident occupied + expected value of unsure slots
    expected_occ = n_occ + sum(r["p_occupied"] for r in res if r["state"] == "unsure")
    return JSONResponse({
        "slots": res,
        "summary": {"total": total, "free": n_free, "occupied": n_occ, "unsure": n_uns,
                    "occupied_estimate": round(expected_occ, 1),
                    "occupancy_pct": round(100 * expected_occ / total, 1),
                    "needs_review": bool(warnings) or n_uns > 0.15 * total},
        "quality": {k: round(v, 1) for k, v in q.items()},
        "warnings": warnings,
        "model": {"baseline": "Classical baseline (edges + colour)", "combined": "CNN + baseline average"}.get(model, "CNN classifier"),
        "timing_ms": {"decode": round(1000 * (t1 - t0), 1), "classify": round(1000 * (t2 - t1), 1),
                      "total": round(1000 * (time.perf_counter() - t0), 1)},
        "thresholds": {"unsure_low": UNSURE_LOW, "unsure_high": UNSURE_HIGH},
    })


@app.post("/api/publish")
async def publish(payload: dict):
    """Counts only. Any extra fields (e.g. images) are dropped on purpose."""
    lot = str(payload.get("lot", "Main car park"))[:60]
    try:
        rec = {"t": float(payload.get("t", time.time())),
               "total": int(payload["total"]), "free": int(payload["free"]),
               "occupied": int(payload["occupied"]), "unsure": int(payload.get("unsure", 0))}
    except (KeyError, ValueError, TypeError):
        raise HTTPException(400, "Need total, free and occupied counts")
    with LOCK:
        e = LOTS.setdefault(lot, {"history": deque(maxlen=96)})
        e["latest"] = rec; e["history"].append(rec)
    return {"ok": True}


@app.get("/api/lots")
def lots():
    with LOCK:
        return {k: {"latest": v["latest"], "history": list(v["history"])} for k, v in LOTS.items()}
