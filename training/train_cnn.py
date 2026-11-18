"""
ParkSense - main CV model (Weeks 6-7): a small CNN that classifies a rectified
64x64 slot crop as free / occupied.  Architecture follows the mAlexNet idea
(Amato et al., 2017: a tiny AlexNet-style network generalises well across car parks)
with BatchNorm and global average pooling -> ~0.1 M parameters, < 1 MB in ONNX.

Training data: UFPR04 'train' days only (Sunny + Cloudy).
Optional extra training crops from the generative translator (--extra *.npz).

  python training/train_cnn.py --name cnn_basic
  python training/train_cnn.py --name cnn_photo --aug photometric
  python training/train_cnn.py --name cnn_gan   --extra data/gen_cyclegan_rain.npz

Outputs: runs/<name>/model.pt, runs/<name>/probs.npy (P(occupied) for every crop in
data/crops.npz), runs/<name>/meta.json (temperature, val metrics) and, with --export,
models/<name>.onnx (+ int8 version).
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(__file__))
from parksense_core import MEAN, STD  # noqa
from evaluate_lib import evaluate  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")


class ParkNet(nn.Module):
    def __init__(self, w=(24, 48, 64, 96), p_drop=0.3):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(
            blk(3, w[0], 5), nn.MaxPool2d(2),        # 32
            blk(w[0], w[1], 3), nn.MaxPool2d(2),     # 16
            blk(w[1], w[2], 3), nn.MaxPool2d(2),     # 8
            blk(w[2], w[3], 3),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(w[3], 2))

    def forward(self, x):
        return self.head(self.f(x))


# ------------------------------------------------------------------ augmentation (numpy, BGR uint8)
def geometric(x, rng):
    if rng.random() < .5: x = x[:, ::-1]
    if rng.random() < .5: x = x[::-1]
    x = np.rot90(x, rng.integers(4))
    # small random scale/shift - simulates imprecise slot outlines drawn by an operator
    s = rng.uniform(0.85, 1.15); tx, ty = rng.uniform(-5, 5, 2)
    M = np.array([[s, 0, 32 * (1 - s) + tx], [0, s, 32 * (1 - s) + ty]], np.float32)
    return cv2.warpAffine(np.ascontiguousarray(x), M, (64, 64), borderMode=cv2.BORDER_REFLECT)


def photometric(x, rng):
    """Classical (non-generative) weather/night simulation - the control arm."""
    x = x.astype(np.float32)
    x = x * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)                       # brightness
    m = x.mean(); x = (x - m) * rng.uniform(0.6, 1.4) + m                       # contrast
    g = x.mean(2, keepdims=True); x = g + (x - g) * rng.uniform(0.5, 1.5)       # saturation
    r = rng.random()
    if r < .15:   # 'night': dark, blue cast, sensor noise
        x = x * rng.uniform(.15, .4) * np.array([1.2, 1.0, .8]) + rng.normal(0, 6, x.shape)
    elif r < .3:  # 'fog': blend toward a bright grey veil
        a = rng.uniform(.3, .65); x = x * (1 - a) + a * rng.uniform(170, 230)
    elif r < .45:  # 'rain': darker, lower contrast, streaks + blur
        x = x * .75 + 20
        for _ in range(rng.integers(5, 20)):
            px, py = rng.integers(0, 64, 2); cv2.line(x, (int(px), int(py)), (int(px) + 3, int(py) + 12), (200, 200, 200), 1)
        x = cv2.GaussianBlur(x, (3, 3), 0)
    if rng.random() < .2:
        x = cv2.GaussianBlur(x, (3, 3), rng.uniform(.3, 1.2))
    return np.clip(x, 0, 255).astype(np.uint8)


def to_tensor(batch):
    x = np.stack([cv2.cvtColor(np.ascontiguousarray(b), cv2.COLOR_BGR2RGB) for b in batch]).astype(np.float32) / 255.
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


@torch.no_grad()
def predict_logits(model, X, bs=512):
    model.eval(); out = []; dev = next(model.parameters()).device
    for i in range(0, len(X), bs):
        out.append(model(to_tensor(X[i:i + bs]).to(dev)).cpu())
    return torch.cat(out)


def fit_temperature(logits, y):
    """Temperature scaling (Guo et al., 2017) on the validation set."""
    t = torch.ones(1, requires_grad=True); opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    yt = torch.from_numpy(y)
    def closure():
        opt.zero_grad(); l = F.cross_entropy(logits / t.clamp(min=0.05), yt); l.backward(); return l
    opt.step(closure)
    return float(t.detach().clamp(min=0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--aug", default="basic", choices=["basic", "photometric"])
    ap.add_argument("--extra", nargs="*", default=[], help="npz files with generated crops X,y")
    ap.add_argument("--train_extra", nargs="*", default=[], help="npz files of extra REAL labelled crops, concatenated to the training set")
    ap.add_argument("--extra_frac", type=float, default=0.5, help="share of each batch drawn from generated crops")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))

    d = dict(np.load("data/crops.npz"))
    tr = np.where(d["split"] == "train")[0]; va = np.where(d["split"] == "val")[0]
    Xtr, ytr = d["X"][tr], d["y"][tr]
    for e in a.train_extra:
        z = np.load(e); Xtr = np.concatenate([Xtr, z["X"]]); ytr = np.concatenate([ytr, z["y"]])
    tr = np.arange(len(ytr))
    ex_X, ex_y = [], []
    for e in a.extra:
        z = np.load(e); ex_X.append(z["X"]); ex_y.append(z["y"])
    if ex_X:
        ex_X = np.concatenate(ex_X); ex_y = np.concatenate(ex_y)
    print(f"[{a.name}] train={len(ytr)} val={len(va)} extra={len(ex_y) if len(ex_X) else 0}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Colab GPU if present
    model = ParkNet().to(dev); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-3)
    steps_per_epoch = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * steps_per_epoch)
    best, best_state = -1, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(len(tr)); tot = 0
        for b in range(steps_per_epoch):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            xs, ys = list(Xtr[idx]), list(ytr[idx])
            if len(ex_X):  # replace a share of the batch with generated crops
                k = int(a.bs * a.extra_frac); j = rng.integers(0, len(ex_y), k)
                xs[:k] = list(ex_X[j]); ys[:k] = list(ex_y[j])
            xs = [geometric(x, rng) for x in xs]
            if a.aug == "photometric":
                xs = [photometric(x, rng) for x in xs]
            x = to_tensor(xs).to(dev); y = torch.tensor(ys).to(dev)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        lv = predict_logits(model, d["X"][va]); acc = (lv.argmax(1).numpy() == d["y"][va]).mean()
        print(f"  ep{ep:02d} loss={tot / steps_per_epoch:.4f} val_acc={acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc >= best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.cpu()
    lv = predict_logits(model, d["X"][va]); T = fit_temperature(lv, d["y"][va])
    logits = predict_logits(model, d["X"]); p = F.softmax(logits / T, 1)[:, 1].numpy()

    out = os.path.join(ROOT, "runs", a.name); os.makedirs(out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out, "model.pt")); np.save(os.path.join(out, "probs.npy"), p)
    rows, pooled = evaluate(d, p)
    meta = dict(name=a.name, aug=a.aug, extra=a.extra, train_extra=a.train_extra, seed=a.seed, epochs=a.epochs, val_acc=float(best),
                temperature=T, params=sum(q.numel() for q in model.parameters()), train_seconds=time.time() - t0,
                rows=rows, pooled=pooled)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    for k, v in pooled.items():
        print(f"  {k:30s} acc={v['acc']:.3f} macroF1={v['macro_f1']:.3f} countMAE={v['count_mae']:.2f}")

    if a.export:
        export(model, T, a.name)


def export(model, T, name):
    import onnx
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
    model.eval(); os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    f32 = os.path.join(ROOT, "models", f"{name}_fp32.onnx")
    torch.onnx.export(model, torch.randn(1, 3, 64, 64), f32, input_names=["crops"], output_names=["logits"],
                      dynamic_axes={"crops": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(f32))
    d = np.load("data/crops.npz"); cal = d["X"][d["split"] == "train"][::25]

    class Reader(CalibrationDataReader):
        def __init__(self): self.it = iter([{"crops": to_tensor(cal[i:i + 32]).numpy()} for i in range(0, len(cal), 32)])
        def get_next(self): return next(self.it, None)
    i8 = os.path.join(ROOT, "models", f"{name}_int8.onnx")
    quantize_static(f32, i8, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    for f in (f32, i8):
        json.dump({"temperature": T, "input": "64x64 RGB, ImageNet mean/std", "classes": ["free", "occupied"]},
                  open(f.replace(".onnx", ".json"), "w"))
        print("exported", f, f"{os.path.getsize(f) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
