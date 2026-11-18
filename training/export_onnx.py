"""Export a trained run to ONNX (fp32 + static int8 QDQ) and check that both agree.
  python training/export_onnx.py --run runs/cnn_gan_s0 --name parksense_cnn
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from train_cnn import ParkNet, export  # noqa
from parksense_core import OccupancyModel  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--name", default="parksense_cnn")
a = ap.parse_args()
m = ParkNet(); m.load_state_dict(torch.load(os.path.join(a.run, "model.pt"))); m.eval()
meta = json.load(open(os.path.join(a.run, "meta.json")))
export(m, meta["temperature"], a.name)

ROOT = os.path.join(os.path.dirname(__file__), "..")
d = np.load("data/crops.npz"); test = d["split"] == "test"
X, y = d["X"][test], d["y"][test]
report = {"source_run": os.path.basename(a.run.rstrip("/"))}
for kind in ("fp32", "int8"):
    path = os.path.join(ROOT, "models", f"{a.name}_{kind}.onnx")
    om = OccupancyModel(path, threads=1)
    t = time.perf_counter(); p = om.predict_proba(list(X)); dt = time.perf_counter() - t
    report[kind] = {"size_kb": round(os.path.getsize(path) / 1024, 1), "test_acc": float(((p >= .5) == y).mean()),
                    "ms_per_slot_1thread": round(1000 * dt / len(X), 4)}
    if kind == "fp32":
        p32 = p
    else:
        report["int8_vs_fp32_agreement"] = float(((p >= .5) == (p32 >= .5)).mean())
json.dump(report, open(os.path.join(ROOT, "results", "export.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
