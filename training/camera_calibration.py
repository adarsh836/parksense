"""
Human-in-the-loop camera calibration: how many operator-labelled frames does a NEW camera need?

For each CNRPark-EXT camera (never seen in training): take k labelled frames from that camera
(what an operator produces by correcting bays in the app), fine-tune the model briefly on them
plus a replay sample of the original training crops (to avoid forgetting), and test on the
remaining frames of the same camera.  k = 0 is the model as deployed.  Repeated over 3 draws.

  python training/camera_calibration.py --run runs/cnn_multi_basic_s0
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn import ParkNet, to_tensor, geometric, predict_logits  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")


def finetune(model, X, y, Xr, yr, rng, steps=60, bs=64, lr=5e-4):
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-3)
    model.train()
    for _ in range(steps):
        i = rng.integers(0, len(y), bs // 2); j = rng.integers(0, len(yr), bs // 2)
        xs = [geometric(x, rng) for x in np.concatenate([X[i], Xr[j]])]
        ys = torch.tensor(np.concatenate([y[i], yr[j]]))
        loss = F.cross_entropy(model(to_tensor(xs)), ys, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--ks", nargs="*", type=int, default=[0, 1, 3])
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    d = np.load("data/crops.npz")
    replay = np.where(d["split"] == "train")[0]
    rng0 = np.random.default_rng(0); replay = rng0.choice(replay, 3000, replace=False)
    Xr, yr = d["X"][replay], d["y"][replay]
    state = torch.load(os.path.join(a.run, "model.pt"))
    res = {}
    for cam in [f"CNR-cam{i}" for i in range(1, 10)]:
        m = d["site"] == cam; frames = np.unique(d["image_id"][m])
        res[cam] = {}
        for k in a.ks:
            accs = []
            for rep in range(3 if k else 1):
                rng = np.random.default_rng(100 + rep)
                cal = rng.choice(frames, k, replace=False) if k else np.array([], int)
                test = m & ~np.isin(d["image_id"], cal)
                model = ParkNet(); model.load_state_dict(state)
                if k:
                    cm = m & np.isin(d["image_id"], cal)
                    finetune(model, d["X"][cm], d["y"][cm], Xr, yr, rng)
                # fixed test set per draw: frames not used for calibration
                p = predict_logits(model, d["X"][test]).argmax(1).numpy()
                accs.append(float((p == d["y"][test]).mean()))
            res[cam][k] = float(np.mean(accs))
        print(cam, {k: round(v, 3) for k, v in res[cam].items()}, flush=True)
    summary = {k: float(np.mean([res[c][k] for c in res])) for k in a.ks}
    print("mean over cameras", {k: round(v, 3) for k, v in summary.items()})
    json.dump({"run": a.run, "per_camera": res, "mean": summary}, open(os.path.join(ROOT, "results", f"camera_calibration_{os.path.basename(a.run.rstrip('/'))}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
