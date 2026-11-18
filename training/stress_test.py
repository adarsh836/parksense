"""
Stress test: how do the trained models behave on translated versions of the SAME
UFPR04 test frames?  Each stress set is compared with the clean originals.

Built-in sets (made here, deterministic):
  sim_night, sim_fog  - classical photometric simulations (no generative model)
Generated sets passed with --sets (e.g. data/stress_cyclegan_rain.npz, diffusion sets).

  python training/stress_test.py --sets data/stress_cyclegan_rain.npz
"""
import argparse, glob, json, os, sys, collections
import numpy as np, cv2, torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from train_cnn import ParkNet, predict_logits  # noqa
from evaluate_lib import slot_metrics, count_error  # noqa
from parksense_core import BaselineModel  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")


def sim_night(x, rng):
    x = x.astype(np.float32) * 0.22 * np.array([1.25, 1.0, 0.8]) + rng.normal(0, 5, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def sim_fog(x, rng):
    return np.clip(x.astype(np.float32) * 0.45 + 0.55 * 205, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=None, help="run-name prefixes, default: all arms in runs/")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    d = np.load("data/crops.npz")
    clean_mask = (d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] != "Rainy")
    real_rain = (d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] == "Rainy")
    Xc, yc, ic = d["X"][clean_mask], d["y"][clean_mask], d["image_id"][clean_mask]
    rng = np.random.default_rng(0)
    sets = {"clean (UFPR04 dry test days)": (Xc, yc, ic),
            "real rain (UFPR04 rainy test days)": (d["X"][real_rain], d["y"][real_rain], d["image_id"][real_rain]),
            "sim night (classical)": (np.stack([sim_night(x, rng) for x in Xc]), yc, ic),
            "sim fog (classical)": (np.stack([sim_fog(x, rng) for x in Xc]), yc, ic)}
    for s in a.sets:
        z = np.load(s); name = os.path.basename(s).replace(".npz", "").replace("stress_", "")
        sets[name] = (z["X"], z["y"], z["image_id"])

    runs = sorted(glob.glob(os.path.join(ROOT, "runs", "cnn_*", "model.pt")))
    arms = collections.defaultdict(list)
    for r in runs:
        name = os.path.basename(os.path.dirname(r)); arm = name.rsplit("_s", 1)[0]
        if a.arms is None or any(arm.startswith(p) for p in a.arms):
            arms[arm].append(os.path.dirname(r))
    res = {}
    bl = BaselineModel(os.path.join(ROOT, "models", "baseline.json"))
    for sname, (X, y, ii) in sets.items():
        p = bl.predict_proba(list(X))
        res.setdefault("baseline_logreg", {})[sname] = {**slot_metrics(y, p), **count_error(y, p, ii)}
    for arm, dirs in arms.items():
        per = collections.defaultdict(list)
        for dd in dirs:
            m = ParkNet(); m.load_state_dict(torch.load(os.path.join(dd, "model.pt"))); T = json.load(open(os.path.join(dd, "meta.json")))["temperature"]
            for sname, (X, y, ii) in sets.items():
                p = torch.softmax(predict_logits(m, X) / T, 1)[:, 1].numpy()
                per[sname].append({**slot_metrics(y, p), **count_error(y, p, ii)})
        res[arm] = {s: {k: float(np.mean([v[k] for v in vs])) for k in ("acc", "macro_f1", "count_mae")} |
                       {"acc_std": float(np.std([v["acc"] for v in vs])), "seeds": len(vs)} for s, vs in per.items()}
        print(arm, {s: round(v["acc"], 3) for s, v in res[arm].items()}, flush=True)
    json.dump(res, open(os.path.join(ROOT, "results", "stress.json"), "w"), indent=1)

    names = list(sets)
    lines = ["| Model | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for arm, v in res.items():
        lines.append(f"| {arm} | " + " | ".join(f"{v[s]['acc']:.3f}" for s in names) + " |")
    open(os.path.join(ROOT, "results", "stress.md"), "w").write("Slot accuracy on stress sets (mean over seeds)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
