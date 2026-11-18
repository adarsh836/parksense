"""Aggregate runs/*/meta.json into results/experiment.md + results/summary.json (mean ± std over seeds)."""
import glob, json, os, collections
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
LABEL = {"baseline_logreg": "A0 Classical baseline (edges + colour)",
         "cnn_basic": "A  CNN, geometric augmentation only",
         "cnn_photo": "B  CNN + classical photometric weather simulation",
         "cnn_gan": "C  CNN + CycleGAN synthetic rain",
         "cnn_gan_photo": "D  CNN + CycleGAN rain + photometric",
         "cnn_diffusion": "E  CNN + diffusion night/fog/rain (Colab)",
         "cnn_multi_basic": "F  CNN, trained on all 3 PKLot parks*",
         "cnn_multi_photo": "G  F + photometric simulation (deployed)*",
         "cnn_multi_diffusion": "H  G + diffusion night/fog/rain (Colab)*"}
GROUPS = ["UFPR04 seen weather", "UFPR04 real rain", "Unseen PKLot parks", "Unseen camera set (CNRPark)", "All unseen rain"]

arms = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(ROOT, "runs", "cnn_*", "meta.json"))):
    m = json.load(open(f)); arms[m["name"].rsplit("_s", 1)[0]].append(m)
base = json.load(open(os.path.join(ROOT, "results", "baseline.json")))["baseline_logreg"]

summary = {"baseline_logreg": {"seeds": 1, "pooled": {g: {k: [base["pooled"][g][k], 0.0] for k in ("acc", "macro_f1", "count_mae")} for g in GROUPS},
                               "rows": base["rows"]}}
for arm, ms in arms.items():
    pooled = {g: {k: [float(np.mean([m["pooled"][g][k] for m in ms])), float(np.std([m["pooled"][g][k] for m in ms]))]
                  for k in ("acc", "macro_f1", "count_mae")} for g in GROUPS}
    rows = []
    for i, r in enumerate(ms[0]["rows"]):
        rows.append({**{k: r[k] for k in ("site", "weather", "n")},
                     **{k: float(np.mean([m["rows"][i][k] for m in ms])) for k in ("acc", "macro_f1", "f1_occ", "count_mae", "count_mae_pct")}})
    summary[arm] = {"seeds": len(ms), "pooled": pooled, "rows": rows,
                    "val_acc": float(np.mean([m["val_acc"] for m in ms])), "temperature": float(np.mean([m["temperature"] for m in ms]))}
json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)

order = [a for a in LABEL if a in summary]
out = ["# ParkSense experiment summary", "",
       "Training data: UFPR04 Sunny + Cloudy training days only. All numbers are on held-out test frames.",
       "*Arm F also trains on UFPR05 and PUCPR (other days), so for F those columns are held-out DAYS, not unseen car parks; CNRPark-EXT stays unseen for every arm.",
       "CNN arms: mean ± std over seeds. Accuracy and macro-F1 per slot; count MAE = mean absolute error in the number of occupied bays per image.", ""]
for metric, nice in [("acc", "Slot accuracy"), ("macro_f1", "Macro-F1"), ("count_mae", "Occupied-count MAE (bays per image)")]:
    out += [f"## {nice}", "", "| Model | seeds | " + " | ".join(GROUPS) + " |", "|---|---|" + "---|" * len(GROUPS)]
    for a in order:
        s = summary[a]
        out.append(f"| {LABEL[a]} | {s['seeds']} | " + " | ".join(
            (f"{s['pooled'][g][metric][0]:.3f} ± {s['pooled'][g][metric][1]:.3f}" if metric != "count_mae" else f"{s['pooled'][g][metric][0]:.2f} ± {s['pooled'][g][metric][1]:.2f}")
            if s["seeds"] > 1 else (f"{s['pooled'][g][metric][0]:.3f}" if metric != "count_mae" else f"{s['pooled'][g][metric][0]:.2f}") for g in GROUPS) + " |")
    out.append("")
out += ["## Per car park and weather (accuracy / macro-F1 / count MAE)", ""]
hdr = "| Car park | Weather | Slots | " + " | ".join(LABEL[a].split("  ")[0] if "  " in LABEL[a] else LABEL[a].split(" ")[0] for a in order) + " |"
out += [hdr, "|---|---|---|" + "---|" * len(order)]
for i, r in enumerate(summary[order[0]]["rows"]):
    cells = []
    for a in order:
        rr = summary[a]["rows"][i]; cells.append(f"{rr['acc']:.3f} / {rr['macro_f1']:.3f} / {rr['count_mae']:.1f}")
    out.append(f"| {r['site']} | {r['weather']} | {r['n']} | " + " | ".join(cells) + " |")
open(os.path.join(ROOT, "results", "experiment.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
