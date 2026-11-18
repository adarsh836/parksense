"""
Final evaluation of the deployed model (Week 10): confusion per group, calibration,
'not sure' coverage, failure gallery, count-error distribution, comparison figure.

  python training/evaluate_final.py --arm cnn_gan
"""
import argparse, glob, json, os, sys, collections, csv
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_lib import SITE_GROUP, slot_metrics, count_error  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); FIG = os.path.join(RES, "figures"); GAL = os.path.join(RES, "gallery")
for p in (FIG, GAL): os.makedirs(p, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#5d656e", "#e1e5e9"
ARM_COL = {"baseline_logreg": "#8a9199", "cnn_basic": "#1d4e9e", "cnn_photo": "#c07d1c", "cnn_gan": "#7a5195", "cnn_multi_photo": "#d1495b"}
ARM_NAME = {"baseline_logreg": "Classical baseline", "cnn_basic": "CNN", "cnn_photo": "CNN + photometric",
            "cnn_gan": "CNN + CycleGAN rain", "cnn_multi_photo": "CNN, 3 PKLot parks + photometric (deployed)*"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def arm_probs(arm):
    ps = [np.load(f) for f in sorted(glob.glob(os.path.join(ROOT, "runs", f"{arm}_s*", "probs.npy")))]
    return np.mean(ps, 0), len(ps)


def fig_arms(summary):
    groups = ["UFPR04 seen weather", "UFPR04 real rain", "Unseen PKLot parks", "Unseen camera set (CNRPark)"]
    nice = ["UFPR04\nseen weather", "UFPR04\nreal rain", "Unseen PKLot\ncar parks", "Unseen camera set\n(CNRPark-EXT)"]
    arms = [a for a in ARM_COL if a in summary]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=150)
    w = 0.8 / len(arms)
    for k, a in enumerate(arms):
        m = [summary[a]["pooled"][g]["macro_f1"][0] for g in groups]
        s = [summary[a]["pooled"][g]["macro_f1"][1] for g in groups]
        x = np.arange(len(groups)) + (k - (len(arms) - 1) / 2) * w
        ax.errorbar(x, m, yerr=s if summary[a]["seeds"] > 1 else None, fmt="o", ms=8, color=ARM_COL[a], label=ARM_NAME[a],
                    ecolor=INK, elinewidth=1, capsize=3, mec="white", mew=1.5, zorder=3)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(nice)
    for k in range(1, len(groups)): ax.axvline(k - 0.5, color=GRID, lw=1)
    ax.set_ylim(0.72, 1.005); ax.set_ylabel("Macro-F1 (bay level)")
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=9)
    ax.set_title("Trained on UFPR04 only (*deployed model: all 3 PKLot parks, other days). Error bars: std over 3 seeds", fontsize=9, color=MUTED, pad=34)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "arms_macro_f1.png")); plt.close(fig)


def reliability(y, p, name):
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    conf = np.where(p >= .5, p, 1 - p); correct = ((p >= .5) == y)
    cb = np.clip(np.digitize(conf, np.linspace(.5, 1, 6)) - 1, 0, 4)
    acc_b = [correct[cb == b].mean() if (cb == b).any() else np.nan for b in range(5)]
    conf_b = [conf[cb == b].mean() if (cb == b).any() else np.nan for b in range(5)]
    n_b = [(cb == b).sum() for b in range(5)]
    ece = np.nansum([n * abs(a - c) for n, a, c in zip(n_b, acc_b, conf_b)]) / len(y)
    return conf_b, acc_b, n_b, float(ece)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); a = ap.parse_args()
    d = dict(np.load(os.path.join(ROOT, "data", "crops.npz")))
    man = {int(r["image_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv")))}
    summary = json.load(open(os.path.join(RES, "summary.json")))
    fig_arms(summary)

    p, nseeds = arm_probs(a.arm)
    y = d["y"]; site = np.array([SITE_GROUP(s) for s in d["site"]]); test = d["split"] == "test"
    out = {"arm": a.arm, "seeds_averaged_for_analysis": nseeds}

    # ---- calibration (reliability) on in-site vs unseen
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    ax.plot([.5, 1], [.5, 1], color=GRID, lw=2, label="Perfect calibration")
    for gname, m, col in [("UFPR04 test", test & (site == "UFPR04"), "#1d4e9e"),
                          ("Unseen car parks + cameras", test & (site != "UFPR04"), "#d1495b")]:
        cb, ab, nb, ece = reliability(y[m], p[m], gname)
        ax.plot(cb, ab, marker="o", ms=6, lw=2, color=col, label=f"{gname} (ECE {ece:.3f})")
        out.setdefault("ece", {})[gname] = ece
    ax.set_xlabel("Model confidence"); ax.set_ylabel("Observed accuracy"); ax.set_xlim(.5, 1); ax.set_ylim(.5, 1.01)
    ax.grid(True, color=GRID); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reliability.png")); plt.close(fig)

    # ---- 'not sure' band: coverage vs accuracy on the remaining slots
    rows = []
    for band in [0.0, 0.1, 0.2, 0.25, 0.3, 0.4]:
        lo, hi = 0.5 - band, 0.5 + band
        for gname, m in [("UFPR04 test", test & (site == "UFPR04")), ("Unseen", test & (site != "UFPR04"))]:
            sure = m & ((p <= lo) | (p >= hi))
            rows.append(dict(band=f"{lo:.2f}-{hi:.2f}", group=gname, unsure_share=float(1 - sure.sum() / m.sum()),
                             acc_on_sure=float(((p[sure] >= .5) == y[sure]).mean())))
    out["unsure_band"] = rows
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=150)
    for gname, col in [("UFPR04 test", "#1d4e9e"), ("Unseen", "#d1495b")]:
        rr = [r for r in rows if r["group"] == gname]
        ax.plot([100 * r["unsure_share"] for r in rr], [r["acc_on_sure"] for r in rr], marker="o", ms=6, lw=2, color=col, label=gname)
        r25 = [r for r in rr if r["band"] == "0.25-0.75"][0]
        ax.annotate("app setting\n0.25–0.75", (100 * r25["unsure_share"], r25["acc_on_sure"]), textcoords="offset points",
                    xytext=(8, -22), fontsize=8, color=MUTED)
    ax.set_xlabel("Slots flagged 'not sure' (%)"); ax.set_ylabel("Accuracy on the remaining slots")
    ax.grid(True, color=GRID); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "unsure_tradeoff.png")); plt.close(fig)

    # ---- confusion + count error per car park
    conf = {}
    for s in ["UFPR04", "UFPR05", "PUCPR", "CNRPark-EXT"]:
        m = test & (site == s); r = slot_metrics(y[m], p[m]); ce = count_error(y[m], p[m], d["image_id"][m])
        conf[s] = {"TN (free→free)": r["tn"], "FP (free→occupied)": r["fp"], "FN (occupied→free)": r["fn"],
                   "TP (occupied→occupied)": r["tp"], "acc": r["acc"], **ce}
    out["confusion"] = conf
    # per CNRPark camera (subgroup) - occlusion by trees differs per camera
    out["per_cnr_camera"] = {}
    for cam in sorted(set(d["site"][site == "CNRPark-EXT"])):
        m = test & (d["site"] == cam); out["per_cnr_camera"][cam] = slot_metrics(y[m], p[m])["acc"]
    # time of day subgroup (low sun / early morning)
    hours = np.array([int(man[int(i)]["time"][:2]) for i in d["image_id"]])
    out["per_hour_band"] = {}
    for lo, hi in [(5, 9), (9, 12), (12, 15), (15, 18), (18, 22)]:
        m = test & (hours >= lo) & (hours < hi)
        out["per_hour_band"][f"{lo:02d}-{hi:02d}h"] = {"acc": float(((p[m] >= .5) == y[m]).mean()), "n": int(m.sum())}

    # ---- failure gallery: most confident mistakes, shown with context from the full frame
    spot_poly = {}
    for l in open(os.path.join(ROOT, "data", "spots.jsonl")):
        s = json.loads(l); spot_poly[s["spot_id"]] = s["poly"]
    err = np.where(test & ((p >= .5) != y))[0]
    err = err[np.argsort(-np.abs(p[err] - .5))]
    tiles, notes = [], []
    per_site = collections.Counter()
    for i in err:
        sg = site[i]
        if per_site[sg] >= 4: continue
        per_site[sg] += 1
        r = man[int(d["image_id"][i])]; img = cv2.imread(os.path.join(ROOT, "data", "images", r["file"]))
        poly = np.array(spot_poly[int(d["spot_id"][i])]).reshape(-1, 2)
        x0, y0 = poly.min(0); x1, y1 = poly.max(0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; h = max(x1 - x0, y1 - y0) * 1.6
        X0, Y0 = int(max(cx - h, 0)), int(max(cy - h, 0)); X1, Y1 = int(min(cx + h, img.shape[1])), int(min(cy + h, img.shape[0]))
        ctxim = img.copy(); cv2.polylines(ctxim, [poly.astype(np.int32)], True, (0, 220, 255), 2)
        t = cv2.resize(ctxim[Y0:Y1, X0:X1], (180, 180))
        t = np.vstack([t, cv2.resize(d["X"][i], (180, 180))])
        lab = f"{sg} {r['weather']}"; lab2 = f"true {'occ' if y[i] else 'free'}, p_occ={p[i]:.2f}"
        cv2.rectangle(t, (0, 0), (180, 34), (40, 40, 40), -1)
        cv2.putText(t, lab, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(t, lab2, (4, 29), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t); notes.append(dict(site=sg, weather=r["weather"], file=r["file"], spot_id=int(d["spot_id"][i]),
                                           true="occupied" if y[i] else "free", p_occupied=float(p[i])))
        if len(tiles) == 16: break
    rows_img = [np.hstack(tiles[k:k + 8]) for k in range(0, len(tiles), 8) if len(tiles[k:k + 8]) == 8]
    if rows_img:
        cv2.imwrite(os.path.join(GAL, "failures.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 88])
    out["failures"] = notes
    json.dump(out, open(os.path.join(RES, "final_eval.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()
