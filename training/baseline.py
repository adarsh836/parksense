"""
ParkSense - classical baseline (Weeks 2-3): edge density + colour variance features
with a logistic-regression decision, trained on UFPR04 train days only.
Also reports a zero-training variant: a single edge-density threshold.
Saves models/baseline.json (used by the app for side-by-side comparison).
"""
import json, os, sys
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(__file__))
from parksense_core import baseline_features, BASELINE_FEATURES  # noqa
from evaluate_lib import evaluate, fmt_table  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")

d = dict(np.load("data/crops.npz"))
print("computing features for", len(d["y"]), "crops")
F = np.stack([baseline_features(x) for x in d["X"]])
np.save("data/baseline_feats.npy", F)
tr = d["split"] == "train"; va = d["split"] == "val"

mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
clf = LogisticRegression(C=1.0, max_iter=2000).fit((F[tr] - mu) / sd, d["y"][tr])
p = clf.predict_proba((F - mu) / sd)[:, 1]
print("val acc", ((p[va] >= .5) == d["y"][va]).mean().round(4))

# single-feature rule: edge density > t (t chosen on val)
ts = np.linspace(0, 0.3, 301)
accs = [((F[va, 0] > t) == d["y"][va]).mean() for t in ts]
t_best = float(ts[int(np.argmax(accs))])
p_rule = (F[:, 0] > t_best).astype(float)

json.dump(dict(features=BASELINE_FEATURES, mean=mu.tolist(), std=sd.tolist(), coef=clf.coef_[0].tolist(),
               intercept=float(clf.intercept_[0]), edge_threshold=t_best),
          open(os.path.join(ROOT, "models", "baseline.json"), "w"), indent=1)

res = {}
for name, pp in [("baseline_logreg", p), ("baseline_edge_rule", p_rule)]:
    rows, pooled = evaluate(d, pp)
    res[name] = dict(rows=rows, pooled=pooled)
    print("\n##", name); print(fmt_table(rows))
    for k, v in pooled.items():
        print(f"  {k:30s} acc={v['acc']:.3f} macroF1={v['macro_f1']:.3f} countMAE={v['count_mae']:.2f}")
np.save("data/p_baseline.npy", p)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump(res, open(os.path.join(ROOT, "results", "baseline.json"), "w"), indent=1)
