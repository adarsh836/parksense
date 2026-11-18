"""Shared evaluation helpers: slot metrics per car park x weather, occupancy-count error."""
import numpy as np

SITE_GROUP = lambda s: "CNRPark-EXT" if str(s).startswith("CNR") else str(s)
ORDER = [("UFPR04", "Sunny"), ("UFPR04", "Cloudy"), ("UFPR04", "Rainy"),
         ("UFPR05", "Sunny"), ("UFPR05", "Cloudy"), ("UFPR05", "Rainy"),
         ("PUCPR", "Sunny"), ("PUCPR", "Cloudy"), ("PUCPR", "Rainy"),
         ("CNRPark-EXT", "Sunny"), ("CNRPark-EXT", "Cloudy"), ("CNRPark-EXT", "Rainy")]


def slot_metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    # macro-F1 over the two classes (occupied / free) - robust to class imbalance
    prec0 = tn / max(tn + fn, 1); rec0 = tn / max(tn + fp, 1)
    f10 = 2 * prec0 * rec0 / max(prec0 + rec0, 1e-9)
    return dict(n=int(len(y)), acc=(tp + tn) / max(len(y), 1), f1_occ=f1, macro_f1=(f1 + f10) / 2,
                tp=tp, tn=tn, fp=fp, fn=fn)


def count_error(y, p, image_id, thr=0.5):
    """Per image: |predicted occupied - true occupied|, absolute and as % of slots."""
    errs, pct = [], []
    for i in np.unique(image_id):
        m = image_id == i
        e = abs(int((p[m] >= thr).sum()) - int(y[m].sum()))
        errs.append(e); pct.append(100.0 * e / m.sum())
    return dict(count_mae=float(np.mean(errs)), count_mae_pct=float(np.mean(pct)), images=len(errs))


def evaluate(d, p, mask=None, thr=0.5):
    """d: dict-like with y, site, weather, split, image_id. p: P(occupied) for all crops.
    Returns rows for every (car park, weather) test group plus pooled summaries."""
    y = d["y"]; site = np.array([SITE_GROUP(s) for s in d["site"]]); w = d["weather"]; sp = d["split"]
    test = (sp == "test") if mask is None else mask
    rows = []
    for s, wx in ORDER:
        m = test & (site == s) & (w == wx)
        if m.sum() == 0:
            continue
        r = dict(site=s, weather=wx, **slot_metrics(y[m], p[m], thr), **count_error(y[m], p[m], d["image_id"][m], thr))
        rows.append(r)
    pooled = {}
    for name, m in [("UFPR04 seen weather", test & (site == "UFPR04") & (w != "Rainy")),
                    ("UFPR04 real rain", test & (site == "UFPR04") & (w == "Rainy")),
                    ("Unseen PKLot parks", test & np.isin(site, ["UFPR05", "PUCPR"])),
                    ("Unseen camera set (CNRPark)", test & (site == "CNRPark-EXT")),
                    ("All unseen rain", test & (w == "Rainy") & (site != "UFPR04"))]:
        if m.sum():
            pooled[name] = dict(**slot_metrics(y[m], p[m], thr), **count_error(y[m], p[m], d["image_id"][m], thr))
    return rows, pooled


def fmt_table(rows, keys=("acc", "macro_f1", "count_mae")):
    out = ["| Car park | Weather | Slots | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for r in rows:
        out.append(f"| {r['site']} | {r['weather']} | {r['n']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    return "\n".join(out)
