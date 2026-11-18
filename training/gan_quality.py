"""
Measure the image-to-image translator instead of eyeballing it.

1. Domain distance: Frechet distance (FID formula) between crop sets, computed in the
   feature space of an independently trained ParkNet (arm A, seed 0) and on simple colour
   statistics.  Good synthetic rain should sit closer to REAL rain than the dry originals do.
2. Label preservation: the arm-A classifier's accuracy on translated crops vs on the
   original crops (same bays, same labels).  A large drop would mean the translator changes
   occupancy appearance (e.g. erases cars) - i.e. it would be generating wrong labels.

  python training/gan_quality.py --stress data/stress_cyclegan_rain.npz
"""
import argparse, json, os, sys
import numpy as np, torch, cv2
from scipy import linalg

sys.path.insert(0, os.path.dirname(__file__))
from train_cnn import ParkNet, to_tensor  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")


@torch.no_grad()
def feats(model, X, bs=512):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(model.f(to_tensor(X[i:i + bs])).numpy())
    return np.concatenate(out)


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0); s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    cs = np.real(linalg.sqrtm(s1 @ s2))
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * cs))


def colour_stats(X):
    hsv = np.stack([cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in X]).astype(np.float32)
    g = hsv[..., 2]
    return np.stack([hsv[..., 1].mean((1, 2)), hsv[..., 1].std((1, 2)), g.mean((1, 2)), g.std((1, 2))], 1) / 255.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stress", required=True); ap.add_argument("--ref_run", default="runs/cnn_basic_s0")
    a = ap.parse_args(); torch.set_num_threads(int(os.environ.get("THREADS", os.cpu_count())))
    d = np.load("data/crops.npz"); z = np.load(a.stress)
    dry_m = (d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] != "Rainy")
    rain_m = (d["site"] == "UFPR04") & (d["split"] == "test") & (d["weather"] == "Rainy")
    dry, syn, real = d["X"][dry_m], z["X"], d["X"][rain_m]
    assert len(dry) == len(syn) and (d["y"][dry_m] == z["y"]).all()
    m = ParkNet(); m.load_state_dict(torch.load(os.path.join(a.ref_run, "model.pt")))
    # real rain is mostly EMPTY bays (holiday days); compare like with like: free bays only and occupied bays only
    out = {}
    for cls, name in [(0, "free bays"), (1, "occupied bays")]:
        yd, yr = d["y"][dry_m] == cls, d["y"][rain_m] == cls
        Fd, Fs, Fr = feats(m, dry[yd]), feats(m, syn[yd]), feats(m, real[yr])
        Cd, Cs, Cr = colour_stats(dry[yd]), colour_stats(syn[yd]), colour_stats(real[yr])
        out[name] = {"n_dry": int(yd.sum()), "n_real_rain": int(yr.sum()),
                     "FD_features dry->real_rain": frechet(Fd, Fr), "FD_features synthetic->real_rain": frechet(Fs, Fr),
                     "FD_colour dry->real_rain": 1e3 * frechet(Cd, Cr), "FD_colour synthetic->real_rain": 1e3 * frechet(Cs, Cr),
                     "mean saturation dry/synthetic/real": [float(c[:, 0].mean()) for c in (Cd, Cs, Cr)],
                     "mean brightness dry/synthetic/real": [float(c[:, 2].mean()) for c in (Cd, Cs, Cr)],
                     "mean contrast dry/synthetic/real": [float(c[:, 3].mean()) for c in (Cd, Cs, Cr)]}
    T = json.load(open(os.path.join(a.ref_run, "meta.json")))["temperature"]
    with torch.no_grad():
        p_dry = torch.softmax(m(to_tensor(dry)) / T, 1)[:, 1].numpy(); p_syn = torch.softmax(m(to_tensor(syn)) / T, 1)[:, 1].numpy()
    y = d["y"][dry_m]
    out["label_preservation"] = {"acc_on_original": float(((p_dry >= .5) == y).mean()), "acc_on_translated": float(((p_syn >= .5) == y).mean()),
                                 "prediction_agreement": float(((p_dry >= .5) == (p_syn >= .5)).mean()),
                                 "mean_abs_pixel_change": float(np.abs(syn.astype(np.float32) - dry.astype(np.float32)).mean())}
    json.dump(out, open(os.path.join(ROOT, "results", "gan_quality.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
