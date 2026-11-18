"""
Turn a folder of translated frames (CycleGAN or diffusion) into labelled slot crops.
A translated frame keeps the scene geometry, so the original bay polygons and labels apply.

Frames must be saved under the ORIGINAL relative file name, e.g.
  data/gen_diffusion/night/PKLot/UFPR04/Sunny/2012-12-11/2012-12-11_10_05_00.jpg

  python training/gen_crops.py --frames data/gen_diffusion/night --out data/gen_diffusion_night.npz
"""
import argparse, csv, json, os, sys, collections
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from parksense_core import rectify  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = {r["file"]: r for r in csv.DictReader(open("data/manifest.csv"))}
    spots = collections.defaultdict(list)
    for l in open("data/spots.jsonl"):
        s = json.loads(l); spots[s["image_id"]].append(s)
    X, y, iid, site, weather = [], [], [], [], []
    for dp, _, fs in os.walk(a.frames):
        for f in fs:
            rel = os.path.relpath(os.path.join(dp, f), a.frames).replace(os.sep, "/")
            r = man.get(rel)
            if r is None:
                continue
            img = cv2.imread(os.path.join(dp, f))
            H, W = int(r["height"]), int(r["width"])
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
            for s in spots[int(r["image_id"])]:
                X.append(rectify(img, s["poly"])); y.append(s["label"]); iid.append(int(r["image_id"]))
                site.append(r["site"]); weather.append(r["weather"])
    np.savez(a.out, X=np.stack(X), y=np.array(y), image_id=np.array(iid), site=np.array(site), weather=np.array(weather))
    print("saved", a.out, len(y), "crops from", len(set(iid)), "frames")


if __name__ == "__main__":
    main()
