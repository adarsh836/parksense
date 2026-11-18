"""
ParkSense - step 2: perspective-rectify every labelled slot into a 64x64 crop.
Writes data/crops.npz with X (N,64,64,3 uint8 BGR), y, image_id, and per-crop metadata.
"""
import argparse, csv, json, os, sys, collections
import numpy as np, cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from parksense_core import rectify, CROP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--splits", nargs="*", default=None, help="only these splits (default: all except train_multi)")
    ap.add_argument("--out", default="data/crops.npz")
    args = ap.parse_args()
    man = {int(r["image_id"]): r for r in csv.DictReader(open(os.path.join(args.data, "manifest.csv")))
           if (r["split"] in args.splits if args.splits else r["split"] != "train_multi")}
    spots = collections.defaultdict(list)
    for l in open(os.path.join(args.data, "spots.jsonl")):
        s = json.loads(l); spots[s["image_id"]].append(s)

    X, y, iid, sid = [], [], [], []
    for k, (im_id, r) in enumerate(man.items()):
        img = cv2.imread(os.path.join(args.images, r["file"]))
        if img is None:
            print("missing", r["file"]); continue
        for s in spots[im_id]:
            X.append(rectify(img, s["poly"], CROP)); y.append(s["label"]); iid.append(im_id); sid.append(s["spot_id"])
        if k % 100 == 0:
            print(k, len(X), flush=True)
    X = np.stack(X); y = np.array(y, np.int64); iid = np.array(iid, np.int64)
    meta = {c: np.array([man[i][c] for i in iid]) for c in ("site", "weather", "split", "dataset", "day")}
    np.savez(args.out, X=X, y=y, image_id=iid, spot_id=np.array(sid), **meta)
    print("saved", X.shape, "occupied share", y.mean().round(3))


if __name__ == "__main__":
    main()
