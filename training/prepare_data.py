"""
ParkSense - step 1: build the experiment manifest from MetaPKLot annotations.

Data sources (both redistributed inside MetaPKLot, github.com/DSBD-Research/MetaPKLot-Dataset):
  * PKLot  (de Almeida et al., 2015)  - CC BY 4.0      - car parks UFPR04, UFPR05, PUCPR
  * CNRPark-EXT (Amato et al., 2017)  - ODbL v1.0      - 9 cameras, a different camera set

Experimental design (cross-site generalisation, Week 6-7 + Week 10):
  * Training car park: UFPR04, Sunny + Cloudy only.  Days (not images) are split
    into train / val / test so no day leaks across splits.
  * UFPR04 Rainy days are split in two halves:
       - "gan_pool": unlabelled rainy footage used ONLY to train the image-to-image
         translator (labels never used) - realistic: rain footage is free, labels are not.
       - "test":     real rainy test images (never seen by any model).
  * UFPR05, PUCPR (PKLot) and CNRPark-EXT cameras: unseen car parks, test only.

Output: data/manifest.csv  (one row per image) and data/spots.jsonl (one row per slot).
"""
import argparse, json, random, collections, csv, os

CLIMATE = {0: "Overcast", 1: "Cloudy", 2: "Sunny", 3: "Rainy"}


def load(ann_dir, name):
    d = json.load(open(os.path.join(ann_dir, f"{name}_spots.json")))
    spots = collections.defaultdict(list)
    for a in d["annotations"]:
        spots[a["image_id"]].append(a)
    return d["images"], spots


def day_of(img):
    return "%04d-%02d-%02d" % tuple(img["date"])


def sample_by_day(images, n_per_day, rng):
    """Spread samples across the day (sorted by time, evenly spaced)."""
    by_day = collections.defaultdict(list)
    for im in images:
        by_day[day_of(im)].append(im)
    out = {}
    for day, ims in by_day.items():
        ims = sorted(ims, key=lambda i: i["time"])
        if len(ims) <= n_per_day:
            out[day] = ims
        else:
            step = len(ims) / n_per_day
            out[day] = [ims[int(k * step + rng.random() * step * 0.5)] for k in range(n_per_day)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/ann")
    ap.add_argument("--removed", default="data/removed_pklot.txt")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    removed = set(l.strip() for l in open(args.removed)) if os.path.exists(args.removed) else set()

    rows, spot_rows = [], []

    def add(im, spots, dataset, site, split, role):
        if im["file_name"].split("/", 1)[-1] in removed:
            return
        weather = CLIMATE[im["climate"]]
        if weather == "Overcast":  # CNRPark calls it OVERCAST; PKLot calls it Cloudy
            weather = "Cloudy"
        rows.append(dict(image_id=im["id"], file=im["file_name"], dataset=dataset, site=site,
                         weather=weather, day=day_of(im), time="%02d:%02d" % tuple(im["time"][:2]),
                         split=split, role=role, width=im["width"], height=im["height"],
                         n_spots=len(spots[im["id"]])))
        for a in spots[im["id"]]:
            spot_rows.append(dict(image_id=im["id"], spot_id=a["id"], label=int(a["category_id"]),
                                  poly=[round(v, 1) for v in a["segmentation"][0]]))

    # ---- training car park: UFPR04 ------------------------------------------------
    ims, spots = load(args.ann, "ufpr04")
    for clim, n_per_day in [(2, 10), (1, 10)]:  # Sunny, Cloudy
        days = sample_by_day([i for i in ims if i["climate"] == clim], n_per_day, rng)
        dl = sorted(days); rng.shuffle(dl)
        n = len(dl); n_tr = round(n * 0.6); n_va = round(n * 0.15)
        for k, d in enumerate(dl):
            split = "train" if k < n_tr else "val" if k < n_tr + n_va else "test"
            for im in days[d]:
                add(im, spots, "PKLot", "UFPR04", split, "labelled")
    rainy = sample_by_day([i for i in ims if i["climate"] == 3], 12, rng)
    dl = sorted(rainy); rng.shuffle(dl)
    for k, d in enumerate(dl):
        role = "gan_pool" if k < len(dl) // 2 else "labelled"
        split = "gan_pool" if role == "gan_pool" else "test"
        for im in rainy[d]:
            add(im, spots, "PKLot", "UFPR04", split, role)

    # ---- unseen PKLot car parks ---------------------------------------------------
    for site, per_weather in [("UFPR05", 40), ("PUCPR", 30)]:
        ims, spots = load(args.ann, site.lower())
        for clim in (2, 1, 3):
            pool = [i for i in ims if i["climate"] == clim]
            for im in rng.sample(pool, min(per_weather, len(pool))):
                add(im, spots, "PKLot", site, "test", "labelled")

    # ---- unseen camera set: CNRPark-EXT -------------------------------------------
    for cam in range(1, 10):
        ims, spots = load(args.ann, f"cnr-camera-{cam}")
        for clim in (2, 0, 3):
            pool = [i for i in ims if i["climate"] == clim]
            for im in rng.sample(pool, min(6, len(pool))):
                add(im, spots, "CNRPark-EXT", f"CNR-cam{cam}", "test", "labelled")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out, "spots.jsonl"), "w") as f:
        for r in spot_rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "files.txt"), "w") as f:
        for r in rows:
            f.write("/" + r["file"] + "\n")

    c = collections.Counter((r["site"] if not r["site"].startswith("CNR") else "CNRPark-EXT", r["weather"], r["split"]) for r in rows)
    for k in sorted(c):
        print(k, c[k])
    print("images:", len(rows), "slots:", len(spot_rows))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
def add_multisite(ann_dir="data/ann", out="data", seed=11, n_frames={"UFPR05": 100, "PUCPR": 60}):
    """Arm F (deployment model): extra TRAINING frames from UFPR05 and PUCPR, drawn only from
    days that contribute no test frame. Appended to the manifest with split='train_multi'."""
    rng = random.Random(seed)
    rows = list(csv.DictReader(open(os.path.join(out, "manifest.csv"))))
    test_days = {(r["site"], r["day"]) for r in rows if r["split"] == "test"}
    new_rows, new_spots = [], []
    for site, n in n_frames.items():
        ims, spots = load(ann_dir, site.lower())
        pool = [i for i in ims if (site, day_of(i)) not in test_days]
        for im in rng.sample(pool, min(n, len(pool))):
            weather = CLIMATE[im["climate"]]
            new_rows.append(dict(image_id=im["id"], file=im["file_name"], dataset="PKLot", site=site, weather=weather,
                                 day=day_of(im), time="%02d:%02d" % tuple(im["time"][:2]), split="train_multi",
                                 role="labelled", width=im["width"], height=im["height"], n_spots=len(spots[im["id"]])))
            for a in spots[im["id"]]:
                new_spots.append(dict(image_id=im["id"], spot_id=a["id"], label=int(a["category_id"]),
                                      poly=[round(v, 1) for v in a["segmentation"][0]]))
    with open(os.path.join(out, "manifest.csv"), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writerows(new_rows)
    with open(os.path.join(out, "spots.jsonl"), "a") as f:
        for r in new_spots: f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "files_multi.txt"), "w") as f:
        for r in new_rows: f.write("/" + r["file"] + "\n")
    print("multi-site training frames:", collections.Counter((r["site"], r["weather"]) for r in new_rows))
