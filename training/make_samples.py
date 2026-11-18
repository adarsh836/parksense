"""Copy a few TEST images (never used for training) + their bay layouts into app/samples,
plus one full day of UFPR05 frames for the occupancy-over-time demo."""
import csv, json, os, shutil, collections, sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "app", "samples")
os.makedirs(OUT, exist_ok=True)

man = list(csv.DictReader(open("data/manifest.csv")))
spots = collections.defaultdict(list)
for l in open("data/spots.jsonl"):
    s = json.loads(l); spots[s["image_id"]].append(s)


def layout_for(polys, w, h):
    """Number bays row by row (top to bottom, left to right) so ids are stable per camera."""
    cs = [(sum(p[1::2]) / (len(p) / 2), sum(p[0::2]) / (len(p) / 2), p) for p in polys]
    cs.sort(key=lambda c: (round(c[0] / 40), c[1]))
    return {"image_size": [w, h], "slots": [{"id": str(i + 1), "points": [[c[2][k], c[2][k + 1]] for k in range(0, len(c[2]), 2)]}
                                            for i, c in enumerate(cs)]}


def pick(site, weather, name):
    rows = [r for r in man if r["site"] == site and r["weather"] == weather and r["split"] == "test"]
    # choose a busy-but-not-full frame: occupancy closest to 60 %
    def occ(r):
        s = spots[int(r["image_id"])]; return sum(x["label"] for x in s) / max(len(s), 1)
    r = min(rows, key=lambda r: abs(occ(r) - 0.6))
    base = f"{site}_{weather}".lower().replace("-", "")
    shutil.copy(os.path.join("data/images", r["file"]), os.path.join(OUT, base + ".jpg"))
    s = spots[int(r["image_id"])]
    L = layout_for([x["poly"] for x in s], int(r["width"]), int(r["height"]))
    json.dump(L, open(os.path.join(OUT, base + "_layout.json"), "w"))
    return {"name": name, "image": f"/samples/{base}.jpg", "layout": f"/samples/{base}_layout.json",
            "source": r["file"], "true_occupied": sum(x["label"] for x in s), "total": len(s)}


samples = [
    pick("UFPR04", "Sunny", "UFPR04 · sunny (training car park, unseen day)"),
    pick("UFPR04", "Rainy", "UFPR04 · real rain (unseen weather)"),
    pick("UFPR05", "Cloudy", "UFPR05 · cloudy (unseen car park)"),
    pick("PUCPR", "Rainy", "PUCPR · rain, 195 bays (unseen car park)"),
    pick("CNR-cam5", "Sunny", "CNRPark-EXT camera 5 · sunny, low camera (known failure)"),
    pick("CNR-cam1", "Rainy", "CNRPark-EXT camera 1 · rain, trees in view"),
]

# --- sequence: UFPR05, Monday 2013-03-11, 16 frames across the day ---------------------
ann = json.load(open("data/ann/ufpr05_spots.json"))
ids = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/seq_ids.json"))
imgs = {i["id"]: i for i in ann["images"]}
by = collections.defaultdict(list)
for a in ann["annotations"]:
    if a["image_id"] in ids:
        by[a["image_id"]].append(a)
seq_dir = os.path.join(OUT, "ufpr05_day"); os.makedirs(seq_dir, exist_ok=True)
frames = []
for k, i in enumerate(ids):
    im = imgs[i]; fn = os.path.basename(im["file_name"])
    shutil.copy(os.path.join("data/images", im["file_name"]), os.path.join(seq_dir, fn))
    frames.append({"image": f"/samples/ufpr05_day/{fn}", "time": "%02d:%02d" % tuple(im["time"][:2]),
                   "true_occupied": sum(a["category_id"] for a in by[i]), "total": len(by[i])})
L = layout_for([a["segmentation"][0] for a in by[ids[3]]], 1280, 720)
json.dump(L, open(os.path.join(seq_dir, "layout.json"), "w"))
seq = [{"name": "UFPR05, Monday 11 March 2013", "layout": "/samples/ufpr05_day/layout.json", "frames": frames}]

json.dump({"samples": samples, "sequences": seq,
           "licence": "PKLot images: CC BY 4.0 (de Almeida et al., 2015). CNRPark-EXT images: ODbL v1.0 (Amato et al., 2017). "
                      "Bay polygons: MetaPKLot (DSBD-Research)."},
          open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(json.dumps(samples, indent=1)); print(len(frames), "sequence frames")
