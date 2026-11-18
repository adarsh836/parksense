#!/usr/bin/env bash
# Download exactly the data ParkSense uses (~250 MB) from the MetaPKLot repository,
# which redistributes PKLot (CC BY 4.0) and CNRPark-EXT (ODbL v1.0) with bay polygons.
#   bash training/fetch_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data ext
if [ ! -d ext/meta/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --no-checkout \
      https://github.com/DSBD-Research/MetaPKLot-Dataset ext/meta
fi
cd ext/meta
git sparse-checkout init --no-cone
printf '/annotations/original/spots/PKLot/\n/annotations/original/spots/CNRPark-EXT/\n/removed_images/\n' > .git/info/sparse-checkout
git -c gc.auto=0 checkout HEAD
cd ../..
mkdir -p data/ann
for f in ext/meta/annotations/original/spots/*/*.tar.xz; do tar xJf "$f" -C data/ann; done
cp ext/meta/removed_images/pklot.txt data/removed_pklot.txt
python training/prepare_data.py --ann data/ann --removed data/removed_pklot.txt --out data
# + 160 training frames from UFPR05/PUCPR on days without test frames (deployment model, arms F/G)
python -c "import sys; sys.path.insert(0, 'training'); import prepare_data as P; P.add_multisite()"
# UFPR05 demo day used by the app (Monday 2013-03-11)
python - <<'EOF'
import json
d = json.load(open("data/ann/ufpr05_spots.json"))
ims = sorted([i for i in d["images"] if tuple(i["date"]) == (2013, 3, 11)], key=lambda i: i["time"])
pick = [ims[int(k * len(ims) / 16)] for k in range(16)]
open("data/seq_files.txt", "w").write("\n".join("/" + i["file_name"] for i in pick) + "\n")
json.dump([i["id"] for i in pick], open("data/seq_ids.json", "w"))
EOF
cat data/files.txt data/files_multi.txt data/seq_files.txt >> ext/meta/.git/info/sparse-checkout
(cd ext/meta && git -c gc.auto=0 checkout HEAD)
ln -sfn "$(pwd)/ext/meta" data/images
echo "Data ready: $(wc -l < data/files.txt) frames listed in data/manifest.csv"
