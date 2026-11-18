# ParkSense: parking occupancy from a fixed camera

MAIB AI 221 Computer Vision with Applications, Final Group Project, **Project 7 (Smart city and transport)**.

ParkSense lets a car-park operator outline the bays in one camera image, then labels every bay in later images as **free**, **occupied** or **not sure**. It shows totals, an occupancy-over-time chart, and a public page that shows **counts only, never images**.

* **Live app:** https://parksense-oqjk.onrender.com (Render free web service; the first request after 15 idle minutes takes about a minute)
* **Main model:** small CNN (ParkNet, ~0.1 M parameters), ONNX fp32 **376 KB** (int8: 108 KB, not deployed because it saturates on Render's non-VNNI CPU); 0.2–1.2 s per image on Render
* **Baseline:** edge-density + colour-variance features with logistic regression
* **Generative AI:** CycleGAN dry→rain image-to-image translation trained on unlabelled rain footage (CPU), plus an InstructPix2Pix night/fog/rain notebook for free Colab
* **Data:** PKLot (CC BY 4.0) and CNRPark-EXT (ODbL v1.0), with bay polygons from MetaPKLot

Headline (bay accuracy, deployed model): 99.7% at the training car park, 98.4% on held-out days of the other PKLot car parks, 81.0% on an unseen camera set (CNRPark-EXT). The protocol model trained on one car park reaches 96.2% on unseen PKLot car parks vs 90.2% for the classical baseline.

Results: [`results/experiment.md`](results/experiment.md) · model card: [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) · report draft: [`docs/REPORT.md`](docs/REPORT.md)

---

## Repository layout

```
app/                    FastAPI back end + HTML/JS front end (deployed to Render)
  main.py               API: /api/predict, /api/publish, /api/lots, pages
  parksense_core.py     shared by app and training: perspective rectification, baseline features, ONNX wrapper
  static/               operator console, public view, help, model card
  samples/              test images (never used for training) + bay layouts + one UFPR05 day
models/                 parksense_cnn_int8.onnx (deployed), fp32 version, baseline.json
training/
  fetch_data.sh         downloads the exact frames used (~250 MB) from MetaPKLot
  prepare_data.py       experiment manifest: day-level splits, unseen car parks
  extract_crops.py      64x64 perspective-rectified bay crops
  baseline.py           classical baseline
  train_cnn.py          CNN training (arms A-E), temperature calibration
  cyclegan.py           generative component: CycleGAN dry->rain, translation of frames
  gen_crops.py          labelled crops from any folder of translated frames
  stress_test.py        accuracy on synthetic rain / night / fog versions of test frames
  summarise.py          mean ± std tables over seeds -> results/experiment.md
  evaluate_final.py     confusion, calibration, 'not sure' trade-off, failure gallery
  export_onnx.py        ONNX export + static int8 quantisation + agreement check
  bench_latency.py      response time of a deployed URL, including the cold start
notebooks/
  diffusion_weather_colab.ipynb   InstructPix2Pix night/fog/rain on a free Colab T4
results/                tables, figures, galleries, JSON outputs
runs/                   per-run metrics (meta.json) and small PyTorch weights
docs/                   model card, report draft
```

## Run the app locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd app && uvicorn main:app --reload
# open http://localhost:8000
```

## Deploy to Render (free, no card)

1. Push this repository to GitHub (public).
2. On render.com, sign in with GitHub, choose **New → Blueprint**, and select the repository. `render.yaml` creates a **free** Python web service.
   (Or **New → Web Service** with build command `pip install -r requirements.txt` and start command `cd app && uvicorn main:app --host 0.0.0.0 --port $PORT`.)
3. Wait for the build, open the URL, and run a sample.
4. Before a demo, open `/api/health` once to wake the service.

Memory use stays well under the 512 MB limit: onnxruntime + OpenCV + FastAPI, a 150 KB model, and no image storage.

## Reproduce the experiments

```bash
pip install -r requirements-train.txt
bash training/fetch_data.sh                    # annotations, manifest, 856 + 160 frames, demo day
python training/extract_crops.py               # 47,172 bay crops
python training/baseline.py
for s in 0 1 2; do
  python training/train_cnn.py --name cnn_basic_s$s --seed $s --epochs 14
  python training/train_cnn.py --name cnn_photo_s$s --seed $s --epochs 14 --aug photometric
done
python training/cyclegan.py train --level crop --bs 4 --minutes 40   # 40 min CPU; minutes on a Colab GPU
python training/cyclegan.py translate --level crop --split train
python training/cyclegan.py translate --level crop --split test
for s in 0 1 2; do
  python training/train_cnn.py --name cnn_gan_s$s --seed $s --epochs 14 --extra data/gen_cyclegan_rain.npz --extra_frac 0.3
done
# deployment model (arm G): all three PKLot car parks (other days) + photometric augmentation
python training/extract_crops.py --splits train_multi --out data/crops_multi.npz
python training/train_cnn.py --name cnn_multi_basic_s0 --epochs 14 --train_extra data/crops_multi.npz
python training/train_cnn.py --name cnn_multi_photo_s0 --epochs 14 --aug photometric --train_extra data/crops_multi.npz
python training/summarise.py
python training/stress_test.py --sets data/stress_cyclegan_rain.npz
python training/gan_quality.py --stress data/stress_cyclegan_rain.npz
python training/camera_calibration.py --run runs/cnn_multi_photo_s0
python training/evaluate_final.py --arm cnn_multi_photo
python training/export_onnx.py --run runs/cnn_multi_photo_s0 --name parksense_cnn
python training/make_docs.py                    # model card from the result files
```

Everything above ran on a 2-core CPU without a GPU. The diffusion arm needs a free Colab T4: open `notebooks/diffusion_weather_colab.ipynb`.

## Layout format

```json
{"image_size": [1280, 720],
 "slots": [{"id": "A1", "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}, ...]}
```
Points are in image pixels and can be in any order around the bay. If a later image has a different resolution, the layout is scaled automatically. The camera must not move.

## Data and licences

| Source | Content used | Licence |
|---|---|---|
| PKLot, de Almeida et al. (2015), *Expert Systems with Applications* 42(11) | frames from UFPR04, UFPR05, PUCPR (sunny, cloudy, rainy) | CC BY 4.0 |
| CNRPark-EXT, Amato et al. (2017), *Expert Systems with Applications* 72 | frames from 9 cameras (sunny, overcast, rainy) | ODbL v1.0 |
| MetaPKLot, DSBD-Research (2026), *Neural Computing and Applications* | revised bay polygons + occupancy labels, file hosting | as the source datasets |
| InstructPix2Pix, Brooks et al. (2023) | Colab notebook only (weights not redistributed) | see model page |

Sample images in `app/samples/` are redistributed under these licences with attribution. Code: MIT licence (see `LICENSE`).

## Responsible AI in one paragraph

ParkSense counts bays. It does not read licence plates, identify vehicles or track people. Uploaded images are processed in memory and never stored, logged or shown on the public page. "Not sure" bays are routed to the operator, who can override any bay before publishing. Accuracy drops on unseen camera set-ups, so a new camera should be checked against a few hand-counted images before its counts are published. See the [model card](docs/MODEL_CARD.md).
