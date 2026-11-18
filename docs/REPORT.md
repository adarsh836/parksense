# ParkSense: parking occupancy from a fixed camera

**MAIB AI 221 Computer Vision with Applications · Final Group Project · Project 7 (Smart city and transport)**
Student: Adarsh Anilkumar (AM25DXB003) · Live app: https://parksense-oqjk.onrender.com · Repository: https://github.com/adarsh836/parksense

---

## Summary

ParkSense counts free and occupied bays in an open-air car park from one fixed camera. The operator outlines each bay once. After that, every image is cut into perspective-rectified bay crops, which a 376 KB CNN classifies in under a millisecond per bay (0.2–1.2 s per image on Render's free tier). The app runs on Render's free tier (FastAPI + onnxruntime), shows each bay as free, occupied or not sure, plots occupancy over time, and publishes counts (never images) to a public page.

Trained on one Brazilian car park (UFPR04), the CNN reaches 99.5% bay accuracy on unseen days there, 99.8% on real rain it never saw, and 96.2% on two unseen car parks, against 90.2% for a classical edge-and-colour baseline. On a different camera set (CNRPark-EXT, Italy) it drops to 79.4%, *below* the baseline (87.9%). The main cause is viewpoint: the CNN misses cars seen from a low, oblique angle. The deployed model (all three PKLot car parks + photometric augmentation) reaches 81.0% there, and three operator-labelled frames per new camera raise it to 91.2% (Section 4.4).

The generative component has two parts. A CycleGAN that turns dry frames into rain, learned from unlabelled rain footage, preserved labels but learned only a subtle change, because at this camera rain barely changes how a bay looks. An InstructPix2Pix diffusion model on a free Colab GPU produced strong night, fog and rain versions. Retraining with either generator did not improve accuracy on real test data. Their real value was as stress tests: the plain CNN falls to chance in night and fog, and the deployed model (trained with classical photometric augmentation) scores 95–99% on classical simulations but only 86% (night) and 66% (fog) on diffusion-generated frames. The app therefore warns on dark or foggy frames and asks for review.

---

## 1 Problem statement

**User and decision.** The primary user is the operator of an open-air car park who wants live counts of free bays without a sensor in every bay; one camera can watch dozens. The secondary users are drivers, who see a public page of free bays per car park. The decision the app supports is where to send drivers (signage, a web page, an app) and, for the operator, when the car park is effectively full.

**What an error costs.** Errors are not symmetric:

* A **free bay reported as occupied** (false positive) costs a little: a driver may skip a car park that still had space.
* An **occupied bay reported as free** (false negative) costs more: drivers are sent to a full car park and circle, the congestion the system should reduce.
* **Count error** matters more than individual bay errors for the public page. Two opposite errors cancel in the count. But near capacity, a count error of one or two bays decides between "3 free" and "FULL".

So the app reports per-bay states with calibrated probabilities, routes a "not sure" band to the operator, warns on dark, foggy or blurred frames, and the evaluation reports count error next to accuracy.

## 2 Data

**Sources and licences.** PKLot (de Almeida et al., 2015; CC BY 4.0) contains 12,416 frames from three car parks in Curitiba, Brazil: UFPR04 (a low camera, 28–37 bays), UFPR05 (5th floor, 40–46 bays) and PUCPR (10th floor, 100–195 bays). Frames carry a weather label and every bay a free/occupied label. CNRPark-EXT (Amato et al., 2017; ODbL v1.0) has frames from nine cameras of a different car park in Pisa, Italy, under sunny, overcast and rainy weather. Both are redistributed with revised bay polygons in MetaPKLot (DSBD-Research, 2026), from which `training/fetch_data.sh` downloads the frames used.

**Sampling and splits.** To fit a 2-core CPU, the project samples 856 frames, giving 47,172 labelled bays (48.6% occupied). Splits follow three rules.

1. **Train on one car park.** Only UFPR04 sunny and cloudy days are used for training. *Days*, not frames, are split 60/15/25 into train, validation and test (207 / 50 / 90 frames), so frames from one day never appear on both sides.
2. **Rain as an unseen condition.** UFPR04 rainy days are split in half by day. One half (63 frames, the "GAN pool") is used *without labels* to train the image-to-image translator. The other half (74 frames) is the real-rain test set.
3. **Unseen car parks and cameras.** UFPR05 (120 frames), PUCPR (90) and CNRPark-EXT (162, 18 per camera) are test-only, balanced over weather.

Caveat: the UFPR04 rainy test days fall in the December recess, so only 20% of bays are occupied (49% on dry test days).

**Preprocessing (Week 2).** Each bay polygon is reduced to four corners, ordered, and warped by a perspective transform into a 64×64 top-down crop. The long side of the bay is always made vertical, so a car appears "nose up" whatever the camera angle. The app uses the same function (`parksense_core.rectify`) on operator-drawn outlines.

## 3 Methods

### 3.1 Classical baseline (Weeks 2–3)
Eight hand-crafted features per crop, each normalised by the crop's median brightness so they survive lighting changes: Canny edge density with automatic thresholds, edge density in the centre of the bay, Laplacian variance, grey-level standard deviation, mean and spread of saturation, hue spread, and mean gradient magnitude. A logistic regression is trained on UFPR04 training crops (it beat a single edge-density threshold, 90.2% vs 84.2% on unseen PKLot car parks).

### 3.2 Main model: ParkNet (Weeks 6–7)
A small CNN in the spirit of mAlexNet (Amato et al., 2017), who found very small networks generalise well across car parks. It has four convolution blocks (5×5 then 3×3 kernels, BatchNorm, ReLU, max-pooling; 24-48-64-96 channels), global average pooling, dropout 0.3 and a two-way linear layer: about 0.1 M parameters. It is trained from scratch (ImageNet weights were not reachable from the build environment), which the brief allows. Training uses AdamW, a one-cycle learning rate (peak 3e-3), label smoothing 0.05, batches of 128 and 14 epochs. Geometric augmentation (flips, 90° rotations, ±15% scale, ±5 px shift) imitates imprecise operator outlines. Temperature scaling (Guo et al., 2017) is fitted on validation data so the "not sure" band means what it says. Every protocol arm is trained with three seeds and reported as mean ± standard deviation.

The model is exported to ONNX (opset 17) and statically quantised to int8 (QDQ format, per-channel weights, calibration on 300 training crops), as in Lecture D. The int8 model is 108 KB (fp32: 376 KB), agrees with fp32 on 99.86% of test bays on the build machine (test accuracy 95.39% vs 95.40%), and runs in 0.48 ms per bay on one CPU thread (fp32: 0.90 ms). On the deployment server it did not agree (Section 4.6), so the fp32 model is deployed.

### 3.3 Generative component: image-to-image weather translation (Week 8)
**CycleGAN, trained on CPU.** An unpaired CycleGAN (Zhu et al., 2017) maps dry UFPR04 crops (domain A, 7,499 crops from training days) to rainy crops (domain B, 2,269 crops from the GAN-pool days, labels never read). Because the translation keeps scene geometry, every translated crop keeps its original label, giving labelled synthetic rain at no labelling cost.

Getting a usable translator took four iterations, each diagnosed rather than guessed:

| Version | Change | What went wrong | Diagnosis |
|---|---|---|---|
| v1 | full frames, ½ resolution, transposed-conv upsampling | checkerboard artefacts | transposed convolutions; fixed with resize-convolution (Odena et al., 2016) |
| v2 | full frames, ¼ resolution (to fit CPU) | cars dissolved into blobs | a car is only ~15×10 px at this scale |
| v3 | 64×64 bay crops | content blurred and lost at 500 iterations | too little compute for a generator learning from scratch |
| v4 | **residual generator** (output = input + learned change, zero-initialised; SimGAN idea, Shrivastava et al., 2017) | discriminator loss stuck at 0.5, output ≈ input | **InstanceNorm in the discriminator** erased each crop's global brightness and saturation, which are the rain cues |
| final | v4 + discriminator without normalisation, spectral norm, identity weight 2 | trains stably, preserves content | — |

Final configuration: residual ResNet generator (24 channels, 5 blocks), spectral-norm PatchGAN, least-squares loss, cycle λ = 10, identity λ = 2, batches of 4, 2,200 iterations (40 minutes on one CPU thread).

**Diffusion (free Colab T4).** `notebooks/diffusion_weather_colab.ipynb` applies InstructPix2Pix (Brooks et al., 2023) at 768×432 (10 steps, image guidance 1.6, text guidance 7) with the prompts "make it night time…", "make it a thick foggy morning" and "make it raining heavily…", then adds back the frame's high-frequency detail. It covers night and fog, for which PKLot has no real images (its 20:00–21:00 frames are as bright as daytime ones). It translated 30 UFPR04 dry test frames (stress test) and 52 training frames (augmentation) per condition in 14 minutes.

**Experiment design.** The generative component is tested twice:

1. **Translator quality.** Fréchet distance (the FID formula) between dry, synthetic-rain and real-rain crops in ParkNet-feature and colour space, and label preservation: does a trained classifier still agree with the original labels on translated crops?
2. **Augmentation.** Arms on identical test sets: A (geometric augmentation only), B (+ classical photometric weather simulation: brightness, contrast, saturation, synthetic night, fog and rain streaks), C (+ CycleGAN rain: 30% of each batch replaced by translated crops). The question is whether C beats A and the cheap classical alternative B on real rain and unseen car parks.
3. **Stress test.** Accuracy on altered copies of the same UFPR04 test frames: CycleGAN rain, classical night and fog, and diffusion night, fog and rain. Arm E retrains A with diffusion frames (3 seeds); arm H retrains the deployed recipe G with them.

### 3.4 Deployment model and camera calibration
The protocol experiment trains on one car park to measure cross-site generalisation honestly. For the deployed app the training set is widened, following MetaPKLot's leave-one-dataset-out rule: UFPR04 plus 160 extra frames (16,300 bays) from UFPR05 and PUCPR, drawn only from days that contribute no test frame. CNRPark-EXT stays completely unseen. Arm F uses geometric augmentation only, arm G adds the photometric simulation; G is deployed (Section 4.1).

A second human-in-the-loop experiment asks how many operator-labelled frames a *new* camera needs. For each CNRPark camera, k frames are labelled (what the operator produces by correcting bays in the app). The model is fine-tuned for 60 steps on them plus a replay sample of 3,000 original training crops, then tested on the camera's remaining frames (k = 0, 1, 3; three random draws).

### 3.5 Application (Render free tier)
FastAPI with onnxruntime and OpenCV-headless, one worker and one thread, well inside the 512 MB limit. The operator page (vanilla JavaScript canvas) loads an image, lets the operator click four corners per bay, saves and loads layouts as JSON, and calls `/api/predict`. Results are drawn as coloured *and lettered* outlines (F, O, ?), because the green/red pair alone fails a deuteranopia check. A car-park-sign summary, a list of "not sure" bays ordered from least to most certain, and per-bay overrides complete the review loop. Several frames produce an occupancy-over-time chart with the not-sure range shaded. "Publish counts" sends only the numbers to `/public`. Three methods are offered: CNN (default), CNN + baseline average (for a new, uncalibrated camera, Section 4.4) and the baseline.

## 4 Results

All numbers are on held-out test frames. CNN arms are mean ± standard deviation over three seeds.

### 4.1 Main comparison

| Model | UFPR04 dry test days | UFPR04 real rain | Unseen PKLot car parks | Unseen camera set (CNRPark-EXT) |
|---|---|---|---|---|
| A0 Classical baseline | 93.2% / 2.37 | 94.7% / 1.65 | 90.2% / 7.46 | **87.9% / 4.55** |
| A CNN | 99.5 ± 0.1% / 0.19 | 99.8 ± 0.1% / 0.07 | 96.2 ± 0.2% / 3.59 | 79.4 ± 0.2% / 7.95 |
| B CNN + photometric simulation | 98.8 ± 0.1% / 0.45 | 98.0 ± 0.8% / 0.68 | 95.8 ± 0.2% / 3.62 | 75.7 ± 0.0% / 9.60 |
| C CNN + CycleGAN rain | 99.5 ± 0.1% / 0.19 | 99.8 ± 0.1% / 0.07 | 96.4 ± 0.4% / 3.39 | 79.2 ± 0.8% / 8.10 |
| E CNN + diffusion night/fog/rain | 99.0 ± 0.2% / 0.37 | 99.3 ± 0.0% / 0.24 | 95.6 ± 1.0% / 4.11 | 78.9 ± 1.4% / 8.15 |
| F CNN, three PKLot car parks\* | 99.7% / 0.11 | 99.9% / 0.03 | 98.8%\* / 0.92 | 78.8% / 8.33 |
| **G F + photometric (deployed)\*** | 99.7% / 0.10 | 99.4% / 0.20 | 98.4%\* / 1.26 | 81.0% / 7.35 |
| H G + diffusion frames\* | 99.2% | 98.4% | 97.8%\* | 79.3% / 8.14 |

*Cells: bay accuracy / mean absolute error in the number of occupied bays per image. \*For F, G and H, UFPR05 and PUCPR are held-out days of car parks seen in training, not unseen car parks.* Full tables with macro-F1 and every car park × weather subgroup: `results/experiment.md`; figure: `results/figures/arms_macro_f1.png`.

The CNN beats the baseline everywhere in PKLot, halving the count error on unseen PKLot car parks (3.6 vs 7.5 bays per image). On CNRPark-EXT the order flips: the baseline is 8.5 points more accurate and makes 43% smaller count errors. Among the deployment candidates, G beats F on the unseen camera set (81.0% vs 78.8%) and on rainy frames of other car parks (96.3% vs 94.2%), and loses 0.5 points on UFPR04 rain. Combined with its stress-test robustness (below), G is the model deployed. F and G are single-seed runs.

### 4.2 Generative component

**Translator quality.** Label preservation is near perfect. A classifier trained without synthetic data is 99.42% accurate on the original dry test crops and 99.39% on their translations, and its decisions agree on 99.97% of bays. The translation is also gentle: the mean absolute pixel change is 0.49 grey levels. For free bays (the bulk of the real-rain test set), synthetic rain moves slightly toward real rain: the Fréchet distance to real rain falls from 0.140 to 0.132 in CNN-feature space and from 0.68 to 0.60 on colour statistics. For occupied bays it does not (0.56 → 0.59). The real-rain property the translator fails to reproduce is reduced contrast: real rainy crops have 19% lower contrast than dry ones, synthetic ones keep the dry value (`results/gan_quality.json`).

**Why so gentle?** A discriminator trained alone to separate real rainy from dry UFPR04 crops reached only 59–68% held-out accuracy: at this camera, rain barely changes a bay, and arm A, which never saw rain, scores 99.8% on real rain.

**Augmentation.** Arm C is indistinguishable from arm A on every test group (differences under one standard deviation). CycleGAN rain therefore **did not help**, because there was little rain-induced error left to fix. Classical photometric augmentation (B) helped where the shift is larger: on *rainy* frames of unseen car parks it is the best arm (94.7% vs 93.3%; count error 3.27 vs 4.59). But it cost 0.7 points at home and 3.7 points on CNRPark, where its brightness jitter made sunny and cloudy CNRPark frames worse (69.7% and 70.9% vs 77.8% and 78.0%). Diffusion frames (E, H) did not help on real data either: E is slightly below A everywhere, and H is below G on every group (`results/diffusion_colab.json`).

**Stress test** (`results/stress.md`; accuracy on altered copies of the UFPR04 dry test frames):

| Model | Clean | Real rain | CycleGAN rain | Classical night | Classical fog | Diffusion night | Diffusion fog | Diffusion rain |
|---|---|---|---|---|---|---|---|---|
| Baseline | 93.2 | 94.7 | 93.1 | 90.3 | 51.1 | 47.1 | 56.1 | 69.6 |
| A CNN | 99.5 | 99.8 | 99.5 | **49.4** | **55.5** | 49.8 | 60.2 | 52.9 |
| B + photometric | 98.8 | 98.0 | 98.7 | 95.6 | 97.3 | 76.4 | 65.7 | 71.8 |
| C + CycleGAN | 99.5 | 99.8 | 99.5 | 52.5 | 55.8 | 49.8 | 60.4 | 52.7 |
| E + diffusion | 99.0 | 99.3 | – | 49.4 | 65.4 | 91.1 | 79.5 | 81.0 |
| F (three car parks) | 99.7 | 99.9 | 99.7 | 70.7 | 53.2 | 66.6 | 58.0 | 57.0 |
| **G deployed** | 99.7 | 99.4 | 99.6 | 95.1 | 99.2 | **85.9** | **65.9** | 78.9 |
| H = G + diffusion | 99.1 | 98.4 | – | 94.1 | 96.5 | 94.9 | 82.3 | 79.5 |

*Accuracy in %; diffusion columns use 30 of the 90 dry test frames.* CycleGAN rain and real rain cost the models the same (nothing), which supports the translator's realism but confirms its mildness. Night and fog push every CNN without matching augmentation to chance. The cross-generator cells are the informative ones: G, trained on classical simulations, scores 95–99% on them but 86% and 66% on diffusion night and fog; E, trained on diffusion frames, scores 91% on diffusion night but 49% on classical night. **Each augmentation mainly teaches the look of its own generator**, so robustness measured with the training generator is optimistic. Diffusion rain lowers every model (53–81%) while real rain does not, so InstructPix2Pix rain is less realistic than the CycleGAN's and may alter cars; its frames were checked by eye only for a sample (`results/gallery/diffusion_examples.jpg` in the notebook output).

### 4.3 Calibration and the "not sure" state

On UFPR04 the deployed model is well calibrated (expected calibration error 0.006). On unseen car parks and cameras it is overconfident (0.034). With the app's band (0.25 < P < 0.75 = not sure), 2.9% of unseen bays are flagged and accuracy on the rest is 95.6%, against 95.0% overall (`results/figures/unsure_tradeoff.png`, `results/figures/reliability.png`). About 5% of unseen bays are wrong, so most cross-camera errors are *confident* errors that the band cannot catch. This shapes the human-in-the-loop design (Section 6).

### 4.4 New cameras: blending and calibration

Averaging the CNN's and baseline's probabilities raises the deployed model's CNRPark accuracy from 81.0% to 85.8% and cuts count error from 7.4 to 5.5 bays, at a cost of 0.5 points on PKLot. It is offered in the app as "CNN + baseline (for a new camera)".

Few-shot calibration is far more effective. Averaged over the nine CNRPark cameras, the deployed model scores 82.2% with no labelled frames (on the calibration test frames), 86.5% with one and **91.2% with three**, clearly above the baseline (87.9%). Three frames improved all nine cameras; the worst camera (cam 5) rose from 69.4% to 87.4%. One frame can hurt (cam 4: 90.8% → 86.8%; cam 8: 89.3% → 80.8%) because a single frame may contain few occupied bays. The protocol-trained model F shows the same pattern at a lower level (80.9% → 84.7% → 88.4%) (`results/camera_calibration_*.json`).

### 4.5 Subgroups and error analysis

**By car park and weather (deployed model G).** UFPR04 99.4–100% in all weather; UFPR05 97.9–98.8%; PUCPR 97.9–98.9%; CNRPark-EXT 76.7% (overcast), 77.6% (sunny) and 88.8% (rainy). In the protocol arm A, PUCPR sunny is the hardest PKLot subgroup (94.1%, 10.8 bays of 195 wrong per image) because of hard tree shadows. Rain is *easier* than sun for every PKLot car park, because shadows disappear.

**By CNRPark camera** (deployed model): cam 2 93.9%, cam 4 90.8%, cam 8 89.3%, cam 3 82.9%, cam 1 80.5%, cam 6 80.2%, cam 7 79.9%, cam 9 73.4%, cam 5 69.4%. The low cameras (5, 7, 9) look across the car park rather than down on it.

**Confusion** (deployed model). On CNRPark-EXT the errors are one-sided: 1,172 occupied bays called free against 78 free bays called occupied. On the PKLot car parks errors are rare and closer to balanced (PUCPR: 134 vs 144 of 17,550 bays).

**Failure gallery** (`results/gallery/failures.jpg`: the 16 most confident mistakes, each with frame context):

* *Viewpoint.* Cars seen from behind or in front at a low angle fill the crop and look nothing like the top-down cars in training. This is the dominant CNRPark failure.
* *Occlusion by trees.* Bays fully under canopy are labelled occupied in PKLot but are invisible in the image; they should be left out of the layout.
* *Label policy.* Motorbikes in a CNRPark bay labelled free are predicted occupied.
* *Edge-of-frame bays, vehicles straddling two bays, and dark cars in shadow on wet asphalt.*

### 4.6 Speed

On Render's free tier, `/api/predict` takes 155–383 ms of server time for 35–55 bays and 1.2 s for the 195-bay PUCPR frame (median of three calls; 0.7–1.9 s round trip from a browser in Dubai; `results/latency_render.json`), inside the 5-second budget. The server uses 158 MB of memory (limit 512 MB). After 17 idle minutes the free service had spun down, and the first request took about **27 seconds**; opening `/api/health` a minute before a demo avoids this.

**A deployment bug found by measuring.** The first Render run gave different counts from the build machine for the same images (UFPR04 sunny: 18 occupied + 3 not sure instead of 22; PUCPR: 109 + 5 instead of 114). Render's CPU supports AVX2 but not VNNI, and onnxruntime's int8 kernels (unsigned activations × signed weights) can saturate on such CPUs. The deployed model was therefore switched to fp32 (376 KB, 0.9 ms per bay), whose answers on Render match the build machine exactly; `/api/health` now reports the CPU flags.

## 5 Discussion

**What worked.** Perspective rectification plus a tiny CNN is accurate and fast enough that the free tier is not a constraint: a 195-bay frame takes about 0.1 s, far inside the 5-second budget. Day-level splits and fully unseen car parks gave a realistic picture of generalisation, unlike the near-perfect scores of PKLot studies that mix frames from the same days.

**What did not.** Neither generator improved the classifier on real data. For the CycleGAN this is a property of the data: rain changes bays very little at this camera, and the classifier already handles real rain. The diffusion model makes large changes, but its frames teach mostly its own style. The generators were most useful as independent stress tests, which exposed that the deployed model's fog robustness was overstated by the classical simulation.

**What the experiment changes in the product.** (1) Frame-level warnings were recalibrated on data: brightness < 50, relative contrast < 0.2, Laplacian variance < 300. They flag 100% of simulated night, 92% of fog and 88% of blurred frames, with 2% false alarms on real frames (`results/quality_gate.json`). (2) The app recommends checking the first frames from any new camera; three corrected frames lift accuracy on an unseen camera from 82% to 91%. (3) The deployed model (G) is trained on all three PKLot car parks with photometric augmentation, which is the best single trade-off between accuracy on real data and robustness to dark and foggy frames.

## 6 Responsible AI

**Privacy by design.** The service never reads licence plates or identifies vehicles or people. Uploaded images are decoded in memory, classified and discarded. They are not written to disk, logged or returned, and the public page receives only four integers per car park. Operators are told (help page) to point cameras at bays, not entrances or footpaths, and to put up signage about occupancy monitoring.

**Bias and coverage.** The data come from two countries, open-air car parks, daylight and mostly European and Brazilian passenger cars. Subgroup results show where the model is weaker: low camera angles (CNRPark cam 5, 7, 9), tree occlusion, and simulated night and fog. Vans, trucks and motorbikes are handled inconsistently. None of this affects individuals directly, but it affects *which car parks* get accurate counts. A car park with a cheap, low camera is served worse, so calibration should be offered to every operator, not only those who notice the errors.

**Human in the loop.** (1) Calibrated "not sure" bays are listed for review and can be overridden before counts are published. Overrides are marked with an asterisk. (2) Section 4.3 showed the band misses most cross-camera errors, so the design does not rely on it alone. Frame-level warnings catch dark, foggy and blurred frames, and new cameras require a calibration check. (3) The operator, not the model, publishes. The model card lists out-of-scope uses (enforcement, identification, barrier control without review).

**Model card.** `docs/MODEL_CARD.md` (also at `/model-card`) is generated from the result files, so its numbers cannot drift from the experiments.

## 7 Limitations and future work

* **Sample size.** 856 test-and-train frames, not all 12,416. Differences under about one point between arms should not be over-read; F, G and H use a single seed.
* **No real night or fog data.** Night and fog are tested only with generated frames (classical and diffusion), which disagree with each other; real labelled night footage is needed before night-time use.
* **No ImageNet pre-training.** A pre-trained MobileNetV3-Small on Colab is the natural comparison for the CNRPark gap.
* **Calibration is offline.** The operator corrects bays in the app, but fine-tuning runs as a script (`camera_calibration.py`); exporting corrections and fine-tuning from the app is the next step.
* **Static layouts.** If the camera moves, the layout must be redrawn; automatic bay detection (a stretch goal) would remove that.
* **Generators aimed at the wrong gap.** Weather was the brief's target, but the costly gap is camera viewpoint. A CycleGAN trained on unlabelled footage of the new camera (CyCADA-style) would target it.

## 8 Conclusion

ParkSense meets the brief's functional and deployment requirements with a model small enough to run on any free tier. Its headline accuracy at the training car park (99.5%) is not the interesting number. The interesting ones are 79–81% on a camera set it has never seen, 91% after three labelled frames, and 66% in diffusion-generated fog, now caught by a frame-level warning. Both generative augmentations were negative on real data, but as stress tests they revealed what a single simulation hid. Saying so clearly is more useful to an operator than an augmentation chosen because it looked impressive.

## References

Amato, G., Carrara, F., Falchi, F., Gennaro, C., Meghini, C. and Vairo, C. (2017) 'Deep learning for decentralized parking lot occupancy detection', *Expert Systems with Applications*, 72, pp. 327–334.
Brooks, T., Holynski, A. and Efros, A. A. (2023) 'InstructPix2Pix: learning to follow image editing instructions', *CVPR*, pp. 18392–18402.
de Almeida, P. R. L., Oliveira, L. S., Britto, A. S., Silva, E. J. and Koerich, A. L. (2015) 'PKLot – a robust dataset for parking lot classification', *Expert Systems with Applications*, 42(11), pp. 4937–4949.
DSBD-Research (2026) 'MetaPKLot – new challenges and protocols for vision-based parking lot management', *Neural Computing and Applications*. doi:10.1007/s00521-026-12398-0.
Guo, C., Pleiss, G., Sun, Y. and Weinberger, K. Q. (2017) 'On calibration of modern neural networks', *ICML*, pp. 1321–1330.
Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017) 'GANs trained by a two time-scale update rule converge to a local Nash equilibrium', *NeurIPS* (Fréchet Inception Distance).
Hoffman, J. et al. (2018) 'CyCADA: cycle-consistent adversarial domain adaptation', *ICML*, pp. 1989–1998.
Mitchell, M. et al. (2019) 'Model cards for model reporting', *FAccT*, pp. 220–229.
Miyato, T., Kataoka, T., Koyama, M. and Yoshida, Y. (2018) 'Spectral normalization for generative adversarial networks', *ICLR*.
Odena, A., Dumoulin, V. and Olah, C. (2016) 'Deconvolution and checkerboard artifacts', *Distill*. doi:10.23915/distill.00003.
Shrivastava, A. et al. (2017) 'Learning from simulated and unsupervised images through adversarial training', *CVPR*, pp. 2107–2116.
Zhu, J.-Y., Park, T., Isola, P. and Efros, A. A. (2017) 'Unpaired image-to-image translation using cycle-consistent adversarial networks', *ICCV*, pp. 2223–2232.

## Appendix A: reproducibility

All commands are in `README.md`. Hardware: 2-core CPU, no GPU, 7 GB RAM. Training time per CNN run: 10–12 minutes (UFPR04 only) and about 25 minutes (three car parks). CycleGAN: 40 minutes. Seeds: 0, 1, 2. Every per-run metric is in `runs/<run>/meta.json`.
