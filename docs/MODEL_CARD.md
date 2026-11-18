# Model card: ParkSense bay-occupancy classifier

**Model:** ParkNet, a 4-block CNN (≈0.1 M parameters) trained from scratch; deployed as ONNX int8 (108 KB).
**Version:** `cnn_multi_photo_s0` (training arm `cnn_multi_photo`), exported 2026-11.
**Input:** a 64×64 perspective-rectified crop of one parking bay, outlined once by the operator.
**Output:** calibrated P(occupied) (temperature scaling, T ≈ 0.47). The app shows *free* (P ≤ 0.25), *occupied* (P ≥ 0.75) or *not sure*.

## Intended use
* Counting free and occupied bays in **open-air car parks** watched by a **fixed, elevated camera**, in **daylight**.
* Decision support for a car-park operator, who reviews "not sure" bays and can override any bay before counts are published.
* Public display of **counts only**.

## Out-of-scope use
* Reading licence plates, identifying vehicles or people, tracking individuals, enforcement or fines.
* Night-time, indoor/multi-storey car parks, moving or handheld cameras, street-level views.
* Safety-critical or fully automatic decisions (e.g. barrier control) without human review.

## Training data
* PKLot (CC BY 4.0): car park **UFPR04** (Sunny + Cloudy training days, 7,499 bays from 207 frames) plus **160 frames from UFPR05 and PUCPR** (16,300 bays) taken only from days with no test frame: 23,799 labelled bays in total.
* No generated data in this arm.
* Splits are by **day**, so no day appears in both training and test.

## Evaluation data (never used for training)
* UFPR04 held-out days (sunny/cloudy) and real **rainy** days.
* PKLot UFPR05 and PUCPR test frames (sunny, cloudy, rainy) from days not used for training. The protocol experiment, trained on UFPR04 alone, reached 96.2% on these unseen car parks (see the report).
* **Unseen camera set:** CNRPark-EXT, 9 cameras in Pisa, Italy (sunny, overcast, rainy).

## Metrics (1 training seed)
| Test group | Accuracy | Macro-F1 | Occupied-count error (bays/image) | Classical baseline accuracy |
|---|---|---|---|---|
| UFPR04, held-out dry days | 99.7% | 0.997 | 0.10 | 93.2% |
| UFPR04, real rain (held-out days) | 99.4% | 0.991 | 0.20 | 94.7% |
| UFPR05 + PUCPR, held-out days | 98.4% | 0.984 | 1.26 | 90.2% |
| CNRPark-EXT (unseen camera set) | 81.0% | 0.808 | 7.35 | 87.9% |

### Subgroups (car park × weather)
| Car park | Weather | Bays | Accuracy | Count error |
|---|---|---|---|---|
| UFPR04 | Sunny | 1830 | 99.5% | 0.18 |
| UFPR04 | Cloudy | 1460 | 100.0% | 0.00 |
| UFPR04 | Rainy | 2666 | 99.4% | 0.20 |
| UFPR05 | Sunny | 1840 | 97.9% | 0.82 |
| UFPR05 | Cloudy | 1840 | 98.8% | 0.57 |
| UFPR05 | Rainy | 1840 | 98.6% | 0.55 |
| PUCPR | Sunny | 5850 | 97.9% | 2.77 |
| PUCPR | Cloudy | 5850 | 98.9% | 1.97 |
| PUCPR | Rainy | 5850 | 98.5% | 1.47 |
| CNRPark-EXT | Sunny | 2196 | 77.6% | 8.59 |
| CNRPark-EXT | Cloudy | 2196 | 76.7% | 9.31 |
| CNRPark-EXT | Rainy | 2196 | 88.8% | 4.13 |

* Per CNRPark camera: cam 1: 80.5%, cam 2: 93.9%, cam 3: 82.9%, cam 4: 90.8%, cam 5: 69.4%, cam 6: 80.2%, cam 7: 79.9%, cam 8: 89.3%, cam 9: 73.4%.
* By time of day: 05-09h: 98.1%, 09-12h: 92.7%, 12-15h: 94.8%, 15-18h: 95.6%, 18-22h: 98.3%.
* Calibration (expected calibration error): UFPR04 test 0.006, Unseen car parks + cameras 0.034.
* "Not sure" band 0.25–0.75: UFPR04 test: 1.0% of bays flagged, 99.8% accurate on the rest; Unseen: 2.9% of bays flagged, 95.6% accurate on the rest.
* **New-camera calibration** (CNRPark-EXT, mean of 9 cameras): 82.2% with no labelled frames, 86.5% after 1 operator-labelled frame, 91.2% after 3.
* **Frame-quality warnings** flag 100.0% of simulated night, 91.7% of fog and 88.0% of blurred frames, with 2.0% false alarms on real frames.

### Stress tests (accuracy on altered UFPR04 test frames)
| Set | Accuracy |
|---|---|
| clean (UFPR04 dry test days) | 99.7% |
| real rain (UFPR04 rainy test days) | 99.4% |
| sim night (classical) | 95.1% |
| sim fog (classical) | 99.2% |
| cyclegan_rain | 99.6% |

## Speed
* int8 ONNX, one CPU thread: 0.48 ms per bay; int8 and fp32 agree on 99.9% of test bays.
* `http://localhost:8000`: server time 26–118 ms per image (28–195 bays)

## Known failures
* **New camera geometry:** accuracy drops on CNRPark-EXT. Low cameras see cars from behind or in front rather than from above; the model mostly misses those cars (false "free"). On unseen cameras most errors are *confident*, so the "not sure" band does not catch them: calibrate a new camera with 3 labelled frames.
* **Night and fog:** simulated night and fog push accuracy to near chance (see stress tests). The app warns on dark, low-contrast and blurred frames.
* **Occlusion:** bays fully under tree canopy cannot be read from the image (remove them from the layout); poles and vehicles straddling two bays cause errors both ways.
* **Dark or shadowed vehicles** on dark, wet asphalt can be missed (false "free").
* **Hard tree shadows** on sunny days: PUCPR sunny is the hardest PKLot subgroup.
* **Label policy:** motorbikes in a bay are counted as occupied; some datasets label them free.
* See `results/gallery/failures.jpg` for the most confident mistakes.

## Ethical considerations
* No personal data is needed or produced. Images are processed in memory and not stored.
* Cameras can capture people and plates incidentally: point them at the bays, use signage, and never publish frames.
* The data come from Brazil and Italy; other countries' bay markings, vehicles and climates are untested.
* Before publishing counts from a new camera, correct its first 3 frames in the app (calibration) and spot-check about 20 frames against a manual count.
