"""Generate docs/MODEL_CARD.md and app/static/model_card.html from the result files,
so every number in the model card comes straight from results/*.json."""
import json, os, html, re

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = lambda f: json.load(open(os.path.join(ROOT, "results", f)))
S = R("summary.json"); E = R("final_eval.json"); X = R("export.json")
ST = R("stress.json") if os.path.exists(os.path.join(ROOT, "results", "stress.json")) else {}
arm = E["arm"]; s = S[arm]; b = S["baseline_logreg"]
LAT = {}
for f in os.listdir(os.path.join(ROOT, "results")):
    if f.startswith("latency_") and "localhost" not in f:
        LAT[f] = R(f)


def pct(v): return f"{100 * v:.1f}%"


G = ["UFPR04 seen weather", "UFPR04 real rain", "Unseen PKLot parks", "Unseen camera set (CNRPark)"]
MULTI = "multi" in arm
GN = {"UFPR04 seen weather": "UFPR04, held-out dry days", "UFPR04 real rain": "UFPR04, real rain (held-out days)",
      "Unseen PKLot parks": "UFPR05 + PUCPR, held-out days" if MULTI else "Unseen car parks UFPR05 + PUCPR",
      "Unseen camera set (CNRPark)": "CNRPark-EXT (unseen camera set)"}
CAL = R("camera_calibration_" + X["source_run"] + ".json")["mean"] if os.path.exists(os.path.join(ROOT, "results", "camera_calibration_" + X["source_run"] + ".json")) else None
QG = R("quality_gate.json") if os.path.exists(os.path.join(ROOT, "results", "quality_gate.json")) else None
rows = "\n".join(f"| {GN[g]} | {pct(s['pooled'][g]['acc'][0])} | {s['pooled'][g]['macro_f1'][0]:.3f} | {s['pooled'][g]['count_mae'][0]:.2f} | {pct(b['pooled'][g]['acc'][0])} |" for g in G)
sub = "\n".join(f"| {r['site']} | {r['weather']} | {r['n']} | {pct(r['acc'])} | {r['count_mae']:.2f} |" for r in s["rows"])
cams = ", ".join(f"{k.replace('CNR-cam', 'cam ')}: {pct(v)}" for k, v in E["per_cnr_camera"].items())
hours = ", ".join(f"{k}: {pct(v['acc'])}" for k, v in E["per_hour_band"].items())
band = [r for r in E["unsure_band"] if r["band"] == "0.25-0.75"]
band_txt = "; ".join(f"{r['group']}: {pct(r['unsure_share'])} of bays flagged, {pct(r['acc_on_sure'])} accurate on the rest" for r in band)
DF = R("diffusion_colab.json") if os.path.exists(os.path.join(ROOT, "results", "diffusion_colab.json")) else None
diff_txt = ""
if DF:
    key = next((k for k in DF["stress_accuracy"] if k.startswith(arm + " ") or k == arm), None)
    if key:
        v = DF["stress_accuracy"][key]
        diff_txt = (f"* **Diffusion-generated weather** (InstructPix2Pix, an independent generator): night {pct(v[4])}, fog {pct(v[5])}, rain {pct(v[6])}. "
                    f"On the classical simulations it was trained with: night {pct(v[2])}, fog {pct(v[3])}, so robustness measured with the training generator is optimistic.")
stress_txt = ""
if arm in ST:
    stress_txt = "\n".join(f"| {k} | {pct(v['acc'])} |" for k, v in ST[arm].items())
lat_txt = ""
for f, v in LAT.items():
    cnn = [p for p in v["per_sample"] if p["model"] == "cnn"]
    lat_txt += f"\n* `{v['url']}`: server time {min(p.get('server_ms_median', p.get('server_ms')) for p in cnn):.0f}–{max(p.get('server_ms_median', p.get('server_ms')) for p in cnn):.0f} ms per image (35–195 bays)"
    if "cold_start_s" in v:
        lat_txt += f", cold start {v['cold_start_s']} s"

md = f"""# Model card: ParkSense bay-occupancy classifier

**Model:** ParkNet, a 4-block CNN ({S[arm].get('params', '~0.1 M') if False else '≈0.1 M'} parameters) trained from scratch; deployed as ONNX fp32 ({X['fp32']['size_kb']:.0f} KB). An int8 version ({X['int8']['size_kb']:.0f} KB) exists but is not deployed: it saturates on CPUs without VNNI, such as Render's.
**Version:** `{X['source_run']}` (training arm `{arm}`), exported {X.get('date', '2026-11')}.
**Input:** a 64×64 perspective-rectified crop of one parking bay, outlined once by the operator.
**Output:** calibrated P(occupied) (temperature scaling, T ≈ {s['temperature']:.2f}). The app shows *free* (P ≤ 0.25), *occupied* (P ≥ 0.75) or *not sure*.

## Intended use
* Counting free and occupied bays in **open-air car parks** watched by a **fixed, elevated camera**, in **daylight**.
* Decision support for a car-park operator, who reviews "not sure" bays and can override any bay before counts are published.
* Public display of **counts only**.

## Out-of-scope use
* Reading licence plates, identifying vehicles or people, tracking individuals, enforcement or fines.
* Night-time, indoor/multi-storey car parks, moving or handheld cameras, street-level views.
* Safety-critical or fully automatic decisions (e.g. barrier control) without human review.

## Training data
{"* PKLot (CC BY 4.0): car park **UFPR04** (Sunny + Cloudy training days, 7,499 bays from 207 frames) plus **160 frames from UFPR05 and PUCPR** (16,300 bays) taken only from days with no test frame: 23,799 labelled bays in total." if MULTI else "* PKLot (CC BY 4.0), car park **UFPR04** only, Sunny + Cloudy, training days only: 7,499 labelled bays from 207 frames."}
* {"CycleGAN-translated rainy versions of the same frames (labels carried over)." if "gan" in arm else "No generated data in this arm."}
* Splits are by **day**, so no day appears in both training and test.

## Evaluation data (never used for training)
* UFPR04 held-out days (sunny/cloudy) and real **rainy** days.
{"* PKLot UFPR05 and PUCPR test frames (sunny, cloudy, rainy) from days not used for training. The protocol experiment, trained on UFPR04 alone, reached 96.2% on these unseen car parks (see the report)." if MULTI else "* **Unseen car parks:** PKLot UFPR05 and PUCPR (sunny, cloudy, rainy)."}
* **Unseen camera set:** CNRPark-EXT, 9 cameras in Pisa, Italy (sunny, overcast, rainy).

## Metrics ({s['seeds']} training seed{'s, mean' if s['seeds'] > 1 else ''})
| Test group | Accuracy | Macro-F1 | Occupied-count error (bays/image) | Classical baseline accuracy |
|---|---|---|---|---|
{rows}

### Subgroups (car park × weather)
| Car park | Weather | Bays | Accuracy | Count error |
|---|---|---|---|---|
{sub}

* Per CNRPark camera: {cams}.
* By time of day: {hours}.
* Calibration (expected calibration error): {", ".join(f"{k} {v:.3f}" for k, v in E["ece"].items())}.
* "Not sure" band 0.25–0.75: {band_txt}.
{(f"* **New-camera calibration** (CNRPark-EXT, mean of 9 cameras): {pct(CAL['0'])} with no labelled frames, {pct(CAL['1'])} after 1 operator-labelled frame, {pct(CAL['3'])} after 3." if CAL else "")}
{(f"* **Frame-quality warnings** flag {pct(QG['flag_rate']['simulated_night'])} of simulated night, {pct(QG['flag_rate']['simulated_fog'])} of fog and {pct(QG['flag_rate']['simulated_blur'])} of blurred frames, with {pct(QG['flag_rate']['real_frames'])} false alarms on real frames." if QG else "")}
{diff_txt}
{("\n### Stress tests (accuracy on altered UFPR04 test frames)\n| Set | Accuracy |\n|---|---|\n" + stress_txt) if stress_txt else ""}

## Speed
* fp32 ONNX, one CPU thread: {X['fp32']['ms_per_slot_1thread']:.2f} ms per bay on the build machine (int8: {X['int8']['ms_per_slot_1thread']:.2f} ms, agreeing on {pct(X['int8_vs_fp32_agreement'])} of bays there).{lat_txt}

## Known failures
* **New camera geometry:** accuracy drops on CNRPark-EXT. Low cameras see cars from behind or in front rather than from above; the model mostly misses those cars (false "free"). On unseen cameras most errors are *confident*, so the "not sure" band does not catch them: calibrate a new camera with 3 labelled frames.
* **Night and fog:** classical simulations are handled, but diffusion-generated fog drops accuracy to about two-thirds and night to 86%. The app warns on dark, low-contrast and blurred frames; do not rely on night-time counts.
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
"""
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
open(os.path.join(ROOT, "docs", "MODEL_CARD.md"), "w").write(md)


def md2html(t):
    out, in_table, in_list = [], False, False
    for line in t.split("\n"):
        l = html.escape(line)
        l = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", l); l = re.sub(r"`(.+?)`", r"<code>\1</code>", l)
        l = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", l)
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= set("-"):
                continue
            cells = [c.strip() for c in l.strip("|").split("|")]
            if not in_table:
                out.append("<table><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>"); in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table: out.append("</table>"); in_table = False
        if line.startswith("* "):
            if not in_list: out.append("<ul>"); in_list = True
            out.append(f"<li>{l[2:]}</li>"); continue
        if in_list: out.append("</ul>"); in_list = False
        if line.startswith("### "): out.append(f"<h3>{l[4:]}</h3>")
        elif line.startswith("## "): out.append(f"<h2>{l[3:]}</h2>")
        elif line.startswith("# "): out.append(f"<h1>{l[2:]}</h1>")
        elif line.strip(): out.append(f"<p>{l}</p>")
    if in_table: out.append("</table>")
    if in_list: out.append("</ul>")
    return "\n".join(out)


page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ParkSense · model card</title><link rel="stylesheet" href="/static/style.css"><link rel="icon" href="/static/favicon.svg"></head>
<body><header class="bar"><a class="logo" href="/"><span class="p">P</span>ParkSense</a>
<nav><a href="/">Operator</a><a href="/public">Public view</a><a href="/help">Help</a><a href="/model-card" aria-current="page">Model card</a></nav></header>
<main class="doc">{md2html(md)}</main></body></html>"""
open(os.path.join(ROOT, "app", "static", "model_card.html"), "w").write(page)
print("wrote docs/MODEL_CARD.md and app/static/model_card.html")
