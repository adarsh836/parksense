"""
Response time of a running ParkSense service, including the cold start.

  python training/bench_latency.py --url https://<your-service>.onrender.com --cold
  python training/bench_latency.py --url http://localhost:8000

--cold: call this after the service has been idle for > 15 minutes (Render spins it down);
the first request's time is reported as the cold start.
Only the standard library is used, so it runs anywhere.
"""
import argparse, json, os, statistics, time, urllib.request, uuid

ROOT = os.path.join(os.path.dirname(__file__), "..")
SAMPLES = os.path.join(ROOT, "app", "samples")


def post_predict(url, img_path, layout_path, model="cnn"):
    b = uuid.uuid4().hex
    parts = []
    def field(name, value, filename=None, ctype=None):
        h = f'--{b}\r\nContent-Disposition: form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "") + "\r\n"
        if ctype: h += f"Content-Type: {ctype}\r\n"
        parts.append(h.encode() + b"\r\n" + value + b"\r\n")
    field("image", open(img_path, "rb").read(), "frame.jpg", "image/jpeg")
    field("layout", open(layout_path, "rb").read())
    field("model", model.encode())
    body = b"".join(parts) + f"--{b}--\r\n".encode()
    req = urllib.request.Request(url + "/api/predict", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t = time.perf_counter(); r = json.loads(urllib.request.urlopen(req, timeout=180).read()); dt = time.perf_counter() - t
    return dt, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--cold", action="store_true")
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args(); url = a.url.rstrip("/")
    out = {"url": url, "when": time.strftime("%Y-%m-%d %H:%M:%S")}
    if a.cold:
        t = time.perf_counter(); urllib.request.urlopen(url + "/api/health", timeout=300).read()
        out["cold_start_s"] = round(time.perf_counter() - t, 2)
        print("cold start", out["cold_start_s"], "s")
    idx = json.load(open(os.path.join(SAMPLES, "index.json")))
    out["per_sample"] = []
    for s in idx["samples"]:
        img = os.path.join(SAMPLES, os.path.basename(s["image"])); lay = os.path.join(SAMPLES, os.path.basename(s["layout"]))
        for model in ("cnn", "baseline"):
            rt, srv = [], []
            for _ in range(a.repeats):
                dt, r = post_predict(url, img, lay, model); rt.append(dt * 1000); srv.append(r["timing_ms"]["total"])
            rec = dict(sample=s["name"], model=model, bays=r["summary"]["total"], round_trip_ms_median=round(statistics.median(rt), 1),
                       server_ms_median=round(statistics.median(srv), 1), ms_per_bay_server=round(statistics.median(srv) / r["summary"]["total"], 2))
            out["per_sample"].append(rec); print(rec)
    host = url.split("//")[-1].split("/")[0].replace(":", "_")
    json.dump(out, open(os.path.join(ROOT, "results", f"latency_{host}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
