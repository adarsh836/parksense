// ParkSense operator console - vanilla JS, no build step.
const $ = (id) => document.getElementById(id);
const cv = $("cv"), ctx = cv.getContext("2d");
const COL = { free: "#1b9e77", occupied: "#d1495b", unsure: "#e9a23b", draw: "#f7f7f2", sel: "#4aa3ff" };

let img = null, imgBlob = null;
let slots = [];            // [{id, points:[[x,y]x4]}] in image pixels
let current = [];          // points of the bay being drawn
let selected = -1;
let results = null;        // {id: {p_occupied, state, confidence, override}}
let mode = "draw";
let lastSummary = null;
let samples = { samples: [], sequences: [] };

// ---------------------------------------------------------------- loading
fetch("/api/samples").then(r => r.json()).then(s => {
  samples = s;
  for (const [i, x] of s.samples.entries()) {
    const o = document.createElement("option"); o.value = i; o.textContent = x.name; $("sample").append(o);
  }
  if (!s.sequences.length) $("seq-sample").hidden = true;
});

async function loadImageBlob(blob) {
  imgBlob = blob;
  const url = URL.createObjectURL(blob);
  img = new Image();
  await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = url; });
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  $("empty").hidden = true; cv.hidden = false; $("tools").hidden = false; $("hint").hidden = false;
  results = null; showSummary(null); draw(); refresh();
}

$("sample").onchange = async (e) => {
  const s = samples.samples[e.target.value]; if (!s) return;
  const blob = await (await fetch(s.image)).blob();
  await loadImageBlob(blob);
  const L = await (await fetch(s.layout)).json();
  setLayout(L);
};
$("file").onchange = (e) => { const f = e.target.files[0]; if (f) loadImageBlob(f); };

function setLayout(L) {
  let list = L.slots || L;
  if (L.image_size && img) {
    const sx = img.naturalWidth / L.image_size[0], sy = img.naturalHeight / L.image_size[1];
    list = list.map(s => ({ id: s.id, points: s.points.map(([x, y]) => [x * sx, y * sy]) }));
  }
  slots = list.map((s, i) => ({ id: String(s.id ?? i + 1), points: s.points }));
  results = null; selected = -1; current = []; showSummary(null); draw(); refresh();
}

// ---------------------------------------------------------------- layout file
$("export").onclick = () => {
  const L = { image_size: img ? [img.naturalWidth, img.naturalHeight] : null,
              slots: slots.map(s => ({ id: s.id, points: s.points.map(p => p.map(v => Math.round(v * 10) / 10)) })) };
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(L, null, 1)], { type: "application/json" }));
  a.download = "parksense_layout.json"; a.click();
};
$("import").onchange = async (e) => {
  const f = e.target.files[0]; if (!f) return;
  try { setLayout(JSON.parse(await f.text())); } catch { say("That file is not a ParkSense layout (JSON with a list of slots).", true); }
};

// ---------------------------------------------------------------- drawing
function toImage(ev) {
  const r = cv.getBoundingClientRect();
  return [(ev.clientX - r.left) * cv.width / r.width, (ev.clientY - r.top) * cv.height / r.height];
}
function inside(pt, poly) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function nextId() { let n = slots.length + 1; const ids = new Set(slots.map(s => s.id)); while (ids.has(String(n))) n++; return String(n); }

cv.addEventListener("pointerdown", (ev) => {
  if (!img) return;
  const pt = toImage(ev);
  const hit = slots.findIndex(s => inside(pt, s.points));
  if (mode === "review") {
    if (hit >= 0 && results) cycleOverride(slots[hit].id);
    return;
  }
  if (current.length === 0 && hit >= 0) { selected = hit === selected ? -1 : hit; draw(); return; }
  current.push(pt); selected = -1;
  if (current.length === 4) {
    slots.push({ id: nextId(), points: current }); current = [];
    if (results) { results = null; showSummary(null); }
  }
  draw(); refresh();
});

function setMode(m) {
  mode = m;
  $("mode-draw").classList.toggle("on", m === "draw"); $("mode-draw").setAttribute("aria-pressed", m === "draw");
  $("mode-review").classList.toggle("on", m === "review"); $("mode-review").setAttribute("aria-pressed", m === "review");
  $("hint").textContent = m === "draw"
    ? "Click the four corners of a bay, in any order around it. The bay closes after the fourth click. Click inside a bay to select it."
    : "Click a bay to set it yourself: occupied, then free, then back to the model's answer.";
  current = []; draw();
}
$("mode-draw").onclick = () => setMode("draw");
$("mode-review").onclick = () => setMode("review");
$("undo").onclick = () => { if (current.length) current.pop(); else if (slots.length) slots.pop(); results = null; showSummary(null); draw(); refresh(); };
$("del").onclick = () => { if (selected >= 0) { slots.splice(selected, 1); selected = -1; results = null; showSummary(null); draw(); refresh(); } };
$("clear").onclick = () => { if (slots.length && confirm("Remove all " + slots.length + " bay outlines?")) { slots = []; current = []; results = null; showSummary(null); draw(); refresh(); } };
document.addEventListener("keydown", (e) => { if ((e.key === "Delete" || e.key === "Backspace") && selected >= 0 && document.activeElement === document.body) $("del").click(); });

function stateOf(id) { const r = results && results[id]; return r ? (r.override || r.state) : null; }

function hatch(color) {
  const c = document.createElement("canvas"); c.width = c.height = 10; const g = c.getContext("2d");
  g.strokeStyle = color; g.lineWidth = 3; g.beginPath(); g.moveTo(0, 10); g.lineTo(10, 0); g.stroke();
  return ctx.createPattern(c, "repeat");
}

function draw() {
  if (!img) return;
  ctx.drawImage(img, 0, 0);
  const lw = Math.max(2, cv.width / 500), fs = Math.max(12, cv.width / 70);
  ctx.font = `700 ${fs}px Overpass, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  slots.forEach((s, i) => {
    const st = stateOf(s.id), col = st ? COL[st] : COL.draw;
    ctx.beginPath(); s.points.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath();
    if (st) { ctx.globalAlpha = 0.35; ctx.fillStyle = st === "unsure" ? hatch(COL.unsure) : col; ctx.fill(); ctx.globalAlpha = 1; }
    ctx.lineWidth = i === selected ? lw * 2 : lw; ctx.strokeStyle = i === selected ? COL.sel : col; ctx.stroke();
    const cx = s.points.reduce((a, p) => a + p[0], 0) / 4, cy = s.points.reduce((a, p) => a + p[1], 0) / 4;
    const tag = st ? ({ free: "F", occupied: "O", unsure: "?" })[st] + (results[s.id].override ? "*" : "") : s.id;
    ctx.lineWidth = 3; ctx.strokeStyle = "rgba(0,0,0,.75)"; ctx.strokeText(tag, cx, cy); ctx.fillStyle = "#fff"; ctx.fillText(tag, cx, cy);
  });
  if (current.length) {
    ctx.strokeStyle = COL.sel; ctx.lineWidth = lw; ctx.beginPath();
    current.forEach(([x, y], k) => k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.stroke();
    current.forEach(([x, y]) => { ctx.fillStyle = COL.sel; ctx.beginPath(); ctx.arc(x, y, lw * 2.2, 0, 7); ctx.fill(); });
  }
}

function refresh() {
  $("nslots").textContent = slots.length;
  $("run").disabled = !(img && slots.length);
}
function say(text, err = false) { $("msg").innerHTML = text ? `<div class="notice ${err ? "err" : ""}">${text}</div>` : ""; }

// ---------------------------------------------------------------- inference
function layoutJSON() { return JSON.stringify({ image_size: [img.naturalWidth, img.naturalHeight], slots }); }

async function predict(blob, layout) {
  const fd = new FormData();
  fd.append("image", blob, "frame.jpg"); fd.append("layout", layout); fd.append("model", $("model").value);
  const r = await fetch("/api/predict", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "The server could not process this image.");
  return j;
}

$("run").onclick = async () => {
  $("run").disabled = true; say("Checking " + slots.length + " bays… (the first request after a quiet period can take about a minute while the free server wakes up)");
  const t0 = performance.now();
  try {
    const j = await predict(imgBlob, layoutJSON());
    results = {}; j.slots.forEach(s => results[s.id] = s);
    say(j.warnings.map(w => `<div class="notice">${w}</div>`).join(""));
    $("timing").textContent = `${j.model}. Server time ${j.timing_ms.total} ms; round trip ${Math.round(performance.now() - t0)} ms.`;
    setMode("review"); updateSummary();
  } catch (e) { say(e.message, true); }
  $("run").disabled = false; draw();
};

function cycleOverride(id) {
  const r = results[id]; if (!r) return;
  r.override = r.override === "occupied" ? "free" : r.override === "free" ? null : "occupied";
  updateSummary(); draw();
}

function updateSummary() {
  if (!results) return showSummary(null);
  const c = { free: 0, occupied: 0, unsure: 0 };
  slots.forEach(s => { const st = stateOf(s.id); if (st) c[st]++; });
  lastSummary = { total: slots.length, ...c };
  showSummary(lastSummary);
  const list = $("review-list"); list.innerHTML = "";
  const todo = slots.filter(s => results[s.id] && (results[s.id].state === "unsure" || results[s.id].override))
    .sort((a, b) => Math.abs(results[a.id].p_occupied - .5) - Math.abs(results[b.id].p_occupied - .5));
  todo.slice(0, 24).forEach(s => {
    const r = results[s.id], li = document.createElement("li"), b = document.createElement("button");
    b.textContent = `Bay ${s.id}: ${r.override ? r.override + " (set by you)" : "not sure, " + Math.round(100 * r.p_occupied) + "% occupied"}`;
    b.onclick = () => cycleOverride(s.id); li.append(b); list.append(li);
  });
  if (todo.length > 24) { const li = document.createElement("li"); li.className = "meta"; li.textContent = `…and ${todo.length - 24} more. Click them on the image.`; list.append(li); }
  if (!list.children.length) list.innerHTML = "<li class='meta'>Nothing to check: every bay has a confident answer.</li>";
}

function showSummary(s) {
  $("sign").hidden = !s; $("review").hidden = !s;
  if (!s) return;
  $("s-free").textContent = s.free; $("s-of").textContent = `free of ${s.total} bays, ${Math.round(100 * s.occupied / s.total)}% occupied`;
  $("s-f").textContent = s.free; $("s-o").textContent = s.occupied; $("s-u").textContent = s.unsure;
  $("s-meter").innerHTML = ["free", "occupied", "unsure"].map(k =>
    `<span style="width:${100 * s[k] / s.total}%;background:${k === "unsure" ? "repeating-linear-gradient(45deg,#e9a23b 0 3px,#fff 3px 5px)" : COL[k]}"></span>`).join("");
}

$("publish").onclick = async () => {
  if (!lastSummary) return;
  const r = await fetch("/api/publish", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lot: $("lot").value || "Main car park", t: Date.now() / 1000, ...lastSummary }) });
  say(r.ok ? `Published. <a href="/public">Open the public view</a> to see the counts.` : "Publishing failed. Try again.", !r.ok);
};

// ---------------------------------------------------------------- occupancy over time
function timeFromName(n) {
  const m = n.match(/(\d{4})-(\d{2})-(\d{2})[_ T](\d{2})[_:.](\d{2})/);
  return m ? `${m[4]}:${m[5]}` : null;
}

async function runSequence(frames, layout, truth) {
  const pts = []; $("seq-status").textContent = "";
  for (const [i, f] of frames.entries()) {
    $("seq-status").textContent = `Frame ${i + 1} of ${frames.length}…`;
    try {
      const j = await predict(f.blob, layout);
      const s = j.summary;
      pts.push({ label: f.label, occ: 100 * s.occupied / s.total, hi: 100 * (s.occupied + s.unsure) / s.total,
                 truth: truth ? truth[i] : null });
      drawChart(pts);
    } catch (e) { $("seq-status").textContent = e.message; return; }
  }
  $("seq-status").textContent = `${frames.length} frames checked.`;
}

$("seq").onchange = async (e) => {
  if (!slots.length) { $("seq-status").textContent = "Outline the bays first (or load a layout)."; return; }
  const files = [...e.target.files].map(f => ({ blob: f, label: timeFromName(f.name) || f.name }))
    .sort((a, b) => a.label.localeCompare(b.label));
  runSequence(files, layoutJSON(), null);
};

$("seq-sample").onclick = async () => {
  const s = samples.sequences[0]; if (!s) return;
  $("seq-status").textContent = "Downloading sample frames…";
  const layout = JSON.stringify(await (await fetch(s.layout)).json());
  const frames = [];
  for (const f of s.frames) frames.push({ blob: await (await fetch(f.image)).blob(), label: f.time });
  runSequence(frames, layout, s.frames.map(f => 100 * f.true_occupied / f.total));
};

function drawChart(pts) {
  const c = $("chart"), g = c.getContext("2d"), dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight; c.width = W * dpr; c.height = H * dpr; g.scale(dpr, dpr);
  const L = 44, R = 12, T = 30, B = 34, x = (i) => L + (pts.length < 2 ? 0 : i * (W - L - R) / (pts.length - 1)), y = (v) => T + (100 - v) * (H - T - B) / 100;
  g.clearRect(0, 0, W, H); g.font = "12px Overpass, sans-serif"; g.fillStyle = "#5d656e"; g.strokeStyle = "#e1e5e9"; g.lineWidth = 1;
  for (const v of [0, 25, 50, 75, 100]) { g.beginPath(); g.moveTo(L, y(v)); g.lineTo(W - R, y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v + "%", L - 6, y(v) + 4); }
  g.textAlign = "center"; const step = Math.ceil(pts.length / 10);
  pts.forEach((p, i) => { if (i % step === 0) g.fillText(p.label, x(i), H - 12); });
  // not-sure band
  g.fillStyle = "rgba(233,162,59,.25)"; g.beginPath();
  pts.forEach((p, i) => i ? g.lineTo(x(i), y(p.hi)) : g.moveTo(x(i), y(p.hi)));
  [...pts].reverse().forEach((p, k) => g.lineTo(x(pts.length - 1 - k), y(p.occ))); g.closePath(); g.fill();
  const line = (key, color, dash) => {
    g.setLineDash(dash); g.strokeStyle = color; g.lineWidth = 2.5; g.beginPath();
    pts.forEach((p, i) => i ? g.lineTo(x(i), y(p[key])) : g.moveTo(x(i), y(p[key]))); g.stroke(); g.setLineDash([]);
  };
  if (pts.some(p => p.truth !== null)) line("truth", "#2b2f33", [6, 5]);
  line("occ", "#d1495b", []);
  g.textAlign = "left"; g.fillStyle = "#d1495b"; g.fillText("Occupied (model)", L, 14);
  g.fillStyle = "#b07a1f"; g.fillText("Range if 'not sure' bays are occupied", L + 125, 14);
  if (pts.some(p => p.truth !== null)) { g.fillStyle = "#2b2f33"; g.fillText("- - Ground truth", L + 370, 14); }
}
