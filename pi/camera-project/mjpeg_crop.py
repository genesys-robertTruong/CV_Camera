#!/usr/bin/env python3
"""Live MJPEG stream from the Arducam 64MP made of real, unscaled sensor pixels:
a centre crop of the sensor at 720p, 1080p or 4K.

    python3 mjpeg_crop.py [--mode 720|1080|3k|4k|full] [--fps N]   (default 720)

--fps locks the frame rate to N (e.g. 25) instead of running at the mode's max.

720/1080/4k use the patched driver's (driver/arducam_64mp) true 1:1 crop
modes -- 1280x768 (~65fps), 1920x1104 (~47fps), 3072x1728 (~30fps), 3840x2208
(~18fps) -- and ISP ScalerCrop trims each to the centre 1280x720 / 1920x1080 /
3072x1728 / 3840x2160 with main the same size, so the ISP does no scaling and output pixels map 1:1 to
sensor pixels. `full` crops the centre 1280x720 out of the stock full-res
mode (~2.3fps) -- a slow reference that doesn't need the patched driver.

Single process, one Picamera2 for its whole lifetime -- no mode switching, so
the reconfigure timeouts that forced mjpeg_server.py's subprocess-per-mode
design don't apply here.
"""
import json
import logging
import os
import socketserver
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs

import cv2
from picamera2 import Picamera2

TUNING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_64mp_sharper.json")
CAMERA_NUM = 0  # see mjpeg_server.py -- re-verify if another camera is added

# mode: (raw sensor mode, output size, buffer count, driver line-length param)
MODES = {
    "720": ((1280, 768), (1280, 720), 6, "llp_720p"),
    "1080": ((1920, 1104), (1920, 1080), 6, "llp_1080p"),
    "3k": ((3072, 1728), (3072, 1728), 4, "llp_3k"),
    "4k": ((3840, 2208), (3840, 2160), 4, "llp_4k"),  # 25MB RGB888 per main buffer
    "full": ((9248, 6944), (1280, 720), 4, None),  # ~80MB CMA per raw buffer
}
MODE = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else "720"
RAW_SIZE, OUT_SIZE, BUFFER_COUNT, LLP_PARAM = MODES[MODE]
DRIVER_PARAMS = "/sys/module/arducam_64mp/parameters"
PIXEL_RATE = 900e6  # ARDUCAM_64MP_PIXEL_RATE in the driver


def _crop_mode_frame_us():
    """The crop mode's minimum frame time, from the driver's live params."""
    def param(name):
        with open(os.path.join(DRIVER_PARAMS, name)) as f:
            return int(f.read())
    lines = RAW_SIZE[1] + param("crop_min_vblank")
    return int(lines * param(LLP_PARAM) / PIXEL_RATE * 1e6)


FIXED_FPS = float(sys.argv[sys.argv.index("--fps") + 1]) if "--fps" in sys.argv else None
if FIXED_FPS:
    if LLP_PARAM and 1e6 / FIXED_FPS < _crop_mode_frame_us():
        sys.exit(f"--fps {FIXED_FPS:g} is faster than the {MODE} mode's max "
                 f"{1e6 / _crop_mode_frame_us():.1f}fps")
    FRAME_LIMITS = (int(1e6 / FIXED_FPS),) * 2
elif LLP_PARAM:
    # Cap the frame time at the mode's own minimum (+1%) so auto-exposure
    # makes up for dim light with gain instead of stretching frames (indoors
    # AE otherwise picks ~30ms and halves 720p's frame rate).
    FRAME_LIMITS = (1000, int(_crop_mode_frame_us() * 1.01))
else:
    FRAME_LIMITS = (100, 1000000)
# cv2.imencode on the CM4 is ~10-15ms at 720p; 3K/4K need all 4 cores to
# keep up with the sensor.
ENCODER_THREADS = 4 if MODE in ("3k", "4k") else 3
JPEG_QUALITY = 85
PORT = 8080
START_LENS = 6.65  # dioptres; starting point before the startup centre focus

PAGE = """\
<html>
<head>
<title>64MP -- 1:1 Sensor Crop</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  html, body { height: 100%; }
  /* Video takes whatever height the controls leave; JS (fitVideo) sizes the
     image to fit that box at the stream's aspect ratio, so the controls are
     always on screen and the tap marker stays aligned with the picture. */
  body { margin: 0; background: #000; color: #ddd; display: flex; flex-direction: column;
         height: 100vh; height: 100dvh; overflow: hidden;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #stage { flex: 1 1 auto; min-height: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; }
  #wrap { position: relative; line-height: 0; }
  #video { display: block; cursor: crosshair; touch-action: manipulation;
           -webkit-user-drag: none; user-select: none; }
  #marker { position: absolute; display: none; box-sizing: border-box; pointer-events: none;
            border: 2px solid #ffd54a; border-radius: 4px; transition: border-color 0.2s; }
  #marker.done { border-color: #5fd36b; } #marker.failed { border-color: #ff5a5a; }
  .seg { display: inline-flex; border: 1px solid #444; border-radius: 4px; overflow: hidden; vertical-align: middle; }
  .seg input { display: none; }
  .seg label { padding: 0.35em 0.7em; cursor: pointer; background: #222; color: #aaa; }
  .seg input:checked + label { background: #2f5d8f; color: #fff; }
  #fstatus { font-family: ui-monospace, monospace; font-size: 0.85em; color: #ffd54a; margin-top: 0.4em; min-height: 1.2em; }
  #controls { flex: 0 0 auto; max-height: 45vh; overflow-y: auto; box-sizing: border-box;
              width: 100%; max-width: 1280px; margin: 0 auto; padding: 0.5em 1em 0.6em;
              overflow-x: hidden; overflow-wrap: anywhere; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 0.4em 1.2em; margin-top: 0.4em; }
  .row > label { flex: 1 1 260px; min-width: 0; }  /* let sliders shrink on phones */
  input[type=range] { width: 100%; accent-color: #4a9eff; }
  button { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 0.35em 0.7em; }
  #info { font-family: ui-monospace, monospace; font-size: 0.8em; color: #9fdca0; margin-top: 0.4em; }
  .hint { color: #888; font-size: 0.75em; margin: 0.3em 0 0; }
</style>
</head>
<body>
<div id="stage"><div id="wrap"><img id="video" src="stream.mjpg" draggable="false" /><div id="marker"></div></div></div>
<div id="controls">
  <div class="row" style="margin-top: 0;">
    <button id="refocus">Focus centre</button>
    <button id="cont">Continuous AF</button>
    <span class="seg" title="Fast: phase-detect AF only. ~1s in 3k/4k; 720p/1080p have too little phase confidence when badly defocused and add a contrast scan (~2.5s). Lands within ~0.1-0.2 dioptre of best focus. Precise: then a contrast sweep on the tapped spot (~1-2s more) that lands on the true sharpness peak.">
      <input type="radio" name="afq" id="afFast" value="0"><label for="afFast">Fast (phase AF)</label>
      <input type="radio" name="afq" id="afPrecise" value="1"><label for="afPrecise">Precise (phase + contrast)</label>
    </span>
  </div>
  <div id="fstatus">Tap the video to focus there.</div>
  <div class="row">
    <label>Manual focus (dioptres): <span id="fv"></span>
      <input type="range" id="focus" min="LENS_MIN" max="LENS_MAX" step="0.01" value="6.65">
    </label>
    <label>Digital zoom (ISP crop): <span id="zv"></span>
      <input type="range" id="zoom" min="1" max="ZOOM_MAX" step="0.05" value="ZOOM_NOW">
    </label>
    <button id="zoom1">1x</button>
  </div>
  <div id="info">loading...</div>
  <p class="hint">MODE_HINT. 1.0x = 1:1 sensor pixels; above that the ISP crops the
  centre and scales it back up. Depth of field is very shallow -- use small focus moves.</p>
</div>
<script>
const video = document.getElementById('video');
// Fit the video into #stage at the stream's aspect ratio.
const ASPECT = OUT_W / OUT_H, stage = document.getElementById('stage');
function fitVideo() {
  const r = stage.getBoundingClientRect();
  const w = Math.max(1, Math.floor(Math.min(r.width, r.height * ASPECT)));
  video.style.width = w + 'px';
  video.style.height = Math.floor(w / ASPECT) + 'px';
}

// Send slider values without flooding the server: at most one request in
// flight per endpoint, latest value wins.
function throttled(path) {
  let pending = false, queued = null;
  const send = (v) => {
    if (pending) { queued = v; return; }
    pending = true;
    fetch(path + v).finally(() => {
      pending = false;
      if (queued !== null) { const q = queued; queued = null; send(q); }
    });
  };
  return send;
}
const f = document.getElementById('focus'), fv = document.getElementById('fv');
const sendFocus = throttled('/focus?position=');
f.addEventListener('input', () => { fv.textContent = f.value; marker.style.display = 'none'; sendFocus(f.value); });

const z = document.getElementById('zoom'), zv = document.getElementById('zv');
const sendZoom = throttled('/zoom?level=');
function zoomLabel() {
  const v = parseFloat(z.value);
  zv.textContent = v.toFixed(2) + 'x' + (v <= 1.0001 ? ' (1:1 sensor pixels)' : ' (ISP upscaled)');
}
function setZoom(v) { z.value = v; zoomLabel(); marker.style.display = 'none'; sendZoom(z.value); }
z.addEventListener('input', () => setZoom(z.value));
document.getElementById('zoom1').onclick = () => setZoom(1);
zoomLabel();
document.getElementById('cont').onclick = () => { marker.style.display = 'none'; fetch('/afmode?mode=continuous'); };

// Tap-to-focus: the AF window is WIN_FRAC of the frame in each dimension,
// centred on the tap (clamped inside the frame), same as the server uses.
const WIN = WIN_FRAC, marker = document.getElementById('marker');
const fstatus = document.getElementById('fstatus');
let polling = null;
function tapFocus(fx, fy) {
  const cx = Math.min(Math.max(fx, WIN / 2), 1 - WIN / 2), cy = Math.min(Math.max(fy, WIN / 2), 1 - WIN / 2);
  Object.assign(marker.style, { display: 'block', left: (cx - WIN / 2) * 100 + '%', top: (cy - WIN / 2) * 100 + '%',
                                width: WIN * 100 + '%', height: WIN * 100 + '%' });
  marker.className = '';
  fstatus.textContent = 'Focusing...';
  const precise = document.getElementById('afPrecise').checked ? 1 : 0;
  fetch('/tapfocus?x=' + fx.toFixed(4) + '&y=' + fy.toFixed(4) + '&refine=' + precise);
  if (!polling) polling = setInterval(pollFocus, 250);
}
function pollFocus() {
  fetch('/focusstatus').then(r => r.json()).then(st => {
    fstatus.textContent = st.text;
    if (st.state === 'done' || st.state === 'failed' || st.state === 'idle') {
      marker.className = st.state === 'failed' ? 'failed' : 'done';
      clearInterval(polling); polling = null;
      if (st.lens !== null) { f.value = st.lens; fv.textContent = st.lens.toFixed(2); }
    }
  }).catch(() => {});
}
video.addEventListener('click', (e) => {
  const r = video.getBoundingClientRect();
  tapFocus((e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
});
document.getElementById('refocus').onclick = () => tapFocus(0.5, 0.5);
// Fast/Precise choice: remembered per browser (default DEFAULT_AFQ).
let afq = 'DEFAULT_AFQ';
try { afq = localStorage.getItem('afq') || afq; } catch (e) {}
document.getElementById(afq === 'fast' ? 'afFast' : 'afPrecise').checked = true;
for (const id of ['afFast', 'afPrecise'])
  document.getElementById(id).addEventListener('change', () => {
    try { localStorage.setItem('afq', id === 'afFast' ? 'fast' : 'precise'); } catch (e) {}
  });
new ResizeObserver(fitVideo).observe(stage);
window.addEventListener('resize', fitVideo);
fitVideo();
// Show the startup centre focus as it happens.
marker.style.display = 'none'; polling = setInterval(pollFocus, 250);
setInterval(() => fetch('/afinfo').then(r => r.text())
  .then(t => document.getElementById('info').textContent = t).catch(() => {}), 1000);
</script>
</body>
</html>
"""


class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


output = StreamingOutput()
picam2 = None
lens_min, lens_max = 0.0, 15.0
af_state_names = {0: "Idle", 1: "Scanning", 2: "Focused", 3: "Failed"}
stats = {"stream_fps": 0.0}


def af_info_text():
    md = picam2.capture_metadata()
    pos = md.get("LensPosition")
    dist = "unknown" if pos is None else ("infinity" if pos == 0 else f"~{100.0 / pos:.0f}cm")
    fps = 1e6 / md["FrameDuration"] if md.get("FrameDuration") else 0
    return (f"AF: {af_state_names.get(md.get('AfState'), md.get('AfState'))}  "
            f"Lens: {pos if pos is None else round(pos, 2)} ({dist})  "
            f"Zoom {view['zoom']:.2f}x  Crop: {md.get('ScalerCrop')}  "
            f"sensor {fps:.1f} fps, streamed {stats['stream_fps']:.1f} fps  "
            f"exp {md.get('ExposureTime')}us gain {md.get('AnalogueGain', 0):.1f}")


# Latest captured frame + its metadata, published by capture_loop so the
# focus code can wait for frames taken *after* a lens move.
latest = {"seq": 0, "frame": None, "lens": None, "af_state": None}
latest_cv = Condition()

# Tap-to-focus. Measured on this lens (tools/af_accuracy.py): libcamera's AF
# gets close fast -- PDAF in the 3k/4k crops, a contrast scan then PDAF in
# 720p/1080p -- but lands up to ~0.1-0.3 dioptre off the true peak, and the
# sharp zone is only ~0.3 dioptre wide. So a tap runs a fast phase stage with
# the AF window on the tapped spot, then (if `refine`, "Precise") a short
# contrast sweep around that result, scored on the same spot.
#
# The fast stage drives libcamera's *continuous* AF and stops it once it has
# converged, rather than using its one-shot (AfMode Auto) cycle: from far out
# of focus, one-shot took 1.5-1.9s and still landed ~0.2 off (and reported
# Failed in 3k), while continuous slews straight there on phase (~0.7s 1.0 ->
# 6.6 in 4k). In 720p/1080p, where phase confidence is ~0 when badly
# defocused, continuous AF itself falls back to a contrast scan first.
CONVERGED_FRAMES = 4       # lens still (and AfState Focused) this many frames = done
PHASE_TIMEOUT = 4.0
#
# The sweep commands a new lens position every frame without waiting for it
# to land: a LensPosition change takes ~6-9 frames to reach the image (requests
# are queued ahead), and frame metadata's LensPosition follows the image to
# within ~1 frame (tools/lens_latency.py) -- so each frame is scored against
# the position *it* reports. Waiting per step instead cost ~0.4s/step. One
# up and one down pass are averaged to cancel that residual ~1-frame skew.
WIN_FRAC = 0.2             # scoring ROI (and page marker) size, fraction of the frame
AF_WIN_FRAC = 0.4          # libcamera AF window size: phase needs more PDAF cells than the ROI
REFINE_HALF_RANGE = 0.30   # dioptres either side of the AF result. Phase AF was measured
                           # landing 0.1-0.2 short; at 0.21 that hit the sweep edge and cost
                           # a second pair of passes (4 short passes ~96 frames vs 2 wide ~60)
REFINE_STEP = 0.03
CROP = None                # 1:1 base crop (absolute sensor coords), set at startup
# Digital zoom = ISP ScalerCrop of a centred sub-rectangle of CROP; main stays
# OUT_SIZE so the ISP scales it back up (1.0 = 1:1 sensor pixels).
ZOOM_MAX = 8.0
view = {"zoom": 1.0, "crop": None}  # current ScalerCrop
view_lock = threading.Lock()
# Phase AF needs enough of the sensor's 16x12 PDAF grid (which covers the raw
# readout window, not the zoomed view), so when zoomed in the AF window is
# kept at least this fraction of the 1:1 crop.
AF_WIN_MIN_FRAC = 0.4  # 0.25 at 3x zoom fell back to a contrast scan (3.2s vs ~0.9s)


def zoom_crop(level):
    cx0, cy0, cw, ch = CROP
    w = max(64, int(cw / level)) & ~1
    h = max(64, int(round(w * ch / cw))) & ~1
    return (cx0 + (cw - w) // 2, cy0 + (ch - h) // 2, w, h)


def set_zoom(level):
    level = min(max(level, 1.0), ZOOM_MAX)
    crop = zoom_crop(level)
    with view_lock:
        view.update(zoom=level, crop=crop)
    picam2.set_controls({"ScalerCrop": crop})
    return level, crop
focus = {"gen": 0, "state": "idle", "text": "", "lens": None}
focus_lock = threading.Lock()


def _wait_frame(after_seq, lens=None, timeout=3.0):
    """Next frame newer than after_seq (and, if given, reporting that lens
    position), as (seq, frame, snapshot of `latest`); None on timeout."""
    deadline = time.time() + timeout
    with latest_cv:
        while True:
            ok = latest["seq"] > after_seq and (
                lens is None or (latest["lens"] is not None and abs(latest["lens"] - lens) < 0.011))
            if ok:
                return latest["seq"], latest["frame"], dict(latest)
            left = deadline - time.time()
            if left <= 0:
                return None
            latest_cv.wait(left)


def _set_focus(gen, state, text, lens=None):
    with focus_lock:
        if gen != focus["gen"]:
            return False  # superseded by a newer tap / manual move
        focus.update(state=state, text=text, lens=lens)
        return True


def _sharpness(frame, roi):
    x, y, w, h = roi
    g = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def tap_focus(fx, fy, refine, gen):
    from libcamera import controls
    cx0, cy0, cw, ch = CROP
    with view_lock:
        vx, vy, vw, vh = view["crop"]
    # Contrast-scoring ROI: WIN_FRAC of the *output* frame (= what's on
    # screen, zoomed or not) centred on the tap -- matches the page marker.
    ow, oh = OUT_SIZE
    rw, rh = int(ow * WIN_FRAC), int(oh * WIN_FRAC)
    roi = (min(max(int(fx * ow) - rw // 2, 0), ow - rw), min(max(int(fy * oh) - rh // 2, 0), oh - rh), rw, rh)
    # AF window in absolute sensor coordinates (libcamera's AfWindows use the
    # same space as ScalerCrop): AF_WIN_FRAC of the view, but never less
    # than AF_WIN_MIN_FRAC of the 1:1 crop, centred on the tapped sensor
    # point and clamped inside the 1:1 crop.
    aw = min(cw, max(int(vw * AF_WIN_FRAC), int(cw * AF_WIN_MIN_FRAC)))
    ah = min(ch, max(int(vh * AF_WIN_FRAC), int(ch * AF_WIN_MIN_FRAC)))
    px, py = vx + fx * vw, vy + fy * vh
    ax = min(max(int(px - aw / 2), cx0), cx0 + cw - aw)
    ay = min(max(int(py - ah / 2), cy0), cy0 + ch - ah)
    picam2.set_controls({"AfMetering": controls.AfMeteringEnum.Windows,
                         "AfWindows": [(ax, ay, aw, ah)]})
    if not _set_focus(gen, "focusing", "Focusing (libcamera AF)..."):
        return
    t0 = time.time()
    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
    seq, frames, still, scanned = latest["seq"], 0, 0, 0
    af_lens, af_state, prev = None, None, None
    converged = False
    while time.time() - t0 < PHASE_TIMEOUT:
        got = _wait_frame(seq)
        if got is None:
            break
        seq, _, md = got
        frames += 1
        af_lens, af_state = md["lens"], md["af_state"]
        if frames > 10 and af_state == 1:
            scanned += 1  # libcamera fell back to a contrast scan (low phase confidence)
        # Ignore the first ~10 frames: the mode/window change is still in
        # the request pipeline (same ~6-9 frame lag as lens moves).
        if frames > 10 and af_lens is not None and prev is not None and abs(af_lens - prev) < 0.015 \
                and af_state == 2:
            still += 1
        else:
            still = 0
        prev = af_lens
        if still >= CONVERGED_FRAMES:
            converged = True
            break
        if gen != focus["gen"]:
            return
    if not converged:
        af_state = 3
    if af_lens is None:
        _set_focus(gen, "failed", "AF got no frames")
        return
    picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": af_lens})
    t_af = time.time() - t0
    how = "Phase AF" if not scanned else "Phase+scan AF"
    if not refine:
        state = "failed" if af_state == 3 else "done"
        _set_focus(gen, state, f"{how} {'did not converge' if af_state == 3 else 'focused'} at "
                               f"{af_lens:.2f} in {t_af:.1f}s", af_lens)
        return
    if not _set_focus(gen, "refining", f"{how} at {af_lens:.2f} in {t_af:.1f}s, refining...", af_lens):
        return

    def sweep(positions):
        """Step the lens through `positions`, one per frame, then keep reading
        until frames report the last one. Returns {reported lens: mean score}."""
        scores = {}
        seq = latest["seq"]

        def take(seq):
            got = _wait_frame(seq)
            if got is None:
                return None
            seq, frame, md = got
            if md["lens"] is not None:
                scores.setdefault(round(md["lens"], 3), []).append(_sharpness(frame, roi))
            return seq

        for lp in positions:
            if gen != focus["gen"]:
                return None
            picam2.set_controls({"LensPosition": float(lp)})
            seq = take(seq)
            if seq is None:
                return None
        for _ in range(15):  # drain the ~6-9 frame lens pipeline
            seq = take(seq)
            if seq is None or round(latest["lens"] or -1, 3) == round(positions[-1], 3):
                break
        take(seq)
        return {lp: sum(v) / len(v) for lp, v in scores.items()}

    def peak(scores, lo, hi):
        """Parabolic-fit peak of the scores reported within [lo, hi], and
        whether it sits at the range edge (i.e. the true peak may be outside)."""
        pts = sorted((lp, sc) for lp, sc in scores.items() if lo - 1e-6 <= lp <= hi + 1e-6)
        if len(pts) < 3:
            return None, False
        i = max(range(len(pts)), key=lambda k: pts[k][1])
        if i in (0, len(pts) - 1):
            return pts[i][0], True
        (l0, s0), (l1, s1), (l2, s2) = pts[i - 1], pts[i], pts[i + 1]
        den = s0 - 2 * s1 + s2
        best = l1 + 0.5 * (s0 - s2) / den * (l2 - l1) if den < 0 else l1
        return best, False

    n = int(round(REFINE_HALF_RANGE / REFINE_STEP))
    centre, best, passes = af_lens, None, 0
    for _ in range(2):  # one re-centre allowed if the peak is at the edge
        grid = [round(min(max(centre + i * REFINE_STEP, lens_min), lens_max), 3) for i in range(-n, n + 1)]
        up = sweep(grid)
        down = sweep(grid[::-1]) if up is not None else None
        if up is None or down is None:
            return  # superseded, or frames stopped
        passes += 2
        pu, edge_u = peak(up, grid[0], grid[-1])
        pd, edge_d = peak(down, grid[0], grid[-1])
        if pu is None or pd is None:
            break
        best = (pu + pd) / 2
        if not (edge_u or edge_d) or grid[0] in (lens_min,) or grid[-1] in (lens_max,):
            break
        centre = best
    if best is None:
        _set_focus(gen, "failed", f"AF at {af_lens:.2f}; refine got no usable frames", af_lens)
        return
    best = round(best, 3)
    picam2.set_controls({"LensPosition": best})
    _set_focus(gen, "done", f"{how} {af_lens:.2f} ({t_af:.1f}s) -> refined {best:.2f} ({best - af_lens:+.2f}, "
                            f"{passes} passes, {time.time() - t0 - t_af:.1f}s)", best)


def start_tap_focus(fx, fy, refine):
    with focus_lock:
        focus["gen"] += 1
        gen = focus["gen"]
        focus.update(state="focusing", text="Focusing...", lens=None)
    threading.Thread(target=tap_focus, args=(fx, fy, refine, gen), daemon=True).start()


def cancel_focus_job():
    with focus_lock:
        focus["gen"] += 1
        focus.update(state="idle", text="", lens=None)


encoded = {"seq": -1, "count": 0}
encoded_lock = threading.Lock()


def _encode_and_publish(seq, frame, slots):
    try:
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        with encoded_lock:
            # Encoders finish out of order; never publish an older frame
            # over a newer one.
            if seq <= encoded["seq"]:
                return
            encoded["seq"] = seq
            encoded["count"] += 1
        output.write(jpg.tobytes())
    finally:
        slots.release()


def capture_loop():
    # cv2.imencode releases the GIL, so a small pool encodes in parallel.
    # `slots` bounds frames in flight: when encoders fall behind, new frames
    # are dropped here instead of queueing up latency.
    slots = threading.Semaphore(ENCODER_THREADS)
    pool = ThreadPoolExecutor(max_workers=ENCODER_THREADS)
    seq = 0
    while True:
        req = picam2.capture_request()
        try:
            frame = req.make_array("main")
            md = req.get_metadata()
        finally:
            req.release()
        seq += 1
        with latest_cv:
            latest.update(seq=seq, frame=frame, lens=md.get("LensPosition"), af_state=md.get("AfState"))
            latest_cv.notify_all()
        if slots.acquire(blocking=False):
            pool.submit(_encode_and_publish, seq, frame, slots)


def stream_fps_loop():
    while True:
        with encoded_lock:
            start = encoded["count"]
        time.sleep(2)
        with encoded_lock:
            stats["stream_fps"] = (encoded["count"] - start) / 2


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        if url.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif url.path == "/index.html":
            hint = (f"Centre {OUT_SIZE[0]}x{OUT_SIZE[1]} of the sensor, {MODE} mode, "
                    f"max {1e6 / FRAME_LIMITS[1]:.0f} fps" if LLP_PARAM else
                    "Centre 1280x720 of the sensor from the full-res mode, ~2-3 fps")
            body = (PAGE.replace("LENS_MIN", str(lens_min)).replace("LENS_MAX", str(lens_max))
                    .replace("MODE_HINT", hint).replace("WIN_FRAC", str(WIN_FRAC))
                    .replace("ZOOM_MAX", str(ZOOM_MAX)).replace("ZOOM_NOW", f"{view['zoom']:.2f}")
                    .replace("OUT_W", str(OUT_SIZE[0])).replace("OUT_H", str(OUT_SIZE[1]))
                    .replace("DEFAULT_AFQ", "fast" if MODE == "full" else "precise").encode())
            self._send(body, "text/html")
        elif url.path == "/focus":
            try:
                pos = float(qs.get("position", ["0"])[0])
            except ValueError:
                pos = 0.0
            pos = max(lens_min, min(pos, lens_max))
            cancel_focus_job()
            picam2.set_controls({"AfMode": 0, "LensPosition": pos})
            self._send(f"ok {pos:.2f}".encode())
        elif url.path == "/afmode":
            # Continuous AF uses the default (centre-weighted) metering again.
            from libcamera import controls
            cancel_focus_job()
            picam2.set_controls({"AfMetering": controls.AfMeteringEnum.Auto,
                                 "AfMode": 0 if qs.get("mode", [""])[0] == "manual" else 2})
            self._send(b"ok")
        elif url.path == "/zoom":
            try:
                level = float(qs.get("level", ["1"])[0])
            except ValueError:
                level = 1.0
            level, crop = set_zoom(level)
            self._send(f"ok {level:.2f} {crop}".encode())
        elif url.path == "/tapfocus":
            try:
                fx = min(max(float(qs.get("x", ["0.5"])[0]), 0.0), 1.0)
                fy = min(max(float(qs.get("y", ["0.5"])[0]), 0.0), 1.0)
            except ValueError:
                fx = fy = 0.5
            start_tap_focus(fx, fy, qs.get("refine", ["1"])[0] == "1")
            self._send(b"focusing")
        elif url.path == "/focusstatus":
            with focus_lock:
                body = json.dumps({k: focus[k] for k in ("state", "text", "lens")}).encode()
            self._send(body, "application/json")
        elif url.path == "/afinfo":
            self._send(af_info_text().encode())
        elif url.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError) as e:
                logging.info("Client %s disconnected: %s", self.client_address, e)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass

    def _send(self, body, content_type="text/plain"):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    picam2 = Picamera2(camera_num=CAMERA_NUM, tuning=Picamera2.load_tuning_file(TUNING_FILE))
    picam2.configure(picam2.create_video_configuration(
        main={"size": OUT_SIZE, "format": "RGB888"},
        raw={"size": RAW_SIZE},
        buffer_count=BUFFER_COUNT,
        controls={"FrameDurationLimits": FRAME_LIMITS},
    ))
    picam2.start()

    # ScalerCrop bounds are only valid after configure(); max_crop is the
    # raw mode's whole window in full-sensor coordinates.
    x, y, w, h = picam2.camera_controls["ScalerCrop"][1]
    crop = (x + (w - OUT_SIZE[0]) // 2, y + (h - OUT_SIZE[1]) // 2, OUT_SIZE[0], OUT_SIZE[1])
    CROP = crop
    view["crop"] = crop
    picam2.set_controls({"ScalerCrop": crop, "AfMode": 0, "LensPosition": START_LENS})
    lens_min, lens_max, _ = picam2.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))

    threading.Thread(target=capture_loop, daemon=True).start()
    # Focus the centre once at startup (tap-to-focus path, refined), rather
    # than trusting the fixed START_LENS for whatever scene is in front.
    start_tap_focus(0.5, 0.5, refine=MODE != "full")
    threading.Thread(target=stream_fps_loop, daemon=True).start()
    try:
        http_server = StreamingServer(("0.0.0.0", PORT), StreamingHandler)
        print(f"Serving {MODE} sensor-crop stream (crop {crop}, frame limits {FRAME_LIMITS}) on :{PORT}/index.html")
        http_server.serve_forever()
    finally:
        picam2.stop()
        picam2.close()
