#!/usr/bin/env python3
"""Live MJPEG stream from the Arducam 64MP made of real, unscaled sensor pixels:
a centre crop of the sensor at 720p, 1080p or 4K.

    python3 mjpeg_crop.py [--mode 720|1080|3k|4k|full] [--fps N] [--stream mjpeg|h264]   (default 720, mjpeg)

--fps locks the frame rate to N (e.g. 25) instead of running at the mode's max.
--stream picks the starting stream; the page can switch it at runtime:
MJPEG (full crop size, software JPEG) or H.264 (ISP-scaled to <= 1080p,
VC4 hardware encoder, /stream.h264 played via MSE/JMuxer in static/).

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
import gc
import json
import logging
import math
import os
import socketserver
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs

import cv2
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import Output

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
# H.264 alternative (page switch, server-wide): the camera is reopened and
# reconfigured (same raw sensor mode, ~1.3s, see switch_stream) so the ISP outputs
# at most 1080p -- this encoder's limit -- and the VC4 hardware H.264
# encoder compresses it. Measured (tools/h264_test.py): it keeps up in every
# mode (720p 64fps, 1080p 46fps, 3k 25, 4k 18) at ~40-60% of one core, vs
# ~300% for MJPEG in 4k. Only one stream's outputs are configured at a time:
# with the full-size MJPEG main as well, zoomed-in frames (the ISP upscaling
# into a 3k main too) cost the 3k H.264 stream 1-4 fps. Zoom, sharpening and
# AF act upstream in the ISP/sensor, so both streams get them. Each encoder
# only runs while a client watches it.
H264_SIZE = (min(1920, OUT_SIZE[0]), min(1080, OUT_SIZE[1]))
H264_BITRATE = 12_000_000
START_STREAM = sys.argv[sys.argv.index("--stream") + 1] if "--stream" in sys.argv else "mjpeg"
STREAM_FPS = FIXED_FPS or (1e6 / FRAME_LIMITS[1] if LLP_PARAM else 2.5)  # nominal, for the encoder
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
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
  .vid { display: block; cursor: crosshair; touch-action: manipulation;
         -webkit-user-drag: none; user-select: none; background: #000; }
  .vid[hidden] { display: none; }
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
  #sys { font-family: ui-monospace, monospace; font-size: 0.8em; color: #8fc3ff; margin-top: 0.2em; }
  .hint { color: #888; font-size: 0.75em; margin: 0.3em 0 0; }
</style>
</head>
<body>
<div id="stage"><div id="wrap"><img id="video" class="vid" draggable="false" hidden /><video id="vh264" class="vid" muted autoplay playsinline disablepictureinpicture hidden></video><div id="marker"></div></div></div>
<div id="controls">
  <div class="row" style="margin-top: 0;">
    <button id="refocus">Focus centre</button>
    <button id="cont">Continuous AF</button>
    <span class="seg" title="Fast: phase-detect AF only. ~1s in 3k/4k; 720p/1080p have too little phase confidence when badly defocused and add a contrast scan (~2.5s). Lands within ~0.1-0.2 dioptre of best focus. Precise: then a contrast sweep on the tapped spot (~1-2s more) that lands on the true sharpness peak.">
      <input type="radio" name="afq" id="afFast" value="0"><label for="afFast">Fast (phase AF)</label>
      <input type="radio" name="afq" id="afPrecise" value="1"><label for="afPrecise">Precise (phase + contrast)</label>
    </span>
    <span class="seg" title="MJPEG: full OUT_W x OUT_H, software JPEG on the CPU, ~1 frame latency, high bandwidth. H.264: ISP-scaled to H264_W x H264_H, VC4 hardware encoder, ~12 Mbit/s, a little more latency.">
      <input type="radio" name="stream" id="stMjpeg"><label for="stMjpeg">MJPEG OUT_W x OUT_H</label>
      <input type="radio" name="stream" id="stH264"><label for="stH264">H.264 H264_W x H264_H</label>
    </span>
  </div>
  <div id="fstatus">Tap the video to focus there.</div>
  <div class="row">
    <label>Manual focus (dioptres): <span id="fv"></span>
      <input type="range" id="focus" min="LENS_MIN" max="LENS_MAX" step="0.01" value="6.65">
    </label>
    <label>Digital zoom (ISP crop): <span id="zv"></span>
      <input type="range" id="zoom" min="0" max="1000" step="1" value="0">
    </label>
    <button id="zoom1">1x</button>
    <label style="flex-basis: 180px;">Zoom sharpening (ISP): <span id="sv"></span>
      <input type="range" id="sharpen" min="0" max="SHARPEN_MAX" step="0.1" value="SHARPEN_NOW">
    </label>
  </div>
  <div id="info">loading...</div>
  <div id="sys"></div>
  <p class="hint">MODE_HINT. 1.0x = 1:1 sensor pixels; above that the ISP crops the
  centre and scales it back up. Depth of field is very shallow -- use small focus moves.</p>
</div>
<script src="jmuxer.min.js"></script>
<script>
const video = document.getElementById('video'), vh264 = document.getElementById('vh264');
// Fit the video into #stage at the stream's aspect ratio.
const ASPECT = OUT_W / OUT_H, stage = document.getElementById('stage');
function fitVideo() {
  const r = stage.getBoundingClientRect();
  const w = Math.max(1, Math.floor(Math.min(r.width, r.height * ASPECT)));
  for (const el of [video, vh264]) {
    el.style.width = w + 'px';
    el.style.height = Math.floor(w / ASPECT) + 'px';
  }
}

// Stream switch. Only the shown stream is fetched, and the server only runs
// an encoder while someone is watching it.
// H.264: /stream.h264 is [u32 length][u32 duration us][Annex B access unit]
// records, starting at a keyframe; JMuxer remuxes them to fMP4 for Media
// Source Extensions (WebCodecs would need an HTTPS page).
const BLANK = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
let h264 = null;
function stopH264() {
  if (!h264) return;
  h264.ctl.abort(); h264.jm.destroy(); h264 = null;
  vh264.removeAttribute('src'); vh264.load();
}
function startH264() {
  const ctl = new AbortController();
  const jm = new JMuxer({ node: vh264, mode: 'video', flushingTime: 0, maxDelay: 400,
                          fps: STREAM_FPS, clearBuffer: true, debug: false });
  const me = h264 = { ctl, jm };
  fetch('/stream.h264', { signal: ctl.signal }).then(async (r) => {
    const reader = r.body.getReader();
    let buf = new Uint8Array(0);
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      const nb = new Uint8Array(buf.length + value.length);
      nb.set(buf); nb.set(value, buf.length); buf = nb;
      let off = 0;
      while (buf.length - off >= 8) {
        const dv = new DataView(buf.buffer, buf.byteOffset + off, 8);
        const len = dv.getUint32(0), dur = dv.getUint32(4);
        if (buf.length - off < 8 + len) break;
        jm.feed({ video: buf.subarray(off + 8, off + 8 + len), duration: dur / 1000 });
        off += 8 + len;
      }
      buf = buf.subarray(off);
    }
  }).catch(() => {}).finally(() => {
    // Server restarted or the connection dropped: reconnect.
    if (h264 === me) setTimeout(() => { if (h264 === me) { stopH264(); startH264(); } }, 1000);
  });
  vh264.play().catch(() => {});
}
// Stay at the live edge (a background tab lets the buffer grow).
setInterval(() => {
  if (h264 && vh264.buffered.length && vh264.buffered.end(0) - vh264.currentTime > 0.5)
    vh264.currentTime = vh264.buffered.end(0) - 0.05;
}, 1000);
// Show `kind`, the server's current stream (the switch is server-wide: it
// reconfigures the camera, ~0.5s, so other viewers follow via /status).
let shown = null;
function setStream(kind) {
  if (kind === shown) return;
  shown = kind;
  document.getElementById(kind === 'h264' ? 'stH264' : 'stMjpeg').checked = true;
  if (kind === 'h264') {
    video.src = BLANK; video.hidden = true;  // closes the MJPEG connection
    vh264.hidden = false;
    if (!h264) startH264();
  } else {
    stopH264(); vh264.hidden = true;
    video.hidden = false; video.src = 'stream.mjpg?t=' + Date.now();
  }
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

// Zoom slider is logarithmic (0..1000 -> 1x..ZOOM_LIMIT) so the low end
// stays usable across a 1-16x range.
const ZLIM = ZOOM_LIMIT, z = document.getElementById('zoom'), zv = document.getElementById('zv');
const sendZoom = throttled('/zoom?level=');
const zoomOf = (t) => Math.pow(ZLIM, t / 1000);
const posOf = (zoom) => Math.round(1000 * Math.log(zoom) / Math.log(ZLIM));
function zoomLabel() {
  const v = zoomOf(z.value);
  zv.textContent = v.toFixed(2) + 'x' + (v <= 1.0001 ? ' (1:1 sensor pixels)' : ' (ISP upscaled)');
}
function setZoomPos(t) {
  z.value = t; zoomLabel(); marker.style.display = 'none'; sendZoom(zoomOf(z.value).toFixed(3));
}
z.addEventListener('input', () => setZoomPos(z.value));
document.getElementById('zoom1').onclick = () => setZoomPos(0);
z.value = posOf(ZOOM_NOW);
zoomLabel();

const sh = document.getElementById('sharpen'), sv = document.getElementById('sv');
const sendSharpen = throttled('/zoom?sharpen=');
const sharpenLabel = () => { sv.textContent = parseFloat(sh.value) === 0 ? 'off' : parseFloat(sh.value).toFixed(1) + 'x'; };
sh.addEventListener('input', () => { sharpenLabel(); sendSharpen(sh.value); });
sharpenLabel();
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
for (const el of [video, vh264])
  el.addEventListener('click', (e) => {
    e.preventDefault();
    const r = el.getBoundingClientRect();
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
if (typeof JMuxer === 'undefined' || !window.MediaSource) document.getElementById('stH264').disabled = true;
function requestStream(kind) {
  fstatus.textContent = 'Switching stream (reconfiguring the camera)...';
  fetch('/streammode?kind=' + kind).then(r => r.text()).then(k => { setStream(k); fstatus.textContent = ''; })
    .catch(() => {});
}
document.getElementById('stMjpeg').addEventListener('change', () => requestStream('mjpeg'));
document.getElementById('stH264').addEventListener('change', () => requestStream('h264'));
setStream('STREAM_KIND');
new ResizeObserver(fitVideo).observe(stage);
window.addEventListener('resize', fitVideo);
fitVideo();
// Show the startup centre focus as it happens.
marker.style.display = 'none'; polling = setInterval(pollFocus, 250);
setInterval(() => fetch('/status').then(r => r.json()).then(st => {
  document.getElementById('info').textContent = st.info;
  document.getElementById('sys').textContent = st.sys;
  setStream(st.stream);
}).catch(() => {}), 1000);
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
stats = {"stream_fps": 0.0, "h264_fps": 0.0, "sys": ""}


def af_info_text():
    md = picam2.capture_metadata()
    pos = md.get("LensPosition")
    dist = "unknown" if pos is None else ("infinity" if pos == 0 else f"~{100.0 / pos:.0f}cm")
    fps = 1e6 / md["FrameDuration"] if md.get("FrameDuration") else 0
    return (f"AF: {af_state_names.get(md.get('AfState'), md.get('AfState'))}  "
            f"Lens: {pos if pos is None else round(pos, 2)} ({dist})  "
            f"Zoom {view['zoom']:.2f}x sharpness {md.get('Sharpness', view['sharpness']):.1f}  "
            f"Crop: {md.get('ScalerCrop')}  "
            f"sensor {fps:.1f} fps, streamed MJPEG {stats['stream_fps']:.1f} / H.264 {stats['h264_fps']:.1f} fps  "
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
ZOOM_MAX = 16.0
ZOOM_LIMIT = ZOOM_MAX      # per mode: capped so the crop stays >= the ISP's minimum (set at startup)
view = {"zoom": 1.0, "crop": None, "sharpen": 1.0, "sharpness": 1.0}  # current ScalerCrop etc.
# Zoom sharpening, in the ISP (libcamera "Sharpness", 0-16, 1 = tuning
# default). Measured (tools/zoom_sharpen.py): the ISP sharpens *before* its
# upscaler, so it adds edge contrast at sensor scale but can't undo the
# upscale's own blur; 3-6 helped at 8-16x, 12 was mostly halos and noise.
# Post-upscale software unsharp masking (tools/usm_test.py) gained little
# and cost ~100ms/frame at 3072x1728 (> the JPEG encode), so it's not used.
# Sharpness = 1 + strength * log2(zoom): 2x->2, 4x->3, 8x->4, 16x->5 at
# strength 1; strength 0 = off.
SHARPEN_MAX_STRENGTH = 2.0


def zoom_sharpness(level, strength):
    return min(16.0, 1.0 + strength * math.log2(max(level, 1.0)))
view_lock = threading.Lock()
# Phase AF needs enough of the sensor's 16x12 PDAF grid (which covers the raw
# readout window, not the zoomed view), so when zoomed in the AF window is
# kept at least this fraction of the 1:1 crop.
AF_WIN_MIN_FRAC = 0.4  # 0.25 at 3x zoom fell back to a contrast scan (3.2s vs ~0.9s)


def zoom_crop(level):
    cx0, cy0, cw, ch = CROP
    # Height first, width from it: at 720p's ~11x limit the crop is only ~64
    # rows, where rounding the other way was ~2% off the aspect ratio.
    h = int(ch / level) & ~1
    w = int(round(h * cw / ch / 2)) * 2
    return (cx0 + (cw - w) // 2, cy0 + (ch - h) // 2, w, h)


def set_zoom(level=None, sharpen=None):
    with view_lock:
        level = view["zoom"] if level is None else min(max(level, 1.0), ZOOM_LIMIT)
        sharpen = view["sharpen"] if sharpen is None else min(max(sharpen, 0.0), SHARPEN_MAX_STRENGTH)
        crop = zoom_crop(level)
        sharpness = zoom_sharpness(level, sharpen)
        view.update(zoom=level, crop=crop, sharpen=sharpen, sharpness=sharpness)
    picam2.set_controls({"ScalerCrop": crop, "Sharpness": sharpness})
    return level, crop, sharpness
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
    """Laplacian variance of `roi` (fractions of the frame). `frame` is main
    BGR in the MJPEG configuration, the lores Y plane in the H.264 one."""
    fh, fw = frame.shape[:2]
    x, y, w, h = (int(roi[0] * fw), int(roi[1] * fh), int(roi[2] * fw), int(roi[3] * fh))
    g = frame[y:y + h, x:x + w]
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
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
    roi = (roi[0] / ow, roi[1] / oh, roi[2] / ow, roi[3] / oh)  # as fractions: frame size depends on the stream
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
    af_ctl["mode"] = 2
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
    af_ctl["mode"] = 0
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
        kind = []  # frame type of the first scored frame; scores across types don't compare

        def take(seq):
            got = _wait_frame(seq)
            if got is None:
                return None
            seq, frame, md = got
            if not kind:
                kind.append(frame.ndim)
            if md["lens"] is not None and frame.ndim == kind[0]:
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
clients = {"mjpeg": 0}  # MJPEG viewers; JPEG encoding (and main copies) only while > 0
clients_lock = threading.Lock()


class H264Broadcast(Output):
    """Fans the hardware encoder's access units out to /stream.h264 clients.
    Each client starts at a keyframe (SPS/PPS are repeated on every one), and
    one that falls behind is dropped back to waiting for the next keyframe
    rather than queueing up latency."""
    MAX_QUEUE = 15

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.life = threading.Lock()
        self.subs = []       # per client: {"q": deque, "cv": Condition, "synced": bool}
        self.count = 0
        self.prev_ts = None
        self.encoder = None
        self.stop_timer = None

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        dur = int(1e6 / STREAM_FPS)
        if timestamp is not None and self.prev_ts is not None and 0 < timestamp - self.prev_ts < 2_000_000:
            dur = timestamp - self.prev_ts
        self.prev_ts = timestamp
        rec = struct.pack(">II", len(frame), dur) + bytes(frame)
        with self.lock:
            self.count += 1
            subs = list(self.subs)
        for s in subs:
            with s["cv"]:
                if not s["synced"]:
                    if not keyframe:
                        continue
                    s["synced"] = True
                if len(s["q"]) >= self.MAX_QUEUE:
                    s["q"].clear()
                    s["synced"] = False
                    continue
                s["q"].append(rec)
                s["cv"].notify()

    # Encoder start/stop is serialised by `life`, not `lock`: stop_encoder
    # waits for the encoder thread, which may be inside outputframe (`lock`).
    def subscribe(self):
        s = {"q": deque(), "cv": Condition(), "synced": False}
        with self.life:
            with self.lock:
                self.subs.append(s)
            if self.stop_timer:
                self.stop_timer.cancel()
                self.stop_timer = None
            if stream["kind"] == "h264":
                self.start_if_watched()
        return s

    def unsubscribe(self, s):
        with self.life:
            with self.lock:
                self.subs.remove(s)
                idle = not self.subs
            if idle and self.encoder is not None and not self.stop_timer:
                # Grace period, so a page reload or stream switch-back doesn't
                # restart the encoder.
                self.stop_timer = threading.Timer(5.0, self._stop_if_idle)
                self.stop_timer.daemon = True
                self.stop_timer.start()

    def start_if_watched(self):
        """Start the encoder if anyone is watching (caller holds `life`; the
        camera must be in the H.264 configuration). A fresh encoder per
        start; iperiod 1s bounds how long a new client waits for a keyframe."""
        with self.lock:
            watched = bool(self.subs)
        if watched and self.encoder is None:
            self.encoder = H264Encoder(bitrate=H264_BITRATE, iperiod=max(1, round(STREAM_FPS)),
                                       framerate=STREAM_FPS)
            self.prev_ts = None
            picam2.start_encoder(self.encoder, self, name="lores")

    def stop_now(self):
        """Before reconfiguring the camera (caller holds `life`)."""
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None
        if self.encoder is not None:
            enc, self.encoder = self.encoder, None
            picam2.stop_encoder(enc)

    def _stop_if_idle(self):
        with self.life:
            self.stop_timer = None
            with self.lock:
                if self.subs or self.encoder is None:
                    return
            enc, self.encoder = self.encoder, None
            picam2.stop_encoder(enc)


h264_out = H264Broadcast()


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
        with cam_lock:  # held off while switch_stream reconfigures the camera
            mjpeg = stream["kind"] == "mjpeg"
            req = picam2.capture_request()
            try:
                # H.264: only the lores Y plane, for the focus scoring (the
                # hardware encoder takes the lores buffer itself).
                frame = req.make_array("main") if mjpeg else                     req.make_array("lores")[:H264_SIZE[1], :H264_SIZE[0]]
                md = req.get_metadata()
            finally:
                req.release()
                # Drop the reference too: a lingering request object keeps
                # the old libcamera camera alive across switch_stream's reopen.
                req = None
        mjpeg = mjpeg and clients["mjpeg"] > 0
        seq += 1
        with latest_cv:
            latest.update(seq=seq, frame=frame, lens=md.get("LensPosition"), af_state=md.get("AfState"),
                          sensor_temp=md.get("SensorTemperature"))
            latest_cv.notify_all()
        if mjpeg and slots.acquire(blocking=False):
            pool.submit(_encode_and_publish, seq, frame, slots)


cam_lock = threading.Lock()
stream = {"kind": START_STREAM}
af_ctl = {"mode": 0}  # AfMode last set (0 manual, 2 continuous), re-applied after a switch


def camera_config(kind):
    if kind == "h264":
        # The encoder takes lores, not a YUV main: zoomed in (ISP upscaling),
        # a lone 1080p YUV420 main held only ~21.5-22 fps in 3k, while a
        # 1080p lores (beside an equally small RGB main, which lores can't
        # exceed) held 24.8 at 4x and 16x (tools/h264_test.py, SINGLE=1 vs
        # MAIN=1920x1080) -- the ISP's main output upscales YUV slowly.
        streams = {"main": {"size": H264_SIZE, "format": "RGB888"},
                   "lores": {"size": H264_SIZE, "format": "YUV420"}}
        buffers = 6
    else:
        streams, buffers = {"main": {"size": OUT_SIZE, "format": "RGB888"}}, BUFFER_COUNT
    return picam2.create_video_configuration(**streams, raw={"size": RAW_SIZE}, buffer_count=buffers,
                                             controls={"FrameDurationLimits": FRAME_LIMITS})


def open_camera():
    return Picamera2(camera_num=CAMERA_NUM, tuning=Picamera2.load_tuning_file(TUNING_FILE))


def switch_stream(kind):
    """Reconfigure the camera for `kind` ("mjpeg" / "h264"), keeping zoom,
    sharpening and focus. ~1.3s: the camera is closed and reopened, because
    after a plain stop/configure/start (~0.5s) the lens stops responding --
    LensPosition commands ignored, metadata frozen (tools/reconfig_lens_test.py);
    a fresh IPA doesn't have that problem."""
    global picam2
    with cam_lock, h264_out.life:
        if stream["kind"] == kind:
            return
        cancel_focus_job()
        lens = latest["lens"]
        h264_out.stop_now()
        picam2.stop()
        picam2.close()
        # The reopen only resets the lens if libcamera's camera manager (and
        # with it the IPA) is really destroyed, which needs every Python
        # reference to the old camera gone -- hence capture_loop's req = None.
        picam2 = None
        gc.collect()
        picam2 = open_camera()
        picam2.configure(camera_config(kind))
        stream["kind"] = kind
        with view_lock:
            ctl = {"ScalerCrop": view["crop"], "Sharpness": view["sharpness"]}
        ctl["AfMode"] = af_ctl["mode"]
        if af_ctl["mode"] == 0 and lens is not None:
            ctl["LensPosition"] = lens
        picam2.set_controls(ctl)
        picam2.start()
        if kind == "h264":
            h264_out.start_if_watched()


def _read(path, default=None):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _cpu_jiffies():
    """(busy, total) jiffies over all cores, from /proc/stat."""
    v = [int(x) for x in _read("/proc/stat", "cpu 0 0 0 0").splitlines()[0].split()[1:]]
    idle = v[3] + (v[4] if len(v) > 4 else 0)  # idle + iowait
    return sum(v) - idle, sum(v)


def _throttled():
    """vcgencmd get_throttled, decoded (bits 0-3 now, 16-19 since boot)."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=1).stdout
        bits = int(out.strip().split("=")[1], 16)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "?"
    names = {0: "under-voltage", 1: "freq capped", 2: "THROTTLED", 3: "soft temp limit"}
    now = [n for b, n in names.items() if bits & (1 << b)]
    past = [n for b, n in names.items() if bits & (1 << (b + 16)) and not bits & (1 << b)]
    return (", ".join(now) or "ok") + (f" (since boot: {', '.join(past)})" if past else "")


def stream_fps_loop():
    """Stream fps plus system load/temps for the page, every 2s."""
    ncpu = os.cpu_count() or 1
    while True:
        with encoded_lock:
            start = encoded["count"]
        start_h = h264_out.count
        busy0, total0 = _cpu_jiffies()
        t0, p0 = time.monotonic(), os.times()
        time.sleep(2)
        el = time.monotonic() - t0
        with encoded_lock:
            stats["stream_fps"] = (encoded["count"] - start) / el
        stats["h264_fps"] = (h264_out.count - start_h) / el
        busy1, total1 = _cpu_jiffies()
        p1 = os.times()
        cpu = 100 * (busy1 - busy0) / max(1, total1 - total0)
        proc = 100 * (p1.user + p1.system - p0.user - p0.system) / el  # % of one core
        soc = _read("/sys/class/thermal/thermal_zone0/temp")
        mhz = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        with latest_cv:
            sensor_t = latest.get("sensor_temp")
        stats["sys"] = (f"CPU {cpu:.0f}% of {ncpu} cores (this server {proc:.0f}% of one core)"
                        + (f" @ {int(mhz) // 1000} MHz" if mhz else "")
                        + (f"  SoC {int(soc) / 1000:.1f}\u00b0C" if soc else "")
                        + (f"  sensor {sensor_t:.0f}\u00b0C" if sensor_t is not None else "")
                        + f"  throttling: {_throttled()}")


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
                    .replace("ZOOM_LIMIT", f"{ZOOM_LIMIT:.3f}").replace("ZOOM_NOW", f"{view['zoom']:.3f}")
                    .replace("SHARPEN_NOW", f"{view['sharpen']:.2f}")
                    .replace("SHARPEN_MAX", str(SHARPEN_MAX_STRENGTH))
                    .replace("OUT_W", str(OUT_SIZE[0])).replace("OUT_H", str(OUT_SIZE[1]))
                    .replace("H264_W", str(H264_SIZE[0])).replace("H264_H", str(H264_SIZE[1]))
                    .replace("STREAM_FPS", f"{STREAM_FPS:.2f}").replace("STREAM_KIND", stream["kind"])
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
            af_ctl["mode"] = 0
            self._send(f"ok {pos:.2f}".encode())
        elif url.path == "/afmode":
            # Continuous AF uses the default (centre-weighted) metering again.
            from libcamera import controls
            cancel_focus_job()
            af_ctl["mode"] = 0 if qs.get("mode", [""])[0] == "manual" else 2
            picam2.set_controls({"AfMetering": controls.AfMeteringEnum.Auto, "AfMode": af_ctl["mode"]})
            self._send(b"ok")
        elif url.path == "/zoom":
            try:
                level = float(qs["level"][0]) if "level" in qs else None
                sharpen = float(qs["sharpen"][0]) if "sharpen" in qs else None
            except ValueError:
                level = sharpen = None
            level, crop, sharpness = set_zoom(level, sharpen)
            self._send(f"ok zoom {level:.2f} crop {crop} sharpness {sharpness:.2f}".encode())
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
        elif url.path == "/streammode":
            # Server-wide: reconfigures the camera, so every viewer switches.
            kind = qs.get("kind", [""])[0]
            if kind in ("mjpeg", "h264"):
                switch_stream(kind)
            self._send(stream["kind"].encode())
        elif url.path == "/status":
            body = json.dumps({"info": af_info_text(), "sys": stats["sys"], "stream": stream["kind"]})
            self._send(body.encode(), "application/json")
        elif url.path == "/afinfo":
            self._send(af_info_text().encode())
        elif url.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            with clients_lock:
                clients["mjpeg"] += 1
            try:
                while True:
                    with output.condition:
                        # Timeout: a client that connects while nothing is
                        # encoding must not wait forever (encoding starts on
                        # the next frame now that it is counted).
                        if not output.condition.wait(timeout=2.0):
                            continue
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError) as e:
                logging.info("Client %s disconnected: %s", self.client_address, e)
            finally:
                with clients_lock:
                    clients["mjpeg"] -= 1
        elif url.path == "/stream.h264":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            sub = h264_out.subscribe()
            try:
                while True:
                    with sub["cv"]:
                        while not sub["q"]:
                            sub["cv"].wait()
                        recs = list(sub["q"])
                        sub["q"].clear()
                    self.wfile.write(b"".join(recs))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError) as e:
                logging.info("H.264 client %s disconnected: %s", self.client_address, e)
            finally:
                h264_out.unsubscribe(sub)
        elif url.path == "/jmuxer.min.js":
            with open(os.path.join(STATIC_DIR, "jmuxer.min.js"), "rb") as f:
                self._send(f.read(), "application/javascript")
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
    picam2 = open_camera()
    picam2.configure(camera_config(stream["kind"]))
    picam2.start()

    # ScalerCrop bounds are only valid after configure(); max_crop is the
    # raw mode's whole window in full-sensor coordinates.
    x, y, w, h = picam2.camera_controls["ScalerCrop"][1]
    crop = (x + (w - OUT_SIZE[0]) // 2, y + (h - OUT_SIZE[1]) // 2, OUT_SIZE[0], OUT_SIZE[1])
    CROP = crop
    view["crop"] = crop
    # Highest zoom whose crop stays >= the ISP's minimum ScalerCrop (64x64 here)
    # in both dimensions: 720p/full ~11x, 1080p and up the full 16x.
    _min_w, _min_h = picam2.camera_controls["ScalerCrop"][0][2:]
    ZOOM_LIMIT = min(ZOOM_MAX, crop[2] / (_min_w + 2), crop[3] / (_min_h + 2))
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
