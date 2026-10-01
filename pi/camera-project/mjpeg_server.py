#!/usr/bin/env python3
"""Live MJPEG stream from Camera 1 (Arducam 64MP), validating true sensor-crop
zoom: the zoom slider switches between distinct sensor modes (each a real,
fixed crop window programmed into the sensor by the custom arducam_64mp
driver), not ISP ScalerCrop of a single fixed mode. Levels 1-6 are 1x1
(unbinned) real sensor crops; level 0 is the stock 4x4-binned wide mode kept
as a fast, wide reference.

Architecture note: each zoom level's camera runs in its OWN subprocess,
spawned fresh on every switch, rather than reconfiguring one long-lived
Picamera2 object in-process. This was forced by testing: reusing one
Picamera2 object across repeated stop()/configure()/start() cycles between
these custom raw modes reliably failed after exactly 3 reconfigures with a
"Camera frontend has timed out" error, regardless of buffer size, buffer
count, added settle delays, or forcing gc.collect() -- pointing at a small
fixed-size resource pool in the libcamera/VC4 pipeline that isn't released
between in-process reconfigures. Standalone `rpicam-still` calls (a fresh
process each time) never once failed across dozens of invocations of every
mode, so a fresh subprocess per switch mirrors that reliably-working pattern.
This process is the supervisor: it owns the HTTP server and relays frames
and focus commands to/from whichever worker subprocess is currently active.
"""
import gc
import json
import logging
import multiprocessing as mp
import os
import queue
import socketserver
import threading
import time
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs

TUNING_FILE = os.path.join(os.path.dirname(__file__), "tuning_64mp_sharper.json")

# Picamera2's camera_num indexes into Picamera2.global_camera_info(), which
# is NOT a fixed physical-port mapping -- it depends on which cameras are
# actually present. arducam_64mp was index 1 while the imx477 (now removed)
# was also attached; with only this one camera left, it's index 0. Confirmed
# via Picamera2.global_camera_info() directly after the reboot that removed
# imx477 -- don't assume this stays 0 if another camera is ever added back.
CAMERA_NUM = 0

# Each level is a real, distinct sensor mode -- see arducam_64mp.c. Level 0 is
# the stock 4x4-binned mode (no `raw` override needed, already the default
# pick for that main size). Levels 1-6 are the custom 1x1-unbinned centre-crop
# modes added this session (register math: x_start=(9248-width)/2,
# y_start=(6944-height)/2, width=16k, height=12k -- see the driver source for
# the full derivation). zoom = 9248 / width (full sensor width / this mode's
# crop width). fps is the real per-mode value as reported by
# `rpicam-hello --list-cameras` (not a guess) -- the driver clamps
# frame_length to at least the mode's height, so the three narrowest crop
# modes hit the ~10fps nominal target but the three widest don't.
ZOOM_LEVELS = [
    {"size": (1920, 1080), "raw": None, "zoom": 9248 / 7680, "fps": 60.04},
    {"size": (3200, 2400), "raw": (3200, 2400), "zoom": 9248 / 3200, "fps": 8.02},
    {"size": (2880, 2160), "raw": (2880, 2160), "zoom": 9248 / 2880, "fps": 8.91},
    {"size": (2560, 1920), "raw": (2560, 1920), "zoom": 9248 / 2560, "fps": 10.00},
    {"size": (2320, 1740), "raw": (2320, 1740), "zoom": 9248 / 2320, "fps": 10.00},
    {"size": (2080, 1560), "raw": (2080, 1560), "zoom": 9248 / 2080, "fps": 10.00},
    {"size": (1968, 1476), "raw": (1968, 1476), "zoom": 9248 / 1968, "fps": 10.00},
]

# Bounding box the live view fits into, preserving each mode's true captured
# aspect ratio (no forced letterbox crop -- level 0 is 16:9, the rest are
# 4:3). Also the actual "main" stream size requested from the ISP -- letting
# it downscale at capture time instead of shipping full-native-crop frames
# (up to 23MB/frame RGB888) cuts per-switch buffer footprint 5-15x, which
# was necessary to get anywhere near reliable in-process reconfiguration
# before the subprocess-per-level redesign made that moot; kept anyway since
# it's still strictly less data to move through IPC and JPEG-encode.
DISPLAY_MAX = (1280, 960)

SW_ZOOM_MIN = 1.0
SW_ZOOM_MAX = 4.0  # secondary/minor control -- now an ISP ScalerCrop applied
                    # in the worker (see camera_worker's "swzoom" command),
                    # not a software crop+resize -- always crisper than a
                    # true digital upscale would be, but still just a crop
                    # of the same captured frame, not real sensor zoom.

# Software JPEG quality (libjpeg-turbo via cv2.imencode, in the worker).
# Raised from an earlier 75 -- measured directly on this board that encode
# time is essentially flat from 75-90 (dominated by DCT/color-conversion
# cost, not entropy coding), so this is close to free; re-verify against the
# real capture+encode loop's fps if this is ever pushed higher than 85-90.
JPEG_QUALITY = 85

# How long the supervisor waits for a worker's response to a control command
# (focus/afinfo/etc) before giving up -- generous, since a refocus scan can
# legitimately take up to ~4s.
CONTROL_TIMEOUT = 6.0


def _fit(size, max_size):
    """Aspect-preserving fit of `size` within `max_size`."""
    w, h = size
    mw, mh = max_size
    scale = min(mw / w, mh / h)
    return (max(1, int(w * scale)), max(1, int(h * scale)))


# ---------------------------------------------------------------------------
# Camera worker: runs in its own subprocess, owns exactly one Picamera2
# instance for exactly one zoom level, for its entire lifetime. Must be
# defined at module level (picklable) since it's spawned via
# multiprocessing's 'spawn' context, which re-imports this file fresh in the
# child -- nothing at module level here does any camera/network work, only
# this function (called explicitly by the supervisor) does.
# ---------------------------------------------------------------------------
def camera_worker(sensor_size, raw_size, display_size, frame_queue, cmd_queue, resp_queue, stop_event):
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder, Quality
    import cv2

    tuning = Picamera2.load_tuning_file(TUNING_FILE)
    picam2 = Picamera2(camera_num=CAMERA_NUM, tuning=tuning)
    # "RGB888" (really BGR-ordered, see the old capture_loop's comment on
    # this Picamera2 quirk) is natively supported by the V4L2 hardware
    # encoder's format table (drivers/media/i2c mapping in
    # picamera2/encoders/v4l2_encoder.py: RGB888 -> V4L2_PIX_FMT_BGR24) --
    # confirmed by reading that source directly, so there was no need to
    # introduce a YUV420 main stream to make hardware encoding work.
    picam2.configure(picam2.create_video_configuration(
        main={"size": display_size, "format": "RGB888"},
        raw={"size": raw_size},
        buffer_count=2,
    ))

    # Base ScalerCrop bounds for this mode, used by the "swzoom" command
    # below -- must be read after configure() (camera_controls reflects the
    # configured mode's real crop bounds only then).
    _min_crop, max_crop, _default_crop = picam2.camera_controls["ScalerCrop"]
    base_x, base_y, base_w, base_h = max_crop
    center_x = base_x + base_w / 2
    center_y = base_y + base_h / 2

    # Hardware MJPEG encode (VideoCore VPU, /dev/video11 via Picamera2's
    # MJPEGEncoder) was tried here and reverted: measured directly on this
    # board, at this exact config (1280x960 main, 1968x1476 raw), it
    # delivered only ~4.5fps versus ~8.6fps for the plain capture_array() +
    # cv2.imencode() loop below -- the V4L2 M2M round-trip is slower than
    # software encode at this resolution, not faster, despite moving work
    # off the ARM cores. Confirmed via a standalone isolated benchmark (no
    # IPC/queue involved) before reverting, so this isn't an artifact of the
    # surrounding architecture -- the hardware path itself is the bottleneck.
    # Encoding still happens here in the worker rather than the supervisor,
    # though: shipping compressed JPEG (~50-100KB) instead of a raw frame
    # (~3.6MB) through frame_queue is a real, independent win kept from that
    # attempt, regardless of which encoder produces the JPEG.
    picam2.start()

    has_af = "AfMode" in picam2.camera_controls
    if has_af:
        picam2.set_controls({"AfMode": 2})
    lens_min, lens_max, _ = picam2.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))

    af_state_names = {0: "Idle", 1: "Scanning", 2: "Focused", 3: "Failed"}

    def af_info_text():
        md = picam2.capture_metadata()
        lens_pos = md.get("LensPosition")
        af_state = md.get("AfState")
        fom = md.get("FocusFoM")
        state_name = af_state_names.get(af_state, str(af_state))
        if lens_pos is None:
            dist = "unknown"
        elif lens_pos == 0:
            dist = "infinity"
        else:
            dist = f"~{100.0 / lens_pos:.0f}cm"
        return f"State: {state_name}  LensPosition: {lens_pos}  ~Focus distance: {dist}  FocusFoM: {fom}"

    def handle_commands():
        while not stop_event.is_set():
            try:
                cmd = cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            kind = cmd.get("type")
            try:
                if kind == "afmode":
                    if has_af:
                        picam2.set_controls({"AfMode": 0 if cmd["mode"] == "manual" else 2})
                    resp_queue.put({"ok": True})
                elif kind == "afrange":
                    if has_af:
                        picam2.set_controls({"AfRange": max(0, min(cmd["range"], 2))})
                    resp_queue.put({"ok": True})
                elif kind == "focus":
                    pos = max(lens_min, min(cmd["position"], lens_max))
                    if has_af:
                        picam2.set_controls({"AfMode": 0, "LensPosition": pos})
                    resp_queue.put({"position": pos})
                elif kind == "refocus":
                    # Runs on this command thread, separate from the capture
                    # loop below, so blocking here for up to ~4s doesn't
                    # stall the live frame stream.
                    if has_af:
                        picam2.set_controls({"AfMode": 1})  # Auto
                        picam2.set_controls({"AfTrigger": 0})  # Start
                        for _ in range(40):
                            time.sleep(0.1)
                            md = picam2.capture_metadata()
                            if md.get("AfState") in (2, 3):
                                break
                        picam2.set_controls({"AfMode": 2})  # back to Continuous
                    resp_queue.put({"ok": True})
                elif kind == "afinfo":
                    resp_queue.put({"text": af_info_text()})
                elif kind == "lens_range":
                    resp_queue.put({"min": lens_min, "max": lens_max})
                elif kind == "swzoom":
                    # "Extra digital zoom": now a hardware ISP crop
                    # (ScalerCrop) instead of a supervisor-side software
                    # crop+resize -- an upgrade, not just a workaround for
                    # moving encoding into this process, since it's now the
                    # ISP doing the crop/scale rather than cv2 in Python.
                    level = max(1.0, min(cmd["level"], SW_ZOOM_MAX))
                    w = int(base_w / level)
                    h = int(base_h / level)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    picam2.set_controls({"ScalerCrop": (x, y, w, h)})
                    resp_queue.put({"level": level})
            except Exception as e:
                resp_queue.put({"error": str(e)})

    cmd_thread = threading.Thread(target=handle_commands, daemon=True)
    cmd_thread.start()

    try:
        while not stop_event.is_set():
            frame = picam2.capture_array("main")
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                try:
                    frame_queue.put(jpg.tobytes(), block=False)
                except queue.Full:
                    pass
    finally:
        picam2.stop()
        picam2.close()


class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


PAGE_TEMPLATE = """\
<html>
<head>
<title>Camera 1 -- Sensor Crop Zoom</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; min-height: 100%%; width: 100%%;
    background: #000; color: #ddd;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  #layout { display: flex; flex-wrap: wrap; min-height: 100vh; width: 100vw; }
  #videowrap {
    flex: 1 1 auto; min-width: 0; min-height: 0;
    display: flex; align-items: center; justify-content: center;
    background: #000;
  }
  #videowrap img { max-width: 100%%; max-height: 100%%; width: auto; height: auto; display: block; }
  #controls {
    flex: 0 0 320px; max-width: 100vw; overflow-y: auto;
    background: #111; padding: 1em; border-left: 1px solid #292929;
  }
  h1 { font-size: 1.1em; margin: 0 0 0.75em; color: #fff; }
  label { color: #ccc; }
  input[type=range] { width: 100%%; accent-color: #4a9eff; }
  select, button {
    background: #222; color: #eee; border: 1px solid #444;
    border-radius: 4px; padding: 0.35em 0.6em; font-size: 0.9em;
  }
  button:active { background: #333; }
  .panel { margin-top: 1.2em; padding: 0.75em; border: 1px solid #292929; border-radius: 6px; }
  .hint { color: #888; font-size: 0.8em; margin: 0.4em 0 0; }
  .divider { margin-top: 1.4em; padding-top: 1em; border-top: 1px solid #292929; }
  .minor { opacity: 0.75; }
  #afinfo { font-family: ui-monospace, monospace; font-size: 0.85em; margin-top: 0.3em; color: #9fdca0; }
  #zoominfo { font-family: ui-monospace, monospace; font-size: 0.85em; margin-top: 0.3em; color: #9fdca0; }
  #switching { color: #e0a030; font-size: 0.8em; margin-top: 0.3em; display: none; }
</style>
</head>
<body>
<div id="layout">

<div id="videowrap">
  <img id="video" src="stream.mjpg" />
</div>

<div id="controls">
  <h1>Sensor Crop Zoom</h1>

  <div>
    <label>Zoom level (real sensor crop mode):
      <input type="range" id="zoom" min="0" max="%(max_level)d" step="1" value="0">
    </label>
    <div id="zoominfo">loading...</div>
    <div id="switching">switching sensor mode...</div>
  </div>
  <p class="hint">
    Each step switches the sensor to a genuinely different, fixed crop mode
    programmed into the driver -- level 0 is the stock 4x4-binned wide mode
    (fast, lower per-pixel detail); levels 1-6 are real 1x1 (unbinned) sensor
    crops, progressively narrower and sharper. This is <b>not</b> ISP digital
    crop of one fixed mode -- switching levels restarts the camera in a fresh
    process (~1-2s) and frame rate varies by level (shown above).
  </p>

  <div class="divider minor">
    <label>Extra digital zoom (ISP crop of the current mode -- not the thing being tested):
      <input type="range" id="swzoom" min="1.0" max="%(sw_max)s" step="0.1" value="1.0">
    </label>
    <span id="swzoomval">1.0x</span>
  </div>

  <div class="panel">
    <div><b>Focus</b></div>
    <div id="afinfo">loading...</div>

    <div style="margin-top: 0.6em;">
      <label>Mode:
        <select id="afmode">
          <option value="continuous">Continuous (auto)</option>
          <option value="manual">Manual</option>
        </select>
      </label>
    </div>

    <div id="continuousControls" style="margin-top: 0.5em;">
      <label>AF range:
        <select id="afrange">
          <option value="0">Normal</option>
          <option value="1">Macro (close-up)</option>
          <option value="2">Full</option>
        </select>
      </label>
      <button id="refocus" style="margin-left: 0.5em;">Refocus now</button>
    </div>

    <div id="manualControls" style="margin-top: 0.5em; display: none;">
      <label>Focus distance:
        <input type="range" id="focusSlider" min="%(lens_min)s" max="%(lens_max)s" step="0.01" value="%(lens_min)s">
      </label>
      <span id="focusVal">infinity</span>
      <div style="margin-top: 0.4em;">
        <button id="focusFarBig" type="button">&laquo; far</button>
        <button id="focusFarSmall" type="button">&lsaquo;</button>
        <button id="focusNearSmall" type="button">&rsaquo;</button>
        <button id="focusNearBig" type="button">near &raquo;</button>
      </div>
      <p class="hint">
        This slider moves in dioptres (1/distance), not distance -- the
        far end is naturally very sensitive (a tiny move changes the focus
        distance by metres) and the near end very fine (the same move is
        only millimetres). The buttons nudge by a fixed dioptre amount
        (&laquo;/&raquo; = 0.5, &lsaquo;/&rsaquo; = 0.05) for precise
        adjustment without needing to drag the slider exactly, which is
        fiddly on a touchscreen.
      </p>
    </div>
  </div>

</div>

</div>

<script>
const ZOOM_INFO = %(zoom_info_json)s;

function updateZoomInfo(level) {
  const z = ZOOM_INFO[level];
  document.getElementById('zoominfo').textContent =
    z.width + "x" + z.height + "  --  " + z.zoom.toFixed(2) + "x  --  " + z.fps.toFixed(2) + " fps";
}

const zoomSlider = document.getElementById('zoom');
const switching = document.getElementById('switching');
const videoImg = document.getElementById('video');
let zoomPending = false, zoomQueued = null;
function sendZoom(level) {
  if (zoomPending) { zoomQueued = level; return; }
  zoomPending = true;
  switching.style.display = 'block';
  fetch('/zoom?level=' + level).then(r => r.json()).then(info => {
    ZOOM_INFO[level] = info;
    updateZoomInfo(level);
    // stream.mjpg's connection belonged to the old worker -- reconnect.
    videoImg.src = 'stream.mjpg?_=' + Date.now();
  }).finally(() => {
    switching.style.display = 'none';
    zoomPending = false;
    if (zoomQueued !== null) { const q = zoomQueued; zoomQueued = null; sendZoom(q); }
  });
}
zoomSlider.addEventListener('input', () => {
  const level = parseInt(zoomSlider.value, 10);
  updateZoomInfo(level);
  sendZoom(level);
});
updateZoomInfo(0);

const swSlider = document.getElementById('swzoom');
const swLabel = document.getElementById('swzoomval');
let swPending = false, swQueued = null;
function sendSwZoom(level) {
  if (swPending) { swQueued = level; return; }
  swPending = true;
  fetch('/swzoom?level=' + level).finally(() => {
    swPending = false;
    if (swQueued !== null) { const q = swQueued; swQueued = null; sendSwZoom(q); }
  });
}
swSlider.addEventListener('input', () => {
  swLabel.textContent = parseFloat(swSlider.value).toFixed(1) + 'x';
  sendSwZoom(swSlider.value);
});

document.getElementById('afrange').addEventListener('change', (e) => {
  fetch('/afrange?range=' + e.target.value);
});
document.getElementById('refocus').addEventListener('click', () => {
  fetch('/refocus');
});

function focusLabel(pos) {
  if (pos <= 0) return 'infinity';
  const cm = 100.0 / pos;
  return (cm < 100 ? cm.toFixed(1) : cm.toFixed(0)) + 'cm';
}

const focusSlider = document.getElementById('focusSlider');
const focusMin = parseFloat(focusSlider.min);
const focusMax = parseFloat(focusSlider.max);
let focusPending = false, focusQueued = null;
function sendFocus(pos) {
  if (focusPending) { focusQueued = pos; return; }
  focusPending = true;
  fetch('/focus?position=' + pos).finally(() => {
    focusPending = false;
    if (focusQueued !== null) { const q = focusQueued; focusQueued = null; sendFocus(q); }
  });
}
function applyFocus(pos) {
  pos = Math.max(focusMin, Math.min(pos, focusMax));
  focusSlider.value = pos;
  document.getElementById('focusVal').textContent = focusLabel(pos);
  sendFocus(pos);
}
focusSlider.addEventListener('input', (e) => applyFocus(parseFloat(e.target.value)));

function nudgeFocus(delta) {
  applyFocus(parseFloat(focusSlider.value) + delta);
}
document.getElementById('focusFarBig').addEventListener('click', () => nudgeFocus(-0.5));
document.getElementById('focusFarSmall').addEventListener('click', () => nudgeFocus(-0.05));
document.getElementById('focusNearSmall').addEventListener('click', () => nudgeFocus(0.05));
document.getElementById('focusNearBig').addEventListener('click', () => nudgeFocus(0.5));

document.getElementById('afmode').addEventListener('change', (e) => {
  const manual = e.target.value === 'manual';
  document.getElementById('manualControls').style.display = manual ? 'block' : 'none';
  document.getElementById('continuousControls').style.display = manual ? 'none' : 'block';
  if (manual) {
    sendFocus(focusSlider.value);
  } else {
    fetch('/afmode?mode=continuous');
  }
});

function pollAfInfo() {
  fetch('/afinfo').then(r => r.text()).then(t => {
    document.getElementById('afinfo').textContent = t;
  }).catch(() => {});
}
setInterval(pollAfInfo, 1000);
pollAfInfo();
</script>
</body>
</html>
"""


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
        elif self.path == "/index.html":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path.startswith("/zoom"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                level = int(qs.get("level", ["0"])[0])
            except ValueError:
                level = 0
            info = set_zoom_level(level)
            self._send_json(info)
        elif self.path.startswith("/swzoom"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                level = float(qs.get("level", ["1.0"])[0])
            except ValueError:
                level = 1.0
            applied = set_sw_zoom(level)
            self._send_text(f"ok {applied:.2f}")
        elif self.path.startswith("/afrange"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                af_range = int(qs.get("range", ["0"])[0])
            except ValueError:
                af_range = 0
            send_control({"type": "afrange", "range": af_range})
            self._send_text("ok")
        elif self.path == "/refocus":
            threading.Thread(target=lambda: send_control({"type": "refocus"}), daemon=True).start()
            self._send_text("refocusing")
        elif self.path.startswith("/afmode"):
            qs = parse_qs(urlparse(self.path).query)
            mode = qs.get("mode", ["continuous"])[0]
            send_control({"type": "afmode", "mode": mode})
            self._send_text("ok")
        elif self.path.startswith("/focus"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                position = float(qs.get("position", ["0"])[0])
            except ValueError:
                position = 0.0
            resp = send_control({"type": "focus", "position": position})
            applied = resp.get("position", position) if resp else position
            self._send_text(f"ok {applied:.2f}")
        elif self.path == "/afinfo":
            resp = send_control({"type": "afinfo"})
            self._send_text(resp.get("text", "unavailable") if resp else "unavailable")
        elif self.path.startswith("/stream.mjpg"):
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
                    self.wfile.write(b"--FRAME\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError) as e:
                logging.info("Client %s disconnected: %s", self.client_address, e)
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass

    def _send_text(self, text):
        body = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


state = {"sw_zoom": 1.0}
state_lock = threading.Lock()

output = StreamingOutput()

# worker holds everything about the currently-running camera subprocess;
# reassigned wholesale (under worker_lock) on every zoom switch. control_lock
# separately serializes command<->response round-trips so two overlapping
# requests (e.g. the 1s /afinfo poll landing during a manual focus drag)
# can't interleave on the shared queues.
worker = None
worker_lock = threading.Lock()
control_lock = threading.Lock()

def _reader_loop():
    """Relays already-hardware-encoded JPEG frames from whichever worker is
    currently active into `output` -- a thin byte relay now (see
    camera_worker's QueueOutput): encoding and the "extra zoom" crop both
    happen in the worker via the ISP/VPU, so there's nothing left to do here
    but move bytes. Survives worker swaps transparently: each iteration
    re-reads the `worker` global, so a switch just means the next iteration
    starts pulling from the new worker's queue."""
    while True:
        w = worker
        if w is None:
            time.sleep(0.05)
            continue
        try:
            frame_bytes = w["frame_queue"].get(timeout=0.3)
        except queue.Empty:
            continue
        except (OSError, EOFError, ValueError):
            # Queue torn down mid-read by a concurrent switch -- benign.
            continue
        output.write(frame_bytes)


def _stop_worker(w):
    """Block until the worker's process has genuinely exited -- this is now
    load-bearing (see _switch_mode): the next worker must never start while
    this one might still hold the camera device open."""
    w["stop_event"].set()
    # A healthy worker's loop checks stop_event every iteration and each
    # iteration normally completes in well under a second (even the slowest
    # mode here is ~8fps), so 1.5s is already generous for the normal case --
    # keeping this short instead matters for the occasional case where the
    # worker's capture_array() has wedged (a "Camera frontend has timed out"
    # condition, seen occasionally even after fixing the old/new startup
    # race -- likely the sensor/CSI hardware occasionally needing longer to
    # settle after a mode-switch than the pipeline's own 1s internal
    # timeout allows for). In that case stop_event is never rechecked (the
    # call never returns), so this timeout is pure "how long we wait before
    # giving up and killing it" -- shorter here means faster recovery.
    w["process"].join(timeout=1.5)
    if w["process"].is_alive():
        w["process"].terminate()  # SIGTERM
        w["process"].join(timeout=1)
    if w["process"].is_alive():
        w["process"].kill()  # SIGKILL -- can't be caught or blocked
        w["process"].join(timeout=1)
    for q in (w["frame_queue"], w["cmd_queue"], w["resp_queue"]):
        q.close()


def _switch_mode(level):
    global worker
    level = max(0, min(level, len(ZOOM_LEVELS) - 1))
    cfg = ZOOM_LEVELS[level]
    with worker_lock:
        old = worker
        # Stop the OLD worker COMPLETELY -- including waiting for its
        # process to actually exit, releasing the physical camera device --
        # before starting the new one. Previously this started the new
        # worker first (to minimize the visible gap), but that meant two
        # separate processes briefly both held /base/soc/.../arducam_64mp@1a
        # open at once. That race, not any hardware/firmware limit, was the
        # real cause of the intermittent "Camera frontend has timed out"
        # failures: confirmed by stress-testing 10 rapid back-to-back mode
        # switches through plain `rpicam-still` alone (no Python, no
        # multiprocessing) with zero failures, which only makes sense if the
        # bug is in how this supervisor was handing the camera off between
        # processes, not in the driver or the hardware.
        worker = None
        if old is not None:
            _stop_worker(old)
            time.sleep(0.3)  # let the kernel/CSI hardware settle after the
                              # old process's fds actually close, before a
                              # fresh process opens the device again.
        display_size = _fit(cfg["size"], DISPLAY_MAX)
        mp_ctx = mp.get_context("spawn")
        frame_queue = mp_ctx.Queue(maxsize=1)
        cmd_queue = mp_ctx.Queue()
        resp_queue = mp_ctx.Queue()
        stop_event = mp_ctx.Event()
        proc = mp_ctx.Process(
            target=camera_worker,
            args=(cfg["size"], cfg["raw"] or cfg["size"], display_size,
                  frame_queue, cmd_queue, resp_queue, stop_event),
            daemon=True,
        )
        proc.start()
        worker = {
            "level": level, "process": proc, "frame_queue": frame_queue,
            "cmd_queue": cmd_queue, "resp_queue": resp_queue,
            "stop_event": stop_event, "display_size": display_size,
        }
        gc.collect()
        # A fresh worker always starts at ScalerCrop=full-frame (swzoom
        # 1.0x) -- re-apply whatever "extra zoom" level was in effect so it
        # survives a zoom-level switch, matching how focus settings are
        # expected to persist.
        with state_lock:
            sw_zoom = state["sw_zoom"]
        if sw_zoom > 1.0:
            send_control({"type": "swzoom", "level": sw_zoom})
        return cfg


def set_zoom_level(level):
    cfg = _switch_mode(level)
    return {
        "width": cfg["size"][0], "height": cfg["size"][1],
        "zoom": cfg["zoom"], "fps": cfg["fps"],
    }


def set_sw_zoom(level):
    level = max(SW_ZOOM_MIN, min(level, SW_ZOOM_MAX))
    with state_lock:
        state["sw_zoom"] = level
    send_control({"type": "swzoom", "level": level})
    return level


def send_control(cmd):
    """Send a control command to the current worker and wait for its
    response. Returns None (rather than raising) if there's no worker yet or
    the round-trip times out -- callers already handle a missing response
    gracefully (e.g. /afinfo shows 'unavailable')."""
    w = worker
    if w is None:
        return None
    with control_lock:
        try:
            w["cmd_queue"].put(cmd)
            return w["resp_queue"].get(timeout=CONTROL_TIMEOUT)
        except (queue.Empty, OSError, EOFError, ValueError):
            return None


if __name__ == "__main__":
    # Only the supervisor process reaches this point -- camera_worker
    # subprocesses (spawned via multiprocessing's 'spawn' context, which
    # re-imports this file fresh) never execute anything below this guard.
    from picamera2 import Picamera2

    # Probe LensPosition bounds once via a throwaway Picamera2 instance --
    # camera_controls is available right after construction, no configure()
    # or start() needed, so this doesn't conflict with the real worker
    # opening the camera afterward.
    _probe = Picamera2(camera_num=CAMERA_NUM)
    LENS_MIN, LENS_MAX, _ = _probe.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))
    _probe.close()

    PAGE = PAGE_TEMPLATE % {
        "max_level": len(ZOOM_LEVELS) - 1,
        "sw_max": SW_ZOOM_MAX,
        "lens_min": LENS_MIN,
        "lens_max": LENS_MAX,
        "zoom_info_json": json.dumps([
            {"width": z["size"][0], "height": z["size"][1], "zoom": z["zoom"], "fps": z["fps"]}
            for z in ZOOM_LEVELS
        ]),
    }

    _switch_mode(0)
    reader_thread = threading.Thread(target=_reader_loop, daemon=True)
    reader_thread.start()

    try:
        address = ("0.0.0.0", 8080)
        http_server = StreamingServer(address, StreamingHandler)
        print("Serving sensor-crop-zoom live view on :8080/index.html")
        http_server.serve_forever()
    finally:
        if worker is not None:
            _stop_worker(worker)
