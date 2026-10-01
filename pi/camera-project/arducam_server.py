#!/usr/bin/env python3
"""Live MJPEG stream from the Arducam USB3 48MP webcam, with browser-
controlled software zoom, digital stabilization, manual focus (plus a
software autofocus sweep), and image controls (white balance, exposure,
gain, brightness/contrast/saturation).

Mirrors mjpeg_server.py's architecture (Camera 1 / CSI 64MP): a parallel
JPEG-encoder thread pool and a decoupled stabilization worker thread. This
is a UVC webcam captured via OpenCV/V4L2 though, not picamera2, so there's
no ScalerCrop-style hardware zoom or picamera2 AF -- zoom here is always a
software crop+upscale, and focus is a raw V4L2 "focus_absolute" control
with no continuous-AF hardware behind it (hence the sweep-based
autofocus-once feature, which is not something built into this camera).

Despite being sold as "48MP", this Pi's USB3 SuperSpeed controller isn't
active (verified via `lsusb -t`: shows 480M/USB2 High-Speed only), and at
that link speed this camera's UVC descriptor advertises exactly ONE video
streaming mode: 1280x720 YUYV @10fps. That's a hardware bandwidth ceiling
on this board, not a bug in this script or something a different
resolution/format request can work around. "Best this camera can offer"
here means best focus/exposure/white-balance/gain at that fixed
resolution, not more pixels or fps.
"""
import logging
import queue
import re
import socketserver
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

# by-id path is stable across reboots; /dev/videoN numbering depends on
# USB/CSI probe order and has landed on this device before now.
DEVICE = "/dev/v4l/by-id/usb-Arducam_Arducam_B0478__USB3_48MP_-video-index0"

# Only mode this camera's UVC descriptor exposes at USB2 High-Speed.
CAPTURE_SIZE = (1280, 720)
FPS = 10

# Output is captured slightly smaller than the sensor frame; the extra
# margin is what stabilization "spends" panning the crop window around to
# cancel jitter. Bigger margin = more stabilization headroom, less final FOV.
OUTPUT_SIZE = (1152, 648)
MARGIN_X = (CAPTURE_SIZE[0] - OUTPUT_SIZE[0]) // 2
MARGIN_Y = (CAPTURE_SIZE[1] - OUTPUT_SIZE[1]) // 2

ZOOM_MIN = 1.0
# No hardware sensor-crop control on a UVC webcam (unlike picamera2's
# ScalerCrop) and no headroom beyond the single 1280x720 capture mode --
# every bit of zoom here is a software crop of that fixed buffer followed
# by an upscale back to OUTPUT_SIZE. Unlike Camera 1, there is no "crisp"
# portion of this slider at all: 1.0x is already the sharpest this camera
# can produce, and it gets progressively softer from there. If you need
# more real detail on a subject, physically moving the camera closer will
# always beat digital zoom here.
ZOOM_MAX = 4.0

MOTION_DOWNSCALE = 4        # motion estimated on a 1280/4 x 720/4 image
SMOOTHING_ALPHA = 0.03      # lower = stronger stabilization, slower to follow intentional pans
JPEG_QUALITY = 85


def _query_ctrl_range(name):
    out = subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, "--list-ctrls"], capture_output=True, text=True, check=True
    ).stdout
    m = re.search(rf"{name}\s+0x[0-9a-f]+ \(\w+\)\s*:\s*min=(-?\d+) max=(-?\d+)", out)
    if not m:
        raise RuntimeError(f"Control '{name}' not found on {DEVICE}")
    return int(m.group(1)), int(m.group(2))


def set_ctrl(name, value):
    subprocess.run(["v4l2-ctl", "-d", DEVICE, f"--set-ctrl={name}={int(value)}"], check=True)


# Queried rather than hardcoded so this keeps working if a different unit
# (or a firmware update) reports different ranges.
FOCUS_MIN, FOCUS_MAX = _query_ctrl_range("focus_absolute")
WB_TEMP_MIN, WB_TEMP_MAX = _query_ctrl_range("white_balance_temperature")
EXPOSURE_MIN, EXPOSURE_MAX = _query_ctrl_range("exposure_time_absolute")
GAIN_MIN, GAIN_MAX = _query_ctrl_range("gain")
BRIGHTNESS_MIN, BRIGHTNESS_MAX = _query_ctrl_range("brightness")
CONTRAST_MIN, CONTRAST_MAX = _query_ctrl_range("contrast")
SATURATION_MIN, SATURATION_MAX = _query_ctrl_range("saturation")

# Controls settable via the generic /ctrl endpoint -- a whitelist so the
# query string can't poke arbitrary v4l2 controls.
ALLOWED_CTRLS = {
    "focus_absolute",
    "white_balance_automatic",
    "white_balance_temperature",
    "auto_exposure",
    "exposure_time_absolute",
    "gain",
    "brightness",
    "contrast",
    "saturation",
}

PAGE_TEMPLATE = """\
<html>
<head><title>Camera 2 (Arducam USB3)</title></head>
<body style="font-family: sans-serif; margin: 0; padding: 1em;">
<h1 style="margin-top: 0;">Camera 2 (Arducam USB3 48MP)</h1>
<div style="display: flex; gap: 1.5em; align-items: flex-start; flex-wrap: wrap;">

<div id="controls" style="flex: 0 0 300px; text-align: left;">

  <div>
    <label>Zoom (software crop only -- always softer than 1.0x):
      <input type="range" id="zoom" min="1.0" max="%(zoom_max)s" step="0.1" value="1.0" style="width: 100%%;">
    </label>
    <span id="zoomval">1.0x</span>
  </div>
  <p style="color: #666; font-size: 0.85em; margin: 0.5em 0 0;">
    This camera has no hardware zoom -- every level above 1.0x crops the
    fixed 1280x720 capture and scales it back up, so it's always a
    trade-off, never a quality gain. For more real detail on a subject,
    move the camera physically closer instead.
  </p>

  <div style="margin-top: 1.2em;">
    <label><input type="checkbox" id="stabilize" checked> Stabilization</label>
  </div>

  <div style="margin-top: 1.2em; padding: 0.75em; border: 1px solid #ccc;">
    <div><b>Focus</b></div>
    <p style="color: #666; font-size: 0.8em; margin: 0.3em 0;">
      This camera has no continuous autofocus hardware -- just a manual
      lens position (0-%(focus_max)s, no fixed direction/unit). Drag to
      focus by eye, or use the sweep below to find it automatically.
    </p>
    <label>Position:
      <input type="range" id="focusSlider" min="%(focus_min)s" max="%(focus_max)s" step="1" value="%(focus_default)s" style="width: 100%%;">
    </label>
    <span id="focusVal">%(focus_default)s</span>
    <div style="margin-top: 0.5em;">
      <button id="autofocus" type="button">Find best focus (~10-15s)</button>
    </div>
    <div id="autofocusStatus" style="font-family: monospace; font-size: 0.8em; margin-top: 0.4em;"></div>
  </div>

  <div style="margin-top: 1.2em; padding: 0.75em; border: 1px solid #ccc;">
    <div><b>Image</b></div>
    <p style="color: #666; font-size: 0.8em; margin: 0.3em 0;">
      Resolution and frame rate are hardware-capped on this board (USB2
      only, single UVC mode) -- these are the real levers left for best
      image quality at that fixed size.
    </p>

    <div style="margin-top: 0.5em;">
      <label><input type="checkbox" id="wbAuto" checked> Auto white balance</label>
    </div>
    <div id="wbManual" style="display: none; margin-top: 0.3em;">
      <label>Temperature (K):
        <input type="range" id="wbTemp" min="%(wb_temp_min)s" max="%(wb_temp_max)s" step="50" value="4500" style="width: 100%%;">
      </label>
      <span id="wbTempVal">4500</span>
    </div>

    <div style="margin-top: 0.6em;">
      <label><input type="checkbox" id="expAuto" checked> Auto exposure</label>
    </div>
    <div id="expManual" style="display: none; margin-top: 0.3em;">
      <label>Exposure time:
        <input type="range" id="expTime" min="%(exp_min)s" max="%(exp_max)s" step="1" value="%(exp_min)s" style="width: 100%%;">
      </label>
      <span id="expTimeVal">%(exp_min)s</span>
    </div>
    <div style="margin-top: 0.3em;">
      <label>Gain:
        <input type="range" id="gain" min="%(gain_min)s" max="%(gain_max)s" step="1" value="%(gain_min)s" style="width: 100%%;">
      </label>
      <span id="gainVal">%(gain_min)s</span>
    </div>
    <p style="color: #666; font-size: 0.8em; margin: 0.4em 0 0;">
      Exposure time and gain only take effect in manual exposure mode --
      the camera's own auto-exposure otherwise controls both.
    </p>

    <div style="margin-top: 0.6em;">
      <label>Brightness:
        <input type="range" id="brightness" min="%(brightness_min)s" max="%(brightness_max)s" step="1" value="0" style="width: 100%%;">
      </label>
    </div>
    <div style="margin-top: 0.3em;">
      <label>Contrast:
        <input type="range" id="contrast" min="%(contrast_min)s" max="%(contrast_max)s" step="1" value="10" style="width: 100%%;">
      </label>
    </div>
    <div style="margin-top: 0.3em;">
      <label>Saturation:
        <input type="range" id="saturation" min="%(saturation_min)s" max="%(saturation_max)s" step="1" value="10" style="width: 100%%;">
      </label>
    </div>
  </div>

</div>

<div id="videowrap" style="flex: 1 1 auto; min-width: 0;">
  <img src="stream.mjpg" width="%(out_w)d" height="%(out_h)d" style="max-width: 100%%; height: auto; display: block;" />
</div>

</div>

<script>
function bindSlider(sliderId, valId, cb) {
  const slider = document.getElementById(sliderId);
  const label = valId ? document.getElementById(valId) : null;
  let pending = false, queued = null;
  function send(v) {
    if (pending) { queued = v; return; }
    pending = true;
    cb(v).finally(() => {
      pending = false;
      if (queued !== null) { const q = queued; queued = null; send(q); }
    });
  }
  slider.addEventListener('input', () => {
    if (label) label.textContent = slider.value;
    send(slider.value);
  });
}
function ctrlFetch(name, value) {
  return fetch('/ctrl?name=' + name + '&value=' + value);
}

bindSlider('zoom', 'zoomval', (v) => fetch('/zoom?level=' + v).then(() => {
  document.getElementById('zoomval').textContent = parseFloat(v).toFixed(1) + 'x';
}));
document.getElementById('stabilize').addEventListener('change', (e) => {
  fetch('/stabilize?on=' + (e.target.checked ? '1' : '0'));
});

bindSlider('focusSlider', 'focusVal', (v) => ctrlFetch('focus_absolute', v));

document.getElementById('autofocus').addEventListener('click', (e) => {
  e.target.disabled = true;
  document.getElementById('autofocusStatus').textContent = 'Sweeping... (~10-15s)';
  fetch('/autofocus_sweep').then(r => r.text()).then(t => {
    document.getElementById('autofocusStatus').textContent = t;
    const m = t.match(/position (\\d+)/);
    if (m) {
      document.getElementById('focusSlider').value = m[1];
      document.getElementById('focusVal').textContent = m[1];
    }
    e.target.disabled = false;
  }).catch(() => { e.target.disabled = false; });
});

document.getElementById('wbAuto').addEventListener('change', (e) => {
  document.getElementById('wbManual').style.display = e.target.checked ? 'none' : 'block';
  ctrlFetch('white_balance_automatic', e.target.checked ? 1 : 0);
});
bindSlider('wbTemp', 'wbTempVal', (v) => ctrlFetch('white_balance_temperature', v));

document.getElementById('expAuto').addEventListener('change', (e) => {
  document.getElementById('expManual').style.display = e.target.checked ? 'none' : 'block';
  ctrlFetch('auto_exposure', e.target.checked ? 0 : 1);
});
bindSlider('expTime', 'expTimeVal', (v) => ctrlFetch('exposure_time_absolute', v));
bindSlider('gain', 'gainVal', (v) => ctrlFetch('gain', v));

bindSlider('brightness', null, (v) => ctrlFetch('brightness', v));
bindSlider('contrast', null, (v) => ctrlFetch('contrast', v));
bindSlider('saturation', null, (v) => ctrlFetch('saturation', v));
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
                level = float(qs.get("level", ["1.0"])[0])
            except ValueError:
                level = 1.0
            level = max(ZOOM_MIN, min(level, ZOOM_MAX))
            with state_lock:
                state["zoom"] = level
            self._send_text(f"ok {level:.2f}")
        elif self.path.startswith("/stabilize"):
            qs = parse_qs(urlparse(self.path).query)
            on = qs.get("on", ["1"])[0] != "0"
            with state_lock:
                state["stabilize"] = on
            self._send_text(f"ok {on}")
        elif self.path.startswith("/ctrl"):
            qs = parse_qs(urlparse(self.path).query)
            name = qs.get("name", [""])[0]
            if name not in ALLOWED_CTRLS:
                self.send_error(400, "unknown control")
                return
            try:
                value = int(float(qs.get("value", ["0"])[0]))
            except ValueError:
                self.send_error(400, "bad value")
                return
            try:
                set_ctrl(name, value)
            except subprocess.CalledProcessError as e:
                self._send_text(f"error: {e}")
                return
            self._send_text("ok")
        elif self.path == "/autofocus_sweep":
            result = _autofocus_sweep()
            self._send_text(result)
        elif self.path == "/stream.mjpg":
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


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


state = {"stabilize": True, "zoom": 1.0}
state_lock = threading.Lock()

cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_SIZE[1])
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency; may be ignored by the V4L2 backend
if not cap.isOpened():
    raise RuntimeError(f"Failed to open {DEVICE}")

BASE_W, BASE_H = CAPTURE_SIZE
CENTER_X = BASE_W / 2
CENTER_Y = BASE_H / 2

# Latest raw captured frame, kept for _autofocus_sweep to read without a
# second concurrent cap.read() (OpenCV VideoCapture isn't safe to read
# from two threads at once).
latest_frame = {"frame": None}
latest_lock = threading.Lock()

PAGE = PAGE_TEMPLATE % {
    "out_w": OUTPUT_SIZE[0],
    "out_h": OUTPUT_SIZE[1],
    "zoom_max": ZOOM_MAX,
    "focus_min": FOCUS_MIN,
    "focus_max": FOCUS_MAX,
    "focus_default": (FOCUS_MIN + FOCUS_MAX) // 2,
    "wb_temp_min": WB_TEMP_MIN,
    "wb_temp_max": WB_TEMP_MAX,
    "exp_min": EXPOSURE_MIN,
    "exp_max": EXPOSURE_MAX,
    "gain_min": GAIN_MIN,
    "gain_max": GAIN_MAX,
    "brightness_min": BRIGHTNESS_MIN,
    "brightness_max": BRIGHTNESS_MAX,
    "contrast_min": CONTRAST_MIN,
    "contrast_max": CONTRAST_MAX,
    "saturation_min": SATURATION_MIN,
    "saturation_max": SATURATION_MAX,
}

output = StreamingOutput()

# Mirrors mjpeg_server.py: cv2.imencode releases the GIL, so independent
# JPEG frames genuinely encode in parallel. Less critical at this camera's
# 10fps than on Camera 1's 30fps target, but kept for the same reason and
# for architectural consistency between the two servers.
encoder_pool = ThreadPoolExecutor(max_workers=2)


def _encode_and_publish(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        output.write(jpg.tobytes())


# Motion estimation runs on its own thread rather than inline in
# capture_loop, mirroring Camera 1's stabilization_worker -- it only needs
# a "recent enough" correction, not the current frame's.
motion_queue = queue.Queue(maxsize=1)
correction_state = {"crop_x": MARGIN_X, "crop_y": MARGIN_Y}
correction_lock = threading.Lock()


def stabilization_worker():
    small_size = (CAPTURE_SIZE[0] // MOTION_DOWNSCALE, CAPTURE_SIZE[1] // MOTION_DOWNSCALE)
    hann = cv2.createHanningWindow(small_size, cv2.CV_32F)
    prev_gray = None
    raw_traj = np.zeros(2)
    smooth_traj = np.zeros(2)

    while True:
        frame = motion_queue.get()
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            try:
                (dx, dy), _response = cv2.phaseCorrelate(prev_gray * hann, gray * hann)
                raw_traj += (dx * MOTION_DOWNSCALE, dy * MOTION_DOWNSCALE)
            except cv2.error:
                pass
        prev_gray = gray

        smooth_traj = smooth_traj * (1 - SMOOTHING_ALPHA) + raw_traj * SMOOTHING_ALPHA
        correction = raw_traj - smooth_traj

        with correction_lock:
            correction_state["crop_x"] = int(np.clip(MARGIN_X + correction[0], 0, CAPTURE_SIZE[0] - OUTPUT_SIZE[0]))
            correction_state["crop_y"] = int(np.clip(MARGIN_Y + correction[1], 0, CAPTURE_SIZE[1] - OUTPUT_SIZE[1]))


def capture_loop():
    while True:
        ok, frame = cap.read()  # BGR888, CAPTURE_SIZE (OpenCV converts YUYV -> BGR)
        if not ok:
            continue

        with latest_lock:
            latest_frame["frame"] = frame

        with state_lock:
            stabilize_on = state["stabilize"]
            zoom = state["zoom"]

        zoom_w = min(BASE_W, max(OUTPUT_SIZE[0], int(BASE_W / zoom)))
        zoom_h = min(BASE_H, max(OUTPUT_SIZE[1], int(BASE_H / zoom)))
        base_x = (BASE_W - zoom_w) // 2
        base_y = (BASE_H - zoom_h) // 2
        crop_x, crop_y = base_x, base_y

        if stabilize_on:
            try:
                motion_queue.put_nowait(frame)
            except queue.Full:
                pass
            if zoom_w == OUTPUT_SIZE[0] and zoom_h == OUTPUT_SIZE[1]:
                # Only apply the stabilization worker's correction when not
                # also zoomed -- the correction is computed in OUTPUT_SIZE
                # crop-window terms, which stops being valid once zoom_w/h
                # differ from that (same simplification the pre-zoom-split
                # version of this script used).
                with correction_lock:
                    crop_x = correction_state["crop_x"]
                    crop_y = correction_state["crop_y"]

        cropped = frame[crop_y:crop_y + zoom_h, crop_x:crop_x + zoom_w]
        if (zoom_w, zoom_h) != OUTPUT_SIZE:
            interp = cv2.INTER_AREA if zoom_w > OUTPUT_SIZE[0] else cv2.INTER_CUBIC
            cropped = cv2.resize(cropped, OUTPUT_SIZE, interpolation=interp)

        encoder_pool.submit(_encode_and_publish, cropped)


def _sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _autofocus_sweep():
    """Coarse-then-fine search over focus_absolute for the sharpest frame.
    There's no hardware autofocus on this camera to trigger -- this is a
    from-scratch software substitute using the same "hill climb on a
    sharpness metric" idea real contrast-detect AF uses.
    """

    def score_at(pos):
        set_ctrl("focus_absolute", pos)
        time.sleep(0.4)  # let the lens motor settle + a fresh frame arrive
        with latest_lock:
            frame = latest_frame["frame"]
        return _sharpness(frame) if frame is not None else -1.0

    span = FOCUS_MAX - FOCUS_MIN
    coarse_step = max(1, span // 12)
    coarse_positions = list(range(FOCUS_MIN, FOCUS_MAX + 1, coarse_step))
    if coarse_positions[-1] != FOCUS_MAX:
        coarse_positions.append(FOCUS_MAX)
    coarse_scores = [(p, score_at(p)) for p in coarse_positions]
    best_p, _ = max(coarse_scores, key=lambda t: t[1])

    fine_step = max(1, coarse_step // 5)
    lo, hi = max(FOCUS_MIN, best_p - coarse_step), min(FOCUS_MAX, best_p + coarse_step)
    fine_scores = [(p, score_at(p)) for p in range(lo, hi + 1, fine_step)]
    best_p, best_s = max(fine_scores, key=lambda t: t[1])

    set_ctrl("focus_absolute", best_p)
    return f"Best focus: position {best_p} (sharpness {best_s:.1f})"


capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()
stabilization_thread = threading.Thread(target=stabilization_worker, daemon=True)
stabilization_thread.start()

try:
    address = ("0.0.0.0", 8080)
    http_server = StreamingServer(address, StreamingHandler)
    print("Serving live view with zoom + stabilization + focus/image controls on :8080/index.html")
    http_server.serve_forever()
finally:
    cap.release()
