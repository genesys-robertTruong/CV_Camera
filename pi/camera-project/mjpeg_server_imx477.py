#!/usr/bin/env python3
"""Live MJPEG stream from the Sony IMX477 (Raspberry Pi HQ Camera sensor,
~12MP) plugged into CAM1, alongside the 64MP module on CAM0 (see
mjpeg_server.py) -- this is the "optical zoom" camera: the HQ Camera board
itself has no zoom or autofocus actuator (confirmed via picamera2:
LensPosition and AfMode controls are both None), so its zoom AND focus are
both purely mechanical -- the physical zoom/focus rings on whatever C/CS-mount
lens is attached. There is no software focus panel here because there is
genuinely nothing to control electronically; the "Camera zoom" slider below
is real (ScalerCrop hardware crop, same mechanism as the other servers),
but it composes with the lens's own manual zoom rather than an AfMode/
LensPosition-based system.

Same architecture as mjpeg_server.py otherwise (parallel JPEG-encoder pool,
decoupled stabilization worker, hardware+software zoom split) -- see that
file's comments for the reasoning. camera_num=0: confirmed via
Picamera2.global_camera_info() with both CSI cameras present -- imx477 is
index 0, arducam_64mp is index 1. Re-verify if the camera set changes.
"""
import queue
import socketserver
import threading
from concurrent.futures import ThreadPoolExecutor
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs
import logging

import cv2
import numpy as np
from picamera2 import Picamera2

# The sensor's 53.77fps mode: a clean 2x2 bin of the full 4056x3040 sensor
# (crop (0,0)/4056x3040 -- confirmed via rpicam-hello's mode list), same
# "full-sensor-width binned" choice made for the other cameras' main modes.
CAPTURE_SIZE = (2028, 1520)
OUTPUT_SIZE = (1920, 1080)
STAB_HEADROOM = 1.2

if CAPTURE_SIZE[0] * OUTPUT_SIZE[1] >= CAPTURE_SIZE[1] * OUTPUT_SIZE[0]:
    FOV_SIZE = (int(CAPTURE_SIZE[1] * OUTPUT_SIZE[0] / OUTPUT_SIZE[1]), CAPTURE_SIZE[1])
else:
    FOV_SIZE = (CAPTURE_SIZE[0], int(CAPTURE_SIZE[0] * OUTPUT_SIZE[1] / OUTPUT_SIZE[0]))

CROP_SIZE = (int(FOV_SIZE[0] / STAB_HEADROOM), int(FOV_SIZE[1] / STAB_HEADROOM))
MARGIN_X = (CAPTURE_SIZE[0] - CROP_SIZE[0]) // 2
MARGIN_Y = (CAPTURE_SIZE[1] - CROP_SIZE[1]) // 2

SW_ZOOM_MIN = 1.0
SW_ZOOM_MAX = 4.0

MOTION_DOWNSCALE = 8
STAB_UPDATE_EVERY = 3
SMOOTHING_ALPHA = 0.03
JPEG_QUALITY = 75

PAGE_TEMPLATE = """\
<html>
<head>
<title>Camera 3 (imx477 HQ)</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%%; width: 100%%; overflow: hidden;
    background: #000; color: #ddd;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  #layout { display: flex; height: 100vh; width: 100vw; }
  #videowrap {
    flex: 1 1 auto; min-width: 0; min-height: 0;
    display: flex; align-items: center; justify-content: center;
    background: #000;
  }
  #videowrap img { max-width: 100%%; max-height: 100%%; width: auto; height: auto; display: block; }
  #controls {
    flex: 0 0 300px; max-width: 42vw; overflow-y: auto;
    background: #111; padding: 1em; border-left: 1px solid #292929;
  }
  h1 { font-size: 1.1em; margin: 0 0 0.75em; color: #fff; }
  label { color: #ccc; }
  input[type=range] { width: 100%%; accent-color: #4a9eff; }
  .panel { margin-top: 1.2em; padding: 0.75em; border: 1px solid #292929; border-radius: 6px; }
  .hint { color: #888; font-size: 0.8em; margin: 0.4em 0 0; }
</style>
</head>
<body>
<div id="layout">

<div id="videowrap">
  <img src="stream.mjpg" width="%(out_w)d" height="%(out_h)d" />
</div>

<div id="controls">
  <h1>Camera 3 (imx477 HQ, optical zoom lens)</h1>
  <p class="hint">
    This is the Raspberry Pi HQ Camera sensor -- it has no built-in zoom or
    autofocus. Both are entirely manual: physical zoom and focus rings on
    the attached lens. There's no focus panel here because there's
    genuinely nothing to control in software for it -- adjust the lens
    itself while watching this live view.
  </p>

  <div style="margin-top: 1em;">
    <label>Camera zoom (hardware crop, real detail up to %(hw_max)sx):
      <input type="range" id="hwzoom" min="1.0" max="%(hw_max)s" step="0.1" value="1.0">
    </label>
    <span id="hwzoomval">1.0x</span>
  </div>
  <div style="margin-top: 0.5em;">
    <label>Extra zoom (software upscale, softens the image):
      <input type="range" id="swzoom" min="1.0" max="%(sw_max)s" step="0.1" value="1.0">
    </label>
    <span id="swzoomval">1.0x</span>
  </div>
  <p class="hint">
    This "camera zoom" is the same digital-sensor-crop mechanism as the
    other cameras on this Pi -- it composes with the lens's own optical
    zoom rather than replacing it. Turn the lens ring for real optical
    zoom; use these sliders on top of that if you want to crop in further.
  </p>

  <div style="margin-top: 1.2em;">
    <label><input type="checkbox" id="stabilize" checked> Stabilization</label>
  </div>

</div>

</div>

<script>
function bindZoomSlider(sliderId, valId, endpoint) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(valId);
  let pending = false, queued = null;
  function send(level) {
    if (pending) { queued = level; return; }
    pending = true;
    fetch(endpoint + '?level=' + level).finally(() => {
      pending = false;
      if (queued !== null) { const q = queued; queued = null; send(q); }
    });
  }
  slider.addEventListener('input', () => {
    label.textContent = parseFloat(slider.value).toFixed(1) + 'x';
    send(slider.value);
  });
}
bindZoomSlider('hwzoom', 'hwzoomval', '/hwzoom');
bindZoomSlider('swzoom', 'swzoomval', '/swzoom');

document.getElementById('stabilize').addEventListener('change', (e) => {
  fetch('/stabilize?on=' + (e.target.checked ? '1' : '0'));
});
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
        elif self.path.startswith("/hwzoom"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                level = float(qs.get("level", ["1.0"])[0])
            except ValueError:
                level = 1.0
            applied = set_hw_zoom(level)
            self._send_text(f"ok {applied:.2f}")
        elif self.path.startswith("/swzoom"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                level = float(qs.get("level", ["1.0"])[0])
            except ValueError:
                level = 1.0
            applied = set_sw_zoom(level)
            self._send_text(f"ok {applied:.2f}")
        elif self.path.startswith("/stabilize"):
            qs = parse_qs(urlparse(self.path).query)
            on = qs.get("on", ["1"])[0] != "0"
            with state_lock:
                state["stabilize"] = on
            self._send_text(f"ok {on}")
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


state = {"stabilize": True, "hw_zoom": 1.0, "sw_zoom": 1.0}
state_lock = threading.Lock()

# camera_num=0: confirmed via Picamera2.global_camera_info() with both CSI
# cameras present -- imx477 is index 0, arducam_64mp is index 1.
picam2 = Picamera2(camera_num=0)
video_config = picam2.create_video_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
picam2.configure(video_config)

_min_crop, _max_crop, _default_crop = picam2.camera_controls["ScalerCrop"]
BASE_X, BASE_Y, BASE_W, BASE_H = _max_crop
CENTER_X = BASE_X + BASE_W / 2
CENTER_Y = BASE_Y + BASE_H / 2

CRISP_ZOOM = min(BASE_W / CAPTURE_SIZE[0], BASE_H / CAPTURE_SIZE[1])


def set_hw_zoom(level):
    level = max(1.0, min(level, CRISP_ZOOM))
    with state_lock:
        state["hw_zoom"] = level
    w = int(BASE_W / level)
    h = int(BASE_H / level)
    x = int(CENTER_X - w / 2)
    y = int(CENTER_Y - h / 2)
    picam2.set_controls({"ScalerCrop": (x, y, w, h)})
    return level


def set_sw_zoom(level):
    level = max(SW_ZOOM_MIN, min(level, SW_ZOOM_MAX))
    with state_lock:
        state["sw_zoom"] = level
    return level


PAGE = PAGE_TEMPLATE % {
    "out_w": OUTPUT_SIZE[0],
    "out_h": OUTPUT_SIZE[1],
    "hw_max": round(CRISP_ZOOM, 1),
    "sw_max": SW_ZOOM_MAX,
}


output = StreamingOutput()
encoder_pool = ThreadPoolExecutor(max_workers=2)


def _encode_and_publish(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        output.write(jpg.tobytes())


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
            correction_state["crop_x"] = int(np.clip(MARGIN_X + correction[0], 0, CAPTURE_SIZE[0] - CROP_SIZE[0]))
            correction_state["crop_y"] = int(np.clip(MARGIN_Y + correction[1], 0, CAPTURE_SIZE[1] - CROP_SIZE[1]))


def capture_loop():
    frame_count = 0
    while True:
        frame = picam2.capture_array("main")
        frame_count += 1

        with state_lock:
            stabilize_on = state["stabilize"]
            extra_zoom = state["sw_zoom"]

        if stabilize_on:
            if frame_count % STAB_UPDATE_EVERY == 0:
                try:
                    motion_queue.put_nowait(frame)
                except queue.Full:
                    pass
            with correction_lock:
                crop_x = correction_state["crop_x"]
                crop_y = correction_state["crop_y"]
        else:
            crop_x, crop_y = MARGIN_X, MARGIN_Y

        sub_w = max(1, int(CROP_SIZE[0] / extra_zoom))
        sub_h = max(1, int(CROP_SIZE[1] / extra_zoom))
        sub_x = crop_x + (CROP_SIZE[0] - sub_w) // 2
        sub_y = crop_y + (CROP_SIZE[1] - sub_h) // 2

        cropped = frame[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w]
        interp = cv2.INTER_CUBIC if extra_zoom > 1.0 else cv2.INTER_LINEAR
        cropped = cv2.resize(cropped, OUTPUT_SIZE, interpolation=interp)

        encoder_pool.submit(_encode_and_publish, cropped)


picam2.start()
capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()
stabilization_thread = threading.Thread(target=stabilization_worker, daemon=True)
stabilization_thread.start()

try:
    address = ("0.0.0.0", 8081)
    http_server = StreamingServer(address, StreamingHandler)
    print("Serving live view with zoom + stabilization on :8081/index.html")
    http_server.serve_forever()
finally:
    picam2.stop()
