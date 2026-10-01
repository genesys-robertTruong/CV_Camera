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
START_LENS = 6.65  # dioptres; best focus for the bench scene (see CLAUDE.md)

PAGE = """\
<html>
<head>
<title>64MP -- 1:1 Sensor Crop</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #000; color: #ddd;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  #video { display: block; max-width: 100%; margin: 0 auto; }
  #controls { max-width: 1280px; margin: 0 auto; padding: 0.8em 1em; }
  input[type=range] { width: 100%; accent-color: #4a9eff; }
  button { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 0.35em 0.7em; }
  #info { font-family: ui-monospace, monospace; font-size: 0.85em; color: #9fdca0; margin-top: 0.5em; }
  .hint { color: #888; font-size: 0.8em; }
</style>
</head>
<body>
<img id="video" src="stream.mjpg" />
<div id="controls">
  <div>
    <button id="cont">Continuous AF</button>
    <button id="refocus">Refocus once</button>
  </div>
  <div style="margin-top: 0.6em;">
    <label>Manual focus (dioptres): <span id="fv"></span>
      <input type="range" id="focus" min="LENS_MIN" max="LENS_MAX" step="0.01" value="6.65">
    </label>
  </div>
  <div id="info">loading...</div>
  <p class="hint">MODE_HINT. 1:1 sensor pixels (no binning, no scaling).
  Depth of field is very shallow -- use small slider moves.</p>
</div>
<script>
const f = document.getElementById('focus'), fv = document.getElementById('fv');
let pending = false, queued = null;
function send(pos) {
  if (pending) { queued = pos; return; }
  pending = true;
  fetch('/focus?position=' + pos).finally(() => {
    pending = false;
    if (queued !== null) { const q = queued; queued = null; send(q); }
  });
}
f.addEventListener('input', () => { fv.textContent = f.value; send(f.value); });
document.getElementById('cont').onclick = () => fetch('/afmode?mode=continuous');
document.getElementById('refocus').onclick = () => fetch('/refocus');
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
            f"Crop: {md.get('ScalerCrop')}  sensor {fps:.1f} fps, streamed {stats['stream_fps']:.1f} fps  "
            f"exp {md.get('ExposureTime')}us gain {md.get('AnalogueGain', 0):.1f}")


def refocus():
    # Blocks up to ~4s; called on its own thread so the stream keeps running.
    picam2.set_controls({"AfMode": 1, "AfTrigger": 0})
    for _ in range(40):
        time.sleep(0.1)
        if picam2.capture_metadata().get("AfState") in (2, 3):
            break


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
        frame = picam2.capture_array("main")
        seq += 1
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
                    .replace("MODE_HINT", hint).encode())
            self._send(body, "text/html")
        elif url.path == "/focus":
            try:
                pos = float(qs.get("position", ["0"])[0])
            except ValueError:
                pos = 0.0
            pos = max(lens_min, min(pos, lens_max))
            picam2.set_controls({"AfMode": 0, "LensPosition": pos})
            self._send(f"ok {pos:.2f}".encode())
        elif url.path == "/afmode":
            picam2.set_controls({"AfMode": 0 if qs.get("mode", [""])[0] == "manual" else 2})
            self._send(b"ok")
        elif url.path == "/refocus":
            threading.Thread(target=refocus, daemon=True).start()
            self._send(b"refocusing")
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
    # Start in manual focus: continuous AF on this narrow crop was seen
    # settling well off (8.8 vs a measured best of 6.65 for the bench scene).
    picam2.set_controls({"ScalerCrop": crop, "AfMode": 0, "LensPosition": START_LENS})
    lens_min, lens_max, _ = picam2.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))

    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=stream_fps_loop, daemon=True).start()
    try:
        http_server = StreamingServer(("0.0.0.0", PORT), StreamingHandler)
        print(f"Serving {MODE} sensor-crop stream (crop {crop}, frame limits {FRAME_LIMITS}) on :{PORT}/index.html")
        http_server.serve_forever()
    finally:
        picam2.stop()
        picam2.close()
