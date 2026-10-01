#!/usr/bin/env python3
"""Live MJPEG stream from Camera 1 with browser-controlled digital zoom and
digital stabilization (translation-only electronic image stabilization).
"""
import logging
import queue
import socketserver
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http import server
from threading import Condition
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
from picamera2 import Picamera2

# CAPTURE_SIZE is the sensor's 2312x1736 mode -- the fastest mode that still
# uses the sensor's full FOV (a clean 4x bin of the full 9248x6944 array),
# needed to hit 30fps.
#
# CAPTURE_SIZE's aspect (4:3, matching the sensor) doesn't match OUTPUT_SIZE's
# (16:9), so FOV_SIZE below is the largest 16:9 region that fits inside
# CAPTURE_SIZE -- cropping off the top/bottom, same as any 4:3 sensor
# producing widescreen output.
#
# CROP_SIZE then shrinks FOV_SIZE by STAB_HEADROOM as panning headroom for
# stabilization, and that crop is resized down to OUTPUT_SIZE. Sizing it as
# a fraction of FOV_SIZE (rather than hardcoding OUTPUT_SIZE pixels, as an
# earlier version of this did) keeps zoom=1.0 showing the sensor mode's true
# full FOV regardless of CAPTURE_SIZE's resolution -- otherwise a bigger
# capture size silently steals FOV from the wide end of the zoom range
# instead of only adding detail at the zoomed-in end.
CAPTURE_SIZE = (2312, 1736)
OUTPUT_SIZE = (1920, 1080)
STAB_HEADROOM = 1.2  # linear; 1.2 matches this app's original margin ratio

if CAPTURE_SIZE[0] * OUTPUT_SIZE[1] >= CAPTURE_SIZE[1] * OUTPUT_SIZE[0]:
    FOV_SIZE = (int(CAPTURE_SIZE[1] * OUTPUT_SIZE[0] / OUTPUT_SIZE[1]), CAPTURE_SIZE[1])
else:
    FOV_SIZE = (CAPTURE_SIZE[0], int(CAPTURE_SIZE[0] * OUTPUT_SIZE[1] / OUTPUT_SIZE[0]))

CROP_SIZE = (int(FOV_SIZE[0] / STAB_HEADROOM), int(FOV_SIZE[1] / STAB_HEADROOM))
MARGIN_X = (CAPTURE_SIZE[0] - CROP_SIZE[0]) // 2
MARGIN_Y = (CAPTURE_SIZE[1] - CROP_SIZE[1]) // 2

SW_ZOOM_MIN = 1.0
SW_ZOOM_MAX = 4.0  # independent of hardware zoom -- see CRISP_ZOOM below for
                    # the hardware slider's range, computed after configure()

MOTION_DOWNSCALE = 8        # motion estimated on a 2312/8 x 1736/8 image
STAB_UPDATE_EVERY = 3       # only feed every Nth frame to the stabilization
                            # worker -- see capture_loop for why
SMOOTHING_ALPHA = 0.03      # lower = stronger stabilization, slower to follow intentional pans
JPEG_QUALITY = 75  # 85 measured ~31% slower to encode for little visible
                    # gain on a live view; see capture_loop/_encode_and_publish

# Filled in further down, once CRISP_ZOOM and the LensPosition range (both
# only known after the camera is configured/started) are available.
PAGE_TEMPLATE = """\
<html>
<head><title>Camera 1</title></head>
<body style="font-family: sans-serif; margin: 0; padding: 1em;">
<h1 style="margin-top: 0;">Camera 1</h1>
<div style="display: flex; gap: 1.5em; align-items: flex-start; flex-wrap: wrap;">

<div id="controls" style="flex: 0 0 300px; text-align: left;">

  <div>
    <label>Camera zoom (hardware, real detail up to %(hw_max)sx):
      <input type="range" id="hwzoom" min="1.0" max="%(hw_max)s" step="0.1" value="1.0" style="width: 100%%;">
    </label>
    <span id="hwzoomval">1.0x</span>
  </div>
  <div style="margin-top: 0.5em;">
    <label>Extra zoom (software upscale, softens the image):
      <input type="range" id="swzoom" min="1.0" max="%(sw_max)s" step="0.1" value="1.0" style="width: 100%%;">
    </label>
    <span id="swzoomval">1.0x</span>
  </div>
  <p style="color: #666; font-size: 0.85em; margin: 0.5em 0 0;">
    Total zoom = camera x extra. Camera zoom stays crisp (real sensor
    detail) up to %(hw_max)sx. Extra zoom crops further into that and
    upscales in software -- always softer, use it to see exactly how much
    the upscale costs you versus the hardware zoom alone.
  </p>

  <div style="margin-top: 1.2em;">
    <label><input type="checkbox" id="stabilize"> Stabilization</label>
  </div>
  <p style="color: #666; font-size: 0.85em; margin: 0.5em 0 0;">
    Off: ~29fps. On: ~18fps (this board's CPU is the bottleneck, not the
    camera). Off by default to meet the 30fps target.
  </p>

  <div style="margin-top: 1.2em; padding: 0.75em; border: 1px solid #ccc;">
    <div><b>Focus</b></div>
    <div id="afinfo" style="font-family: monospace; font-size: 0.85em; margin-top: 0.3em;">loading...</div>

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
        <input type="range" id="focusSlider" min="%(lens_min)s" max="%(lens_max)s" step="0.01" value="%(lens_min)s" style="width: 100%%;">
      </label>
      <span id="focusVal">infinity</span>
      <div style="margin-top: 0.4em;">
        <button id="focusFarBig" type="button">&laquo; far</button>
        <button id="focusFarSmall" type="button">&lsaquo;</button>
        <button id="focusNearSmall" type="button">&rsaquo;</button>
        <button id="focusNearBig" type="button">near &raquo;</button>
      </div>
      <p style="color: #666; font-size: 0.8em; margin: 0.4em 0 0;">
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

<div id="videowrap" style="flex: 1 1 auto; min-width: 0;">
  <img src="stream.mjpg" width="%(out_w)d" height="%(out_h)d" style="max-width: 100%%; height: auto; display: block;" />
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
        elif self.path.startswith("/afrange"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                af_range = int(qs.get("range", ["0"])[0])
            except ValueError:
                af_range = 0
            picam2.set_controls({"AfRange": max(0, min(af_range, 2))})
            self._send_text("ok")
        elif self.path == "/refocus":
            threading.Thread(target=_do_refocus, daemon=True).start()
            self._send_text("refocusing")
        elif self.path.startswith("/afmode"):
            qs = parse_qs(urlparse(self.path).query)
            mode = qs.get("mode", ["continuous"])[0]
            picam2.set_controls({"AfMode": 0 if mode == "manual" else 2})
            self._send_text("ok")
        elif self.path.startswith("/focus"):
            qs = parse_qs(urlparse(self.path).query)
            try:
                position = float(qs.get("position", [str(LENS_MIN)])[0])
            except ValueError:
                position = LENS_MIN
            applied = set_manual_focus(position)
            self._send_text(f"ok {applied:.2f}")
        elif self.path == "/afinfo":
            self._send_text(_af_info_text())
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


state = {"stabilize": False, "hw_zoom": 1.0, "sw_zoom": 1.0}
state_lock = threading.Lock()

picam2 = Picamera2()
# 2312x1736 RGB frames are ~12MB each (vs ~48MB at 4624x3472), so the
# default buffer_count comfortably fits this board's ~900MB RAM without the
# buffer_count reduction earlier, higher-resolution attempts needed.
video_config = picam2.create_video_configuration(main={"size": CAPTURE_SIZE, "format": "RGB888"})
picam2.configure(video_config)

# Baseline (zoom=1.0) is the sensor mode's full field of view, not the raw
# pixel array -- the driver already crops/bins to this for the chosen mode.
_min_crop, _max_crop, _default_crop = picam2.camera_controls["ScalerCrop"]
BASE_X, BASE_Y, BASE_W, BASE_H = _max_crop
CENTER_X = BASE_X + BASE_W / 2
CENTER_Y = BASE_Y + BASE_H / 2

# Above this zoom level, the hardware crop would be smaller than
# CAPTURE_SIZE, so the ISP would have to upscale to fill it -- there's no
# more real detail available from the sensor past this point. This is the
# max of the "camera zoom" slider; the "extra zoom" slider (set_sw_zoom)
# is independent and always applied in software regardless of hw_zoom, so
# the two can be experimented with separately.
CRISP_ZOOM = min(BASE_W / CAPTURE_SIZE[0], BASE_H / CAPTURE_SIZE[1])

# LensPosition is in dioptres (1/metres); 0 = infinity, higher = closer.
LENS_MIN, LENS_MAX, _lens_default = picam2.camera_controls.get("LensPosition", (0.0, 15.0, 0.0))


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


def set_manual_focus(position):
    position = max(LENS_MIN, min(position, LENS_MAX))
    # AfMode must be Manual for LensPosition to actually move the lens --
    # in Continuous/Auto mode the algorithm just overrides it.
    picam2.set_controls({"AfMode": 0, "LensPosition": position})
    return position


PAGE = PAGE_TEMPLATE % {
    "out_w": OUTPUT_SIZE[0],
    "out_h": OUTPUT_SIZE[1],
    "hw_max": round(CRISP_ZOOM, 1),
    "sw_max": SW_ZOOM_MAX,
    "lens_min": LENS_MIN,
    "lens_max": LENS_MAX,
}


output = StreamingOutput()

# JPEG encoding one 1920x1080 frame takes ~60ms single-threaded -- a hard
# ~16fps ceiling on its own, well under the 30fps target. cv2.imencode
# releases the GIL, so independent frames (there's no inter-frame JPEG
# dependency, unlike video codecs) genuinely encode in parallel across
# cores. This board has 4 cores total; capture_loop and stabilization_worker
# (below) are each a separate CPU-bound thread, so 2 encoder workers (not
# more) leaves them room to run without oversubscribing the 4 cores --
# measured worse throughput with 3 encoder workers once the stabilization
# thread was added (5 competing CPU-bound threads on 4 cores).
encoder_pool = ThreadPoolExecutor(max_workers=2)


def _encode_and_publish(img):
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        output.write(jpg.tobytes())


# Motion estimation (resize + grayscale + phaseCorrelate, ~12-13ms) runs on
# its own thread instead of inline in capture_loop. It doesn't need to run
# once per output frame -- stabilization only needs a "recent enough"
# correction, not the current frame's. Decoupling it turns two sequential
# per-frame costs (~19ms capture + ~12ms motion-estimate) that were both
# blocking every output frame into two things that run concurrently, which
# is what actually got this to 30fps -- the encoder threadpool and
# INTER_LINEAR swaps alone weren't enough since capture+motion-estimate
# was still >33ms combined in the same thread. A maxsize=1 queue means the
# worker always works on the freshest frame and silently drops others if
# it (rarely) falls behind, rather than building a backlog.
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
        # picamera2's "RGB888" format actually hands back BGR-ordered data
        # (a longstanding naming quirk) -- exploited here to skip a manual
        # channel-reversal copy of the full CAPTURE_SIZE frame every loop.
        frame = picam2.capture_array("main")  # CAPTURE_SIZE, already BGR-ordered
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

        # "Extra zoom" (sw_zoom, independent of the hardware zoom slider) is
        # applied here in software: crop further into the stabilized window
        # and upscale. sw_zoom is always >= 1.0 (see set_sw_zoom), so this
        # is a no-op crop at its default.
        sub_w = max(1, int(CROP_SIZE[0] / extra_zoom))
        sub_h = max(1, int(CROP_SIZE[1] / extra_zoom))
        sub_x = crop_x + (CROP_SIZE[0] - sub_w) // 2
        sub_y = crop_y + (CROP_SIZE[1] - sub_h) // 2

        cropped = frame[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w]
        # INTER_CUBIC for the upscaling (>CRISP_ZOOM) case: noticeably
        # sharper than INTER_LINEAR for magnification, without INTER_LANCZOS4's
        # cost (measured ~53ms/frame -- would blow the frame budget outright).
        # INTER_LINEAR for the non-upscaling case: ~2.3x faster than
        # INTER_AREA (measured 12ms vs 28ms) with negligible quality
        # difference here since this resize is always close to 1:1 (the
        # actual downsampling from CAPTURE_SIZE already happened on-sensor
        # via ScalerCrop).
        interp = cv2.INTER_CUBIC if extra_zoom > 1.0 else cv2.INTER_LINEAR
        cropped = cv2.resize(cropped, OUTPUT_SIZE, interpolation=interp)

        encoder_pool.submit(_encode_and_publish, cropped)


_AF_STATE_NAMES = {0: "Idle", 1: "Scanning", 2: "Focused", 3: "Failed"}


def _af_info_text():
    md = picam2.capture_metadata()
    lens_pos = md.get("LensPosition")
    af_state = md.get("AfState")
    fom = md.get("FocusFoM")
    state_name = _AF_STATE_NAMES.get(af_state, str(af_state))
    # LensPosition is in dioptres (1/metres) for this driver -- e.g. 6.0
    # means focused at about 1/6 m =~ 17cm. If AF reports "Focused" but the
    # image still looks soft, it's very likely locked onto something close
    # in frame instead of your actual subject -- check this distance
    # against how far the real subject is, not just whether AF says OK.
    if lens_pos is None:
        dist = "unknown"
    elif lens_pos == 0:
        dist = "infinity"
    else:
        dist = f"~{100.0 / lens_pos:.0f}cm"
    return f"State: {state_name}  LensPosition: {lens_pos}  ~Focus distance: {dist}  FocusFoM: {fom}"


def _do_refocus():
    # Force a fresh AF scan: Continuous mode normally refocuses on its own,
    # but if it's stuck on the wrong subject, switching to Auto + Trigger
    # forces it to restart the search rather than just nudge from where it
    # already is.
    picam2.set_controls({"AfMode": 1})  # Auto
    picam2.set_controls({"AfTrigger": 0})  # Start
    for _ in range(40):  # up to ~4s
        time.sleep(0.1)
        md = picam2.capture_metadata()
        if md.get("AfState") in (2, 3):  # Focused or Failed
            break
    picam2.set_controls({"AfMode": 2})  # back to Continuous


picam2.start()
if "AfMode" in picam2.camera_controls:
    picam2.set_controls({"AfMode": 2})  # 0=Manual, 1=Auto, 2=Continuous
capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()
stabilization_thread = threading.Thread(target=stabilization_worker, daemon=True)
stabilization_thread.start()

try:
    address = ("0.0.0.0", 8080)
    http_server = StreamingServer(address, StreamingHandler)
    print("Serving live view with zoom + stabilization on :8080/index.html")
    http_server.serve_forever()
finally:
    picam2.stop()
