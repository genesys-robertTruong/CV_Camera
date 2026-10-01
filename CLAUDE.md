# CV_Camera

Computer-vision camera project running on a Raspberry Pi. Code is developed in
this Windows repo and runs on the Pi; there is no git on the Pi itself.

## Target device

- Host: `william@192.168.159.116` (hostname `cv-camera`)
- Raspberry Pi Compute Module 4 (VC4 ISP, `bcm2835_unicam_legacy`), aarch64,
  Debian 13 (trixie), kernel 6.18.34+rpt-rpi-v8, Python 3.13, OpenCV 4.10, picamera2
- ~900 MB RAM, 512 MB CMA, ~1.7 GB free on the SD card -- keep large captures off it
- `william` has passwordless sudo
- Project dir on the Pi: `~/camera-project` (mirrored here in `pi/camera-project/`)
- Cameras: Arducam 64MP Hawk-Eye (Sony IMX682, Quad-Bayer with on-chip
  remosaic) on CAM0 (`dtoverlay=arducam-64mp,cam0`, 2-lane, 456 MHz link),
  running our patched driver (`driver/`). Code also exists for an IMX477 HQ
  camera (currently removed) and an Arducam B0478 USB 48MP (USB2-limited to
  1280x720 @ 10fps). Check `rpicam-hello --list-cameras` before assuming
  which are attached.

## SSH access

Use the project key in this folder (non-interactive, works from the Bash tool):

```sh
ssh -i id_ed25519_cv_camera -o BatchMode=yes -o IdentitiesOnly=yes william@192.168.159.116 "<cmd>"
scp -i id_ed25519_cv_camera -o BatchMode=yes -o IdentitiesOnly=yes <src> william@192.168.159.116:camera-project/
```

- Never read, print or extract `password.txt`. If password auth is ever
  needed, pass the file to a tool indirectly (e.g. an `SSH_ASKPASS` helper).
- Don't generate new SSH keys; use `id_ed25519_cv_camera`.
- `password.txt` and the key files are gitignored -- keep it that way.

## Code layout (`pi/camera-project/`)

| File | Purpose |
|---|---|
| `mjpeg_crop.py` | **Current 64MP stream**: true 1:1 centre crop, `--mode 720` (default, ~65fps), `1080` (~45fps), `3k` (3072x1728, ~27fps streamed; `--fps 25` locks a steady 25), `4k` (~17fps streamed, ~100 Mbit/s, ~300% CPU), `full` (centre 720p from the full-res mode, ~2.3fps). `--fps N` locks any mode to N fps. Page: video fitted to the window, tap-to-focus, Fast/Precise AF switch, manual focus, digital zoom 1-16x on a log slider (ISP `ScalerCrop` of the 1:1 crop, scaled back up by the ISP; 1.0x = 1:1; capped per mode at the ISP's 64x64 minimum crop -- 720p/full ~10.9x), zoom sharpening in the ISP (`Sharpness` = 1 + strength * log2(zoom), strength slider 0-2; the ISP sharpens before upscaling, so it adds edge contrast but can't remove upscale blur; post-upscale software USM cost ~100ms/frame at 3K for little gain, see `tools/zoom_sharpen.py`, `tools/usm_test.py`) |
| `mjpeg_server.py` | Older 64MP server: zoom slider over the *old* custom driver's crop modes (1968x1476..3200x2400) -- those modes no longer exist with our driver, and weren't really 1:1 anyway (see below) |
| `tools/` | Measurement scripts, run on the Pi from `~/camera-project` (data in `work/`): `bin_test.py` (is a mode really 1:1?), `llp_sweep_one.py` (one line-length trial), `soak.py` (per-frame integrity), `af_trace.py` / `af_accuracy.py` / `lens_latency.py` (AF behaviour) |
| `mjpeg_server_csi64mp_backup.py` | Older 64MP server: single mode + ScalerCrop/software zoom + stabilization |
| `mjpeg_server_imx477.py` | IMX477 HQ camera server (manual lens, no AF) |
| `arducam_server.py` | USB 48MP webcam via OpenCV/V4L2, software zoom/AF sweep |
| `tuning_64mp_sharper.json` | libcamera tuning file used by `mjpeg_server.py` |
| `models/` | OpenCV Haar cascade (face detection) |
| `install_pivariety_pkgs.sh`, `*_links.txt`, `packages.txt` | Arducam Pivariety driver/libcamera installer and package links |

All servers serve on port 8080 (`http://192.168.159.116:8080/index.html`), so
only run one at a time. Shared architecture: capture thread -> parallel JPEG
encoder pool, plus a decoupled stabilization worker.

## Workflow

1. Edit files under `pi/camera-project/` here.
2. `scp` them to `~/camera-project/` on the Pi.
3. Run on the Pi, e.g. `cd ~/camera-project && python3 mjpeg_server.py > mjpeg_server.log 2>&1 &`
   and stop it with `pkill -f mjpeg_server.py`.
4. If you change files on the Pi directly, copy them back here before committing.

Images, logs, `.deb` files, `captures/` and `work/` (scratch output) are not tracked.

## Sensor driver (`driver/`)

`driver/arducam_64mp/arducam_64mp.c` is upstream raspberrypi/linux
`rpi-6.18.y` (first commit, unmodified) plus our changes (`git diff` that
commit to see them, all marked `CV_Camera`):

- Three **true 1:1 centre-crop** modes, built from the stock 8000x6000 1:1
  register set (`crop_1to1_base_regs`) plus a per-mode window list:
  1280x768, 1920x1104, 3072x1728, 3840x2208. Heights are 768/1104/2208 rather than
  720/1080/2160 so they don't collide with the stock *binned* modes of those
  sizes; the ISP crops to 720/1080/2160. All verified 1:1 with
  `tools/bin_test.py`.
- Module params `llp_720p` / `llp_1080p` / `llp_3k` / `llp_4k` (line length per mode)
  and `crop_min_vblank`, writable at runtime via
  `/sys/module/arducam_64mp/parameters/` and read on the next camera start --
  no reload needed to tune. Below a mode's minimum line length the image is
  garbage, not just slow.

| Mode | Min llp (measured) | Default | fps | Limit |
|---|---|---|---|---|
| 1280x768 | 15000 ok, 14800 bad | 16000 | 64.8 | sensor row time ~16.7us |
| 1920x1104 | 15000 ok, 14000 bad | 16000 | 46.7 | same sensor row time |
| 3072x1728 | 16000 ok, 15500 bad | 17000 | 29.0 | 2-lane MIPI (largest 16:9 crop that clears 25fps) |
| 3840x2208 | 20000 ok, 19500 bad | 21500 | 18.1 | 2-lane MIPI: 456MHz DDR x2 = 1.824Gbps = 182.4Mpx/s RAW10, so a 3840px row is 21.05us + per-line overhead |

vblank < 100 intermittently drops rows. All defaults: 0 corrupt frames in
30s soaks (`tools/soak.py`). The sensor row-time floor doesn't scale with
width, so it's internal (readout/remosaic); only 3K and 4K are link-bound.
Rough model for a new crop: min llp = max(15000, 4.93 * width + ~1000), then
fps = 900e6 / ((height + 100) * llp * 1.07). New modes also need width % 16,
height % 12 and a window start on a 16-px (x) / 8-px (y) grid. More 4K
fps would need fewer rows, a higher link frequency (undocumented PLL regs),
or 4 lanes (CM4 CAM1 is 4-lane; unverified whether this module wires 4).

Install: copy `driver/` to the Pi, run `./install.sh`, then **reboot**
(`./install.sh --rollback` restores the module that was there before,
`arducam_64mp.ko.pre-cv-camera`). **Never `rmmod`/`modprobe` the sensor
driver**: `bcm2835_unicam_legacy` can't take the subdev re-registering and the
kernel BUGs in `media_gobj_create` -- only a reboot recovers.

Why the old custom driver (built 2026-09-25, source lost; backed up as
`arducam_64mp.ko.pre-cv-camera`) was replaced: its "1:1" crop modes were
effectively 2x2-binned and re-expanded inside the sensor (raw correlated 0.944
with 2x2-binned full-res vs 0.923 with true full-res; blocky edges), and they
kept the full mode's 46770-px line length, capping them at ~10fps. That line
length is what a 9152-px row needs over the 2-lane link; a 1280-px row doesn't.

## Autofocus

The sensor (IMX682) does **PDAF on-chip**: each frame's embedded data carries a
16x12 phase/confidence grid (3rd embedded line), parsed by Arducam's libcamera
fork (`CamHelperArducam64MP::parsePdafData`) and used by `rpi.af` -- no driver
change was needed. The lens is a separate VCM (`dw9807_vcm`); LensPosition is
dioptres, 0-15 -> VCM 0-1023 (`rpi.af` "map" in the tuning file).

Measured (tools/af_trace.py, tools/af_accuracy.py, tools/lens_latency.py):
- Phase confidence scales with the crop: the PDAF grid covers the readout
  window, so 3k/4k cells hold plenty of PD pixels (pure phase), 720p/1080p
  have ~0 confidence when badly defocused and libcamera falls back to a
  contrast scan first.
- Phase AF lands 0.1-0.2 dioptre short of the true sharpness peak; the sharp
  zone is only ~0.3 dioptre wide. libcamera's one-shot (AfMode Auto) cycle is
  worse: slower from far defocus and 0.2-0.3 off. Its continuous mode, stopped
  once converged, is the fast path.
- A LensPosition change reaches the image ~6-9 frames later (queued
  requests); frame metadata's LensPosition tracks the image within ~1 frame.

`mjpeg_crop.py` tap-to-focus: tap/click the video (or "Focus centre", which
also runs at startup). Phase stage = continuous AF with a 40%-of-frame AF
window on the tap (20% was too few PDAF cells -> fell back to scanning), stopped
when AfState=Focused and the lens is still for 4 frames, then held. Page
switch: **Fast** = phase stage only (3k 0.9s, 4k 1.4s from far defocus;
720p/1080p ~2.5s with the scan); **Precise** = then a contrast sweep of the
20% tapped ROI, lens stepped every frame (+-0.30 dioptre, 0.03 steps), up and
down passes averaged, parabolic fit (adds ~1.5-2.5s; all modes land on the
same peak). "Continuous AF" button = plain libcamera continuous, centre-weighted.
With digital zoom, taps map through the zoomed view; the AF window is kept
>= 40% of the 1:1 crop (the PDAF grid covers the raw readout, not the zoomed
view -- 25% at 3x zoom fell back to scanning, 3.2s vs 1.0s).

## Conventions

- Comments explain *why*, including hardware facts found by testing (mode lists,
  camera indices, bandwidth limits). Keep that style and re-verify such facts
  when the hardware changes.
