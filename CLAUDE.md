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
| `mjpeg_crop720.py` | **Current 64MP stream**: true 1:1 centre 1280x720 at ~65fps from the patched driver's 1280x768 crop mode (`--full` = same pixels from the full-res mode, ~2.3fps) |
| `mjpeg_server.py` | Older 64MP server: zoom slider over the *old* custom driver's crop modes (1968x1476..3200x2400) -- those modes no longer exist with our driver, and weren't really 1:1 anyway (see below) |
| `tools/` | Measurement scripts, run on the Pi from `~/camera-project` (data in `work/`): `bin_test.py` (is a mode really 1:1?), `llp_sweep_one.py` (one line-length trial), `soak.py` (per-frame integrity) |
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

- New **1280x768 true 1:1 centre crop** mode, registers copied from the stock
  8000x6000 1:1 mode with only window/size/frame length changed. 768 not 720
  so it doesn't collide with the stock *binned* 1280x720 mode; crop to 720 in
  the ISP. Window x 3984..5263, y 3088..3855.
- Module params `crop_llp` (line length, default 16000) and `crop_min_vblank`
  (default 100), writable at runtime via `/sys/module/arducam_64mp/parameters/`
  and read on the next camera start -- no reload needed to tune.
- Measured limits: llp >= 15000 gives a correct image (70fps), <= 14800 is
  garbage; vblank < 100 intermittently drops rows. 16000/100 = 64.8fps, 0 bad
  frames in a 1942-frame soak.

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

## Focusing the 64MP

Autofocus (even with `--autofocus-window`) can land slightly off -- the depth
of field at full res is very shallow (sharpness peak only ~0.2 dioptre wide).
For a critical subject, sweep `--autofocus-mode manual --lens-position <d>` in
0.1 steps at 4624x3472, score the subject ROI with Laplacian variance, then
shoot 9248x6944 at the best position. Objects ~on the bench behind the box
were sharpest at ~6.65.

## Conventions

- Comments explain *why*, including hardware facts found by testing (mode lists,
  camera indices, bandwidth limits). Keep that style and re-verify such facts
  when the hardware changes.
