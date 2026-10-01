# CV_Camera

Computer-vision camera project running on a Raspberry Pi. Code is developed in
this Windows repo and runs on the Pi; there is no git on the Pi itself.

## Target device

- Host: `william@192.168.159.116` (hostname `cv-camera`)
- Raspberry Pi, aarch64, Debian 13 (trixie), Python 3.13, OpenCV 4.10, picamera2
- ~900 MB RAM, ~1.7 GB free on the SD card -- keep large captures off it
- Project dir on the Pi: `~/camera-project` (mirrored here in `pi/camera-project/`)
- Cameras: Arducam 64MP (Hawk-Eye, CSI, custom `arducam_64mp` driver with
  extra 1x1 crop modes). Code also exists for an IMX477 HQ camera (currently
  removed) and an Arducam B0478 USB 48MP (USB2-limited to 1280x720 @ 10fps).
  Check `rpicam-hello --list-cameras` before assuming which are attached.

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
| `mjpeg_server.py` | Main server: Arducam 64MP live MJPEG view. Zoom switches between real sensor crop modes; each mode runs in a fresh worker subprocess (reusing one Picamera2 across reconfigures times out after ~3 switches) |
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

Images, logs, `.deb` files and `captures/` are not tracked.

## Conventions

- Comments explain *why*, including hardware facts found by testing (mode lists,
  camera indices, bandwidth limits). Keep that style and re-verify such facts
  when the hardware changes.
