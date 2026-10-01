#!/bin/sh
# Build and install the patched arducam_64mp driver on the Pi.
# Run on the Pi from this directory:  ./install.sh
#   ./install.sh --rollback   restore the module that was installed before
#                             this script first ran.
# Both need a REBOOT to take effect. Don't rmmod/modprobe the sensor driver:
# bcm2835_unicam_legacy can't handle the sensor subdev re-registering
# ("Rejecting subdev ... (Already set!!)") and the kernel BUGs in
# media_gobj_create -- seen 2026-10-01, needed a reboot to recover.
#
# The 1:1 crop mode's timing is tuned at runtime without reloading:
#   echo 12000 | sudo tee /sys/module/arducam_64mp/parameters/crop_llp
#   echo 100   | sudo tee /sys/module/arducam_64mp/parameters/crop_min_vblank
# (read on the next camera start). Persist with /etc/modprobe.d/arducam_64mp.conf:
#   options arducam_64mp crop_llp=12000 crop_min_vblank=100
set -eu
cd "$(dirname "$0")"

KO_DIR=/lib/modules/$(uname -r)/kernel/drivers/media/i2c
KO=$KO_DIR/arducam_64mp.ko
BACKUP=$KO_DIR/arducam_64mp.ko.pre-cv-camera

if [ "${1:-}" = "--rollback" ]; then
    [ -f "$BACKUP" ] || { echo "no backup at $BACKUP"; exit 1; }
    sudo cp "$BACKUP" "$KO"
else
    make -C arducam_64mp
    # Keep the module that was there before we ever touched it (only once).
    [ -f "$BACKUP" ] || sudo cp "$KO" "$BACKUP"
    sudo cp arducam_64mp/arducam_64mp.ko "$KO"
fi
sudo depmod -a
echo "Installed. Reboot to load it: sudo reboot"
