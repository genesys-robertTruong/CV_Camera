"""Compare ISP sharpening strengths at several digital-zoom levels (3k mode).
Writes work/zs_<zoom>.png: one row per Sharpness value, centre region of the
zoomed output, plus the measured ScalerCrop actually applied."""
import os, time, sys
import cv2, numpy as np
from picamera2 import Picamera2

ZOOMS = [float(z) for z in (sys.argv[1:] or ["4", "8", "16"])]
SHARP = [1.0, 3.0, 6.0, 12.0]
p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (3072, 1728), "format": "RGB888"},
                                         raw={"size": (3072, 1728)}, buffer_count=4))
p.start(); p.set_controls({"AfMode": 0, "LensPosition": 6.89}); time.sleep(1.5)
x, y, w, h = p.camera_controls["ScalerCrop"][1]
print("ScalerCrop limits", p.camera_controls["ScalerCrop"], "Sharpness", p.camera_controls.get("Sharpness"))
for z in ZOOMS:
    cw = max(64, int(w / z)) & ~1; ch = max(64, int(round(cw * h / w))) & ~1
    crop = (x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)
    rows = []
    for sh in SHARP:
        p.set_controls({"ScalerCrop": crop, "Sharpness": sh}); time.sleep(0.6)
        req = p.capture_request(); img = req.make_array("main"); md = req.get_metadata(); req.release()
        tile = img[1728 // 2 - 300: 1728 // 2 + 300, 3072 // 2 - 700: 3072 // 2 + 700].copy()
        cv2.putText(tile, f"zoom {z:g}x  Sharpness {sh:g}  crop {md['ScalerCrop'][2]}x{md['ScalerCrop'][3]}",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        rows.append(tile)
    cv2.imwrite(f"work/zs_{z:g}.png", np.vstack(rows))
    print("wrote", z, crop)
p.stop(); p.close()
