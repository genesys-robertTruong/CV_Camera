"""How many frames after set_controls(LensPosition) does frame metadata report
the new position, and when does the image actually change? (3k mode)"""
import os, time
import cv2, numpy as np
from picamera2 import Picamera2

p = Picamera2(0, tuning=Picamera2.load_tuning_file(os.path.abspath("tuning_64mp_sharper.json")))
p.configure(p.create_video_configuration(main={"size": (3072, 1728), "format": "RGB888"},
                                         raw={"size": (3072, 1728)}, buffer_count=4))
p.start(); p.set_controls({"AfMode": 0, "LensPosition": 6.0}); time.sleep(1.5)

def sharp(img):
    g = cv2.cvtColor(img[600:1100, 1200:1900], cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

for target in (6.8, 6.2, 6.86):
    t0 = time.time(); p.set_controls({"LensPosition": target}); rows = []
    for i in range(10):
        req = p.capture_request(); md = req.get_metadata(); s = sharp(req.make_array("main")); req.release()
        rows.append(f"{i}:{md['LensPosition']:.3f}/{s:.0f}")
    print(f"-> {target}: " + "  ".join(rows))
p.stop(); p.close()
