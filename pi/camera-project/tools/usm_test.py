"""Post-upscale software sharpening: visual comparison on the zoom_sharpen.py
tiles, and per-frame cost at 3072x1728 on the CM4."""
import time, cv2, numpy as np

def usm_fast(img, zoom, amount):
    """Unsharp mask whose radius follows the zoom. The blur is a resize down by
    ~zoom/2 and back up (cheap stand-in for a big Gaussian)."""
    f = max(1.0, zoom / 2)
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w / f)), max(1, int(h / f))), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

def usm_gauss(img, zoom, amount):
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=0.6 * zoom)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)

for z in (8, 16):
    g = cv2.imread(f"work/zs_{z}.png"); tiles = [g[i * 600:(i + 1) * 600] for i in range(4)]
    base = tiles[1]  # ISP Sharpness 3
    rows = [base]
    for name, fn, a in (("fast", usm_fast, 1.0), ("fast", usm_fast, 2.0), ("gauss", usm_gauss, 1.5)):
        t = fn(base, z, a).copy()
        cv2.rectangle(t, (0, 50), (700, 100), (0, 0, 0), -1)
        cv2.putText(t, f"+ SW USM {name} amount {a:g}", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        rows.append(t)
    cv2.imwrite(f"work/usm_{z}.png", np.vstack(rows))

frame = cv2.resize(cv2.imread("work/zs_16.png")[:600], (3072, 1728))
for name, fn in (("fast", usm_fast), ("gauss", usm_gauss)):
    for z in (2, 8, 16):
        fn(frame, z, 1.0); t = time.perf_counter()
        for _ in range(10): fn(frame, z, 1.0)
        print(f"{name} zoom {z}: {(time.perf_counter() - t) * 100:.1f} ms/frame at 3072x1728")
t = time.perf_counter()
for _ in range(10): cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f"(for scale: imencode {(time.perf_counter() - t) * 100:.1f} ms/frame)")
