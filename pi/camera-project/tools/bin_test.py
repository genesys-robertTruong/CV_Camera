"""Is the crop-mode raw really 2x2-binned data? Compare it against the true full-res raw
(same sensor pixels) and against that same full-res raw degraded by 2x2 binning per Bayer plane."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from res_test import dng_raw  # noqa: E402

full, _ = dng_raw("work/full.dng")
cy, cx = 6944 // 2, 9248 // 2
cf = np.array(full[cy - 360: cy + 360, cx - 640: cx + 640]).astype(np.float64)
crop = np.load(sys.argv[1] if len(sys.argv) > 1 else "work/raw1968.npy").astype(np.float64)
ch, cw = crop.shape; cc = crop[ch // 2 - 360: ch // 2 + 360, cw // 2 - 640: cw // 2 + 640]

def plane(c): return c[0::2, 1::2]
def norm(a): a = a - a.mean(); return a / a.std()
def corr(a, b): return float((norm(a) * norm(b)).mean())
def bin2_up(p, oy=0, ox=0):
    """2x2 box-bin a Bayer plane (with phase oy,ox) then nearest-upsample back to the same grid."""
    q = np.pad(p, ((oy, 2), (ox, 2)), mode="edge")
    h, w = (q.shape[0] // 2) * 2, (q.shape[1] // 2) * 2
    b = q[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    return np.repeat(np.repeat(b, 2, 0), 2, 1)[oy: oy + p.shape[0], ox: ox + p.shape[1]]

pf, pc = plane(cf), plane(cc)
# search a small shift in case the two windows aren't pixel-aligned
best = max(((corr(np.roll(np.roll(pf, dy, 0), dx, 1)[4:-4, 4:-4], pc[4:-4, 4:-4]), dy, dx)
            for dy in range(-3, 4) for dx in range(-3, 4)))
print(f"crop vs full (true 1:1):           corr {best[0]:.4f}  at shift {best[1:]}")
pf = np.roll(np.roll(pf, best[1], 0), best[2], 1)
for oy in (0, 1):
    for ox in (0, 1):
        print(f"crop vs full-2x2-binned phase {oy}{ox}: corr {corr(bin2_up(pf, oy, ox)[4:-4, 4:-4], pc[4:-4, 4:-4]):.4f}")
# blockiness inside the crop plane itself: are pixel pairs (2k,2k+1) more alike than (2k+1,2k+2)?
for name, p in [("full", pf), ("crop", pc)]:
    dx = np.abs(np.diff(p, axis=1)); dy = np.abs(np.diff(p, axis=0))
    print(f"{name} plane neighbour diff  x even/odd {dx[:, 0::2].mean():.2f}/{dx[:, 1::2].mean():.2f}"
          f"  y even/odd {dy[0::2].mean():.2f}/{dy[1::2].mean():.2f}")
