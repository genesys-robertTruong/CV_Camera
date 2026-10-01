"""Compare effective resolution: crop mode (1968x1476) raw vs full-res (9248x6944) DNG raw,
over the same centre 1280x720 sensor pixels."""
import struct, time, os, sys, numpy as np, cv2

def dng_raw(path):
    f = open(path, "rb"); bo = "<" if f.read(2) == b"II" else ">"
    f.read(2); (ifd0,) = struct.unpack(bo + "I", f.read(4))
    def read_ifd(off):
        f.seek(off); (n,) = struct.unpack(bo + "H", f.read(2)); tags = {}
        for _ in range(n):
            tag, typ, cnt, val = struct.unpack(bo + "HHII", f.read(12)); tags[tag] = (typ, cnt, val)
        return tags
    def vals(t):
        typ, cnt, val = t; size = {3: 2, 4: 4, 13: 4}[typ]; fmt = {3: "H", 4: "I", 13: "I"}[typ]
        if cnt * size <= 4:
            raw = struct.pack(bo + "I", val)[: cnt * size]
        else:
            f.seek(val); raw = f.read(cnt * size)
        return struct.unpack(bo + fmt * cnt, raw)
    ifds = [read_ifd(ifd0)]
    if 330 in ifds[0]:
        ifds += [read_ifd(o) for o in vals(ifds[0][330])]
    for t in ifds:
        w, h = vals(t[256])[0], vals(t[257])[0]
        if w > 5000:
            bps = vals(t[258])[0]; off = vals(t[273])[0]
            return np.memmap(path, dtype="<u2" if bps == 16 else np.uint8, mode="r", offset=off, shape=(h, w)), bps
    raise SystemExit("raw IFD not found")

def hf_ratio(plane):
    """Fraction of spectral power above half-Nyquist (|f| > 0.25 cycles/px) on a Bayer plane."""
    p = plane.astype(np.float64); p -= p.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(p * np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(p.shape[0]))[:, None]; fx = np.fft.fftshift(np.fft.fftfreq(p.shape[1]))[None, :]
    hi = (np.abs(fx) > 0.25) | (np.abs(fy) > 0.25); dc = (fx == 0) & (fy == 0)
    return P[hi].sum() / P[~dc].sum()

def green_zoom(c, label):
    g = c[0::2, 1::2].astype(np.float32); g = np.clip((g - g.min()) / (np.percentile(g, 99.5) - g.min()) * 255, 0, 255).astype(np.uint8)
    z = cv2.resize(g[90:270, 160:480], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.putText(z, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3); return z

if __name__ == "__main__":
    full, bps = dng_raw("work/full.dng")
    cy, cx = 6944 // 2, 9248 // 2
    cf = np.array(full[cy - 360: cy + 360, cx - 640: cx + 640]).astype(np.float32)
    crop = np.load("work/raw1968.npy").astype(np.float32)
    cc = crop[738 - 360: 738 + 360, 984 - 640: 984 + 640]
    for name, c in [("full 9248 mode", cf), ("crop 1968 mode", cc)]:
        print(f"{name}: mean {c.mean():.0f}  HF ratio Gr {hf_ratio(c[0::2,1::2]):.4f}  Gb {hf_ratio(c[1::2,0::2]):.4f}")
    cv2.imwrite("work/raw_green_full_vs_crop.png", np.vstack([green_zoom(cf, "full 9248 raw G"), green_zoom(cc, "crop 1968 raw G")]))
