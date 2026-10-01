"""Surface descriptors defined for ANY surface, textured or smooth (Phase 3).

From SEM (43x, 1.157 um/px): the laser tracks are rough, superhydrophobic bands of width w; the plateaus between
them are smooth and look like REF-H (Data in Brief 61:111697 says REF-H "resembles the surface between the
laser-made channels"). A square grid of pitch s therefore leaves a smooth-plateau area fraction
    phi = (max(s - w, 0) / s)^2        (phi = 1 for REF-H, by definition)
Track widths were measured by local-variance segmentation of the SEM images (see measure_tracks), then pooled
per depth class over s >= 200 um, where tracks do not merge:  deep ~45 um, shallow ~30 um (+/- ~10 um).
"""
import glob, os
import numpy as np

W_TRACK = {25: 45.0, 6: 30.0}          # um, measured; sensitivity-test +/-10 um
PX_43X = 500 / 432                      # um per pixel at 43x


def phi_smooth(spacing, depth, w=W_TRACK):
    spacing, depth = np.asarray(spacing, float), np.asarray(depth, float)
    wid = np.where(depth >= 15, w[25], w[6])
    out = np.where(spacing > 0, (np.clip(spacing - wid, 0, None) / np.where(spacing > 0, spacing, 1)) ** 2, 1.0)
    return np.where(depth <= 0, 1.0, out)


def add_surface(d, w=W_TRACK):
    d = d.copy()
    d["phi"] = phi_smooth(d.spacing, d.depth, w)          # smooth-plateau fraction
    d["tex"] = 1 - d["phi"]                               # laser-textured fraction
    d["texvol"] = d["tex"] * d["depth"]                   # um: texture volume per area (air-cushion proxy)
    return d


def measure_tracks(sem_dir, win=11):
    """Textured-area fraction per surface from 43x SEM, and implied track width. Returns dict name -> (frac, w)."""
    from PIL import Image
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    S = {}
    for f in glob.glob(os.path.join(sem_dir, "*_43.tif")):
        a = ndi.gaussian_filter(np.asarray(Image.open(f), float)[:1880], 1)
        m = ndi.uniform_filter(a, win)
        S[os.path.basename(f)[:-7]] = np.sqrt(np.maximum(ndi.uniform_filter(a * a, win) - m * m, 0))
    thr = threshold_otsu(np.concatenate([s[::8, ::8].ravel() for s in S.values()]))
    out = {}
    for n, s in S.items():
        frac = float((s > thr).mean())
        if n == "REF-H": out[n] = (frac, 0.0); continue
        sp = int(n[1:]); out[n] = (frac, sp * (1 - np.sqrt(1 - min(frac, 0.999))) - (win - 1) * PX_43X)
    return out
