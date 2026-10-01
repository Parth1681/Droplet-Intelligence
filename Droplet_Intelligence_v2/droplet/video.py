"""Measure a drop impact from a backlit side-view video and compare it with the model, automatically.

    python -m droplet.video --video impact.mp4 --fps 5000 --mm-per-px 0.02 --fluid 0 --surface D200
    python -m droplet.video --watch incoming/ --fps 5000 --mm-per-px 0.02 --fluid 0 --surface D200

Per video, with no manual steps:
  1. frames are decoded with ffmpeg and turned to greyscale;
  2. the drop and the substrate are dark on a bright background: one Otsu threshold over the video
     separates them, and the substrate is the dark band that spans the frame width at the bottom;
  3. before contact the drop is the largest dark blob above the substrate: D0 from its area
     (equivalent circle), V from a straight-line fit of its centroid height against time;
  4. after contact the spreading diameter D(t) is the blob's horizontal extent; beta_max = max D / D0;
  5. the GP predicts beta_max from D0, V, the fluid and the surface (as soon as D0 and V are known),
     and the measurement is checked against the 90% interval.

--watch processes every new video that appears in a folder and appends a row to results.csv there.
The measurement steps follow the usual high-speed analysis (e.g. JoVE 60778); scale and capture
frame rate must come from the experiment.
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from .data import ROOT
from .predict import Predictor

FLUIDS_NOTE = 'preset keys 0, 20, 60, 78, 91 (wt% glycerol) from models/release.json'


def read_frames(path, max_frames=4000):
    p = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                        '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True)
    W, H = map(int, p.stdout.strip().split(',')[:2])
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-frames:v', str(max_frames), '-f', 'rawvideo',
                          '-pix_fmt', 'gray', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def otsu(values):
    h, e = np.histogram(values, bins=256, range=(0, 256)); c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def analyse(frames, fps, mm_per_px, baseline=None, threshold=None, fit_frames=8):
    N, H, W = frames.shape
    thr = threshold if threshold is not None else otsu(frames[:, ::2, ::2].ravel())
    dark = frames < thr
    if baseline is None:                      # substrate: rows dark across >= 80% of the width in most frames
        rowfrac = np.median(dark.mean(2), 0)
        sub = np.where(rowfrac >= .8)[0]
        if not len(sub):
            raise ValueError('No substrate band found; pass --baseline (pixel row of the surface)')
        baseline = _band_top(sub)
    recs = []
    for i in range(N):
        m = dark[i, :baseline]
        lab, n = ndi.label(m)
        if not n:
            recs.append(None); continue
        sizes = ndi.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
        if sizes[k - 1] < 20:
            recs.append(None); continue
        ys, xs = np.nonzero(lab == k)
        recs.append(dict(area=float(len(ys)), cy=float(ys.mean()), top=int(ys.min()), bottom=int(ys.max()),
                         left=int(xs.min()), right=int(xs.max())))
    touch = [i for i, r in enumerate(recs) if r and r['bottom'] >= baseline - 2]
    if not touch:
        raise ValueError('The drop never reaches the substrate in this video')
    contact = touch[0]
    pre = [i for i in range(contact) if recs[i] and recs[i]['top'] > 0][-fit_frames:]
    if len(pre) < 3:
        raise ValueError('Fewer than 3 frames of the falling drop before contact')
    d0_px = float(np.median([2 * np.sqrt(recs[i]['area'] / np.pi) for i in pre]))
    slope = np.polyfit(pre, [recs[i]['cy'] for i in pre], 1)[0]           # px per frame, downward positive
    V = slope * mm_per_px * 1e-3 * fps
    after = [(i, recs[i]['right'] - recs[i]['left'] + 1) for i in range(contact, N) if recs[i]]
    widths = np.array([w for _, w in after], float)
    j = int(np.argmax(widths)); dmax_px = float(widths[j])
    series = [dict(t_ms=(i - contact) / fps * 1e3, D_mm=w * mm_per_px) for i, w in after]
    return dict(D0_mm=d0_px * mm_per_px, V=float(V), beta_max=dmax_px / d0_px, Dmax_mm=dmax_px * mm_per_px,
                t_max_ms=(after[j][0] - contact) / fps * 1e3, contact_frame=contact, baseline_row=int(baseline),
                threshold=thr, frames=N, pre_contact_frames=pre, series=series)


def _band_top(sub):
    """Top row of the bottom-most contiguous run of substrate rows."""
    top = sub[-1]
    for r in sub[::-1][1:]:
        if r == top - 1: top = r
        else: break
    return int(top)


def run_one(path, a, predictor=None):
    m = analyse(read_frames(path), a.fps, a.mm_per_px, a.baseline, a.threshold)
    predictor = predictor or Predictor()
    fl = predictor.release['fluids'][a.fluid] if a.fluid else dict(rho=a.rho, mu=a.mu, sigma=a.sigma)
    p = predictor.predict(dict(D_mm=m['D0_mm'], V=m['V'], surface=a.surface, **fl))
    lo, hi = p['interval']['lower'], p['interval']['upper']
    out = dict(video=str(path), measured=dict(D0_mm=m['D0_mm'], V=m['V'], beta_max=m['beta_max'], t_max_ms=m['t_max_ms']),
               predicted=dict(beta_max=p['beta_max'], interval_90=[lo, hi], status=p['reliability']['status'],
                              reasons=p['reliability']['reasons']),
               agreement='inside interval' if lo <= m['beta_max'] <= hi else 'OUTSIDE interval: check surface, fluid or setup',
               error=m['beta_max'] - p['beta_max'], detection=dict(contact_frame=m['contact_frame'], baseline_row=m['baseline_row'],
               threshold=m['threshold'], frames=m['frames']), series=m['series'])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--video'); g.add_argument('--watch', help='folder to watch for new videos')
    ap.add_argument('--fps', type=float, required=True, help='capture frame rate of the camera (frames per second)')
    ap.add_argument('--mm-per-px', type=float, required=True)
    ap.add_argument('--fluid', default='0', help=FLUIDS_NOTE + '; or give --rho --mu --sigma with --fluid ""')
    ap.add_argument('--rho', type=float); ap.add_argument('--mu', type=float); ap.add_argument('--sigma', type=float)
    ap.add_argument('--surface', default='D200', help='surface name (D50..S800, REF-H)')
    ap.add_argument('--baseline', type=int, help='pixel row of the substrate (auto if omitted)')
    ap.add_argument('--threshold', type=float, help='grey level separating drop from background (auto if omitted)')
    a = ap.parse_args()
    if a.video:
        print(json.dumps({k: v for k, v in run_one(a.video, a).items() if k != 'series'}, indent=2)); return
    folder = Path(a.watch); done = set(); log = folder / 'results.csv'; pred = Predictor()
    if not log.exists():
        log.write_text('video,D0_mm,V_ms,beta_measured,beta_predicted,lower,upper,agreement,status\n')
    print(f'Watching {folder} for .mp4/.avi/.mov/.webm files. Ctrl+C to stop.', flush=True)
    while True:
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.webm', '.mkv') and f not in done:
                s1 = f.stat().st_size; time.sleep(1)
                if f.stat().st_size != s1: continue                        # still being written
                done.add(f)
                try:
                    r = run_one(f, a, pred); m, p = r['measured'], r['predicted']
                    with log.open('a') as fh:
                        fh.write(f"{f.name},{m['D0_mm']:.4f},{m['V']:.4f},{m['beta_max']:.4f},{p['beta_max']:.4f},"
                                 f"{p['interval_90'][0]:.4f},{p['interval_90'][1]:.4f},{r['agreement']},{p['status']}\n")
                    (f.with_suffix('.json')).write_text(json.dumps(r, indent=2))
                    print(f"{f.name}: measured {m['beta_max']:.3f}, predicted {p['beta_max']:.3f} [{p['interval_90'][0]:.3f}, {p['interval_90'][1]:.3f}] -> {r['agreement']}", flush=True)
                except Exception as e:
                    print(f'{f.name}: could not analyse ({e})', flush=True)
        time.sleep(2)


if __name__ == '__main__':
    main()
