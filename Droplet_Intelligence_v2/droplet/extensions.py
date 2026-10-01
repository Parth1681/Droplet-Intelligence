"""Two extensions toward surfaces outside the training set (October 2026).

Step 1  Physical image reader. The smooth plateau fraction is measured directly from one SEM image:
        laser tracks are rough (high local intensity variance), plateaus are smooth. The variance window
        is fixed in micrometres, so any magnification works once the pixel size is known (read from JEOL
        metadata, or supplied). The variance threshold is calibrated on training surfaces only.
        Candidates: phi from the image instead of the geometry proxy, with and without texture volume.

Step 2  Wettability. Contact angles enter through the Lee et al. (2016) correction
        sqrt(beta^2 - beta0^2), beta0 from the spherical cap at the advancing angle, or as a GP input.
        REF-H has no measured angle in the dataset; a literature value for a similar smooth coating is
        used and swept.

Protocol (same as the paper): leave one surface out on the 12 textured surfaces, paired bootstrap over
surfaces (4,000 draws, seed 23) against the baseline, REF-H predicted only after training on all
textured surfaces and never used for any choice.

    python -m droplet.extensions --workers 4
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, json, time, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .data import ROOT, load
from .models import optimizer
from .benchmark import bootstrap

OUT = ROOT / 'results' / 'extensions'
SEM = ROOT / 'data' / 'sem_original'
UM_PER_PX_AT_MAG1 = 500 / 432 * 43      # calibrated from the 500 um scale bar on the 43x images (2560 px wide)
WINDOW_UM = 12.7                        # local variance window, physical size
FOOTER_Y = 1880                         # JEOL annotation bar starts here
REFH_ANGLE_LIT = 114.5                  # static angle, fluorinated phosphonic SAM on smooth Al (Molecules 2024)
Z90 = 1.6448536269514722


# ------------------------------------------------------------------ Step 1: image reader
def pixel_size(path):
    """um per pixel from JEOL metadata (Mag/<n>)."""
    tag = Image.open(path).tag_v2.get(40094)
    if tag is None:
        raise ValueError(f'{path}: no JEOL magnification tag; supply the pixel size')
    meta = dict(x.split('/', 1) for x in tag.decode('utf-16-le').strip('\x00').split(';') if '/' in x)
    return UM_PER_PX_AT_MAG1 / float(meta['Mag'])


def roughness_map(path, um_per_px=None, footer_y=FOOTER_Y):
    """Local standard deviation of intensity over a WINDOW_UM square, footer removed."""
    um = um_per_px or pixel_size(path)
    a = np.asarray(Image.open(path).convert('L'), float)[:footer_y]
    a = ndi.gaussian_filter(a, 1)
    win = max(3, int(round(WINDOW_UM / um)) | 1)
    m = ndi.uniform_filter(a, win)
    return np.sqrt(np.maximum(ndi.uniform_filter(a * a, win) - m * m, 0))


def otsu(values, bins=512):
    h, edges = np.histogram(values, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1); m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


_MAPS = {}
def maps(names, mag=43):
    for n in names:
        if n not in _MAPS:
            _MAPS[n] = roughness_map(SEM / f'{n}_{mag}.tif')
    return {n: _MAPS[n] for n in names}


def calibrate_threshold(train_surfaces):
    m = maps(train_surfaces)
    return otsu(np.concatenate([x[::8, ::8].ravel() for x in m.values()]))


def image_phi(surface, threshold):
    """Smooth plateau fraction = share of pixels below the roughness threshold."""
    return float((maps([surface])[surface] <= threshold).mean())


# ------------------------------------------------------------------ Step 2: wettability
def angles():
    w = pd.read_csv(ROOT / 'data/raw/03 - Contact angles - Water.csv', encoding='utf-8-sig', index_col=0).T
    w.columns = ['static', 'advancing', 'receding', 'hysteresis']
    return w.astype(float)


def beta0(theta_deg):
    t = np.radians(theta_deg)
    return np.sin(t) * (4 / ((1 - np.cos(t)) ** 2 * (2 + np.cos(t)))) ** (1 / 3)


# ------------------------------------------------------------------ candidates
CANDIDATES = {
    'baseline':        dict(phi='geom', texvol=True,  wet=None),
    'image_phi':       dict(phi='image', texvol=True, wet=None),
    'image_only':      dict(phi='image', texvol=False, wet=None),
    'lee_beta0':       dict(phi='geom', texvol=True,  wet='lee'),
    'angle_input':     dict(phi='geom', texvol=True,  wet='feature'),
}


def design(d, spec, threshold, theta):
    d = d.copy()
    if spec['phi'] == 'image':
        cache = {s: image_phi(s, threshold) for s in d.surface.unique()}
        d['phi'] = d.surface.map(cache)
        d['texvol'] = (1 - d.phi) * d.depth
    cols = ['logRe', 'logWe', 'logD', 'phi'] + (['texvol'] if spec['texvol'] else [])
    if spec['wet'] == 'feature':
        d['cos_adv'] = np.cos(np.radians(d.surface.map(theta)))
        cols.append('cos_adv')
    d['b0'] = beta0(d.surface.map(theta)) if spec['wet'] == 'lee' else 0.0
    return d, cols


def target(d, spec):
    if spec['wet'] == 'lee':
        return np.log(np.sqrt(np.maximum(d.beta.values ** 2 - d.b0.values ** 2, 1e-6)))
    return np.log(d.beta.values)


def untarget(mu, sd, d, spec):
    """Back to ln(beta). For Lee: beta = sqrt(g^2 + beta0^2), first-order delta method for sd."""
    if spec['wet'] == 'lee':
        g2 = np.exp(2 * mu); b2 = g2 + d.b0.values ** 2
        return 0.5 * np.log(b2), sd * g2 / b2
    return mu, sd


def fit_predict(train, test, spec):
    y = target(train, spec)
    cols = train.attrs['cols']
    sc = StandardScaler().fit(train[cols])
    k = ConstantKernel(1.) * Matern(np.ones(len(cols)), nu=1.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(.001, (1e-6, .1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        gp = GaussianProcessRegressor(k, normalize_y=True, optimizer=optimizer, random_state=23).fit(sc.transform(train[cols]), y)
    mu, sd = gp.predict(sc.transform(test[cols]), return_std=True)
    return untarget(mu, sd, test, spec), str(gp.kernel_)


def job(task):
    name, held, refh_angle = task
    spec = CANDIDATES[name]
    with threadpool_limits(limits=1):
        t, r = load()
        theta = angles()['advancing'].to_dict(); theta['REF-H'] = refh_angle
        train_s = sorted(s for s in t.surface.unique() if s != held)
        thr = calibrate_threshold(train_s) if spec['phi'] == 'image' else None
        start = time.time()
        if held == 'REF-H':
            train, test = t, r
        else:
            train, test = t[t.surface != held], t[t.surface == held]
        train, cols = design(train, spec, thr, theta); test, _ = design(test, spec, thr, theta)
        train.attrs['cols'] = cols
        (mu, sd), kern = fit_predict(train, test, spec)
    return dict(model=name, heldout=held, refh_angle=refh_angle, row_id=test.row_id.tolist(), mu=mu.tolist(), sd=sd.tolist(),
                threshold=thr, phi_test=float(test.phi.iloc[0]), kernel=kern, seconds=time.time() - start)


def metrics(beta, mu, sd, groups):
    p = np.exp(mu); per = {s: float(np.sqrt(np.mean((beta[groups == s] - p[groups == s]) ** 2))) for s in np.unique(groups)}
    return dict(rmse=float(np.sqrt(np.mean((beta - p) ** 2))), mape=float(np.mean(np.abs(p - beta) / beta) * 100),
                bias=float(np.mean(p - beta)), coverage=float(np.mean(np.abs(np.log(beta) - mu) <= Z90 * sd)),
                worst=max(per.values()), worst_surface=max(per, key=per.get), per_surface=per)


def run(workers=4):
    OUT.mkdir(parents=True, exist_ok=True)
    t, r = load(); surfaces = sorted(t.surface.unique())
    tasks = [(m, s, REFH_ANGLE_LIT) for m in CANDIDATES for s in surfaces + ['REF-H']]
    tasks += [('lee_beta0', 'REF-H', a) for a in (100.0, 110.0, 120.0, 130.0)]
    tasks += [('angle_input', 'REF-H', a) for a in (100.0, 130.0)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as pool:
        res = list(pool.map(job, tasks))
    print(f'{len(res)} fits in {time.time() - t0:.0f}s', flush=True)

    # Step 1 check: image phi vs geometry phi, threshold calibrated on the 12 textured surfaces
    thr_all = calibrate_threshold(surfaces)
    geom = t.groupby('surface').phi.first().to_dict(); geom['REF-H'] = 1.0
    reader = {s: dict(image_phi=image_phi(s, thr_all), geometry_phi=float(geom[s]), um_per_px=pixel_size(SEM / f'{s}_43.tif'))
              for s in surfaces + ['REF-H']}
    reader_fold = {z['heldout']: z['phi_test'] for z in res if z['model'] == 'image_only' and z['heldout'] != 'REF-H'}

    beta = t.beta.values; groups = t.surface.values
    pred = {}; summary = {}
    for m in CANDIDATES:
        mu = np.zeros(len(t)); sd = np.zeros(len(t))
        for z in res:
            if z['model'] == m and z['heldout'] != 'REF-H':
                mu[z['row_id']] = z['mu']; sd[z['row_id']] = z['sd']
        pred[m] = (mu, sd)
        rz = next(z for z in res if z['model'] == m and z['heldout'] == 'REF-H' and z['refh_angle'] == REFH_ANGLE_LIT)
        summary[m] = dict(spec=CANDIDATES[m], loso=metrics(beta, mu, sd, groups),
                          refh=metrics(r.beta.values[rz['row_id']], np.array(rz['mu']), np.array(rz['sd']), r.surface.values[rz['row_id']]),
                          refh_phi_used=rz['phi_test'])
    for m in CANDIDATES:
        summary[m]['delta_vs_baseline_ci95'] = bootstrap(beta, np.exp(pred[m][0]), np.exp(pred['baseline'][0]), groups)
    sweep = {f"{z['model']}@{z['refh_angle']:g}": metrics(r.beta.values[z['row_id']], np.array(z['mu']), np.array(z['sd']), r.surface.values[z['row_id']])
             for z in res if z['heldout'] == 'REF-H' and z['model'] in ('lee_beta0', 'angle_input')}
    out = dict(protocol='LOSO on 12 textured surfaces; paired surface bootstrap 4000 draws seed 23 vs baseline; '
                        'REF-H predicted after training on all textured surfaces, not used for any choice.',
               image_reader=dict(window_um=WINDOW_UM, um_per_px_at_mag1=UM_PER_PX_AT_MAG1, threshold_all_textured=thr_all,
                                 per_surface=reader, heldout_fold_phi=reader_fold),
               angles_used='Water advancing angle per surface (file 03); REF-H assumed %.1f deg (literature), swept 100 to 130.' % REFH_ANGLE_LIT,
               candidates=summary, refh_angle_sweep=sweep, seconds=time.time() - t0)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2))
    rows = []
    for m in CANDIDATES:
        mu, sd = pred[m]
        rows += [dict(model=m, row_id=int(i), surface=s, beta=b, prediction=float(np.exp(u)), mu_log=float(u), sd_log=float(v))
                 for i, s, b, u, v in zip(t.row_id, groups, beta, mu, sd)]
    pd.DataFrame(rows).to_csv(OUT / 'loso_predictions.csv', index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4)
    o = run(ap.parse_args().workers)
    for m, s in o['candidates'].items():
        print(f"{m:12s} LOSO {s['loso']['rmse']:.4f} cov {s['loso']['coverage']:.3f} CI {s['delta_vs_baseline_ci95']} | "
              f"REF-H {s['refh']['rmse']:.4f} mape {s['refh']['mape']:.2f}% cov {s['refh']['coverage']:.3f} bias {s['refh']['bias']:+.4f} phi {s['refh_phi_used']:.3f}")
    for k, v in o['refh_angle_sweep'].items():
        print(f"  sweep {k:18s} REF-H {v['rmse']:.4f} cov {v['coverage']:.3f} bias {v['bias']:+.4f}")
    for s, v in o['image_reader']['per_surface'].items():
        print(f"  reader {s:6s} image {v['image_phi']:.3f} geometry {v['geometry_phi']:.3f}")
