"""Predict beta_max for a new surface from one SEM image (no spacing or depth needed).

    python -m droplet.predict_image --image new_43x.tif --D_mm 2.5 --V 1.5 --rho 997.2 --mu 0.000944 --sigma 0.0725
    python -m droplet.predict_image --image other.png --um-per-px 1.16 --footer-y 0 ...

The smooth plateau fraction phi is read from the image (droplet.extensions, step 1) and fed to the
"image_only" GP (inputs ln Re, ln We, ln D0, phi), trained on all 12 textured surfaces. In leave one
surface out testing this model scored RMSE 0.0397 against 0.0400 for the geometry baseline (paired
bootstrap interval includes zero) and read the smooth plate REF-H as phi = 1.000.

Validity: the reader is calibrated at 43x (about 1.16 um per pixel, about 3 mm field of view). At 100x
and 350x it under-reads phi on textured surfaces, so other pixel sizes are refused unless --force.
"""
import argparse, json, pickle
import numpy as np
import pandas as pd
from .data import ROOT, load, features
from . import extensions as E

MODEL = E.OUT / 'image_only_gp.pkl'
CAL_UM = E.UM_PER_PX_AT_MAG1 / 43


def trained():
    if MODEL.exists():
        return pickle.loads(MODEL.read_bytes())
    s = json.loads((E.OUT / 'summary.json').read_text())
    thr = s['image_reader']['threshold_all_textured']
    t, _ = load()
    t, cols = E.design(t, E.CANDIDATES['image_only'], thr, {})
    t.attrs['cols'] = cols
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(t[cols])
    k = ConstantKernel(1.) * Matern(np.ones(len(cols)), nu=1.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(.001, (1e-6, .1))
    gp = GaussianProcessRegressor(k, normalize_y=True, optimizer=E.optimizer, random_state=23).fit(sc.transform(t[cols]), np.log(t.beta.values))
    phis = t.groupby('surface').phi.first()
    bundle = dict(gp=gp, scaler=sc, cols=cols, threshold=thr, phi_range=[float(phis.min()), float(phis.max())],
                  support={c: [float(t[c].min()), float(t[c].max())] for c in ['Re', 'We', 'D', 'V']},
                  loso=s['candidates']['image_only']['loso'])
    MODEL.write_bytes(pickle.dumps(bundle))
    return bundle


def read_phi(path, um_per_px=None, footer_y=E.FOOTER_Y, force=False):
    um = um_per_px or E.pixel_size(path)
    if not force and abs(um / CAL_UM - 1) > .25:
        raise SystemExit(f'Pixel size {um:.3f} um/px is outside the calibrated range ({CAL_UM:.3f} um/px, i.e. about 43x). '
                         'Use an image near 43x, or pass --force to read it anyway (phi will be biased low).')
    b = trained()
    return float((E.roughness_map(path, um, footer_y) <= b['threshold']).mean()), um


def predict(phi, D_mm, V, rho, mu, sigma):
    b = trained()
    d = features(pd.DataFrame([dict(D=D_mm / 1000, V=V, rho=rho, mu=mu, sigma=sigma)]))
    d['phi'] = phi
    m, s = b['gp'].predict(b['scaler'].transform(d[b['cols']]), return_std=True)
    reasons = [f'{k} outside measured range' for k, (lo, hi) in b['support'].items() if not lo <= float(d[k].iloc[0]) <= hi]
    lo, hi = b['phi_range']
    if not lo <= phi <= hi:
        reasons.append(f'phi {phi:.3f} outside the training surfaces ({lo:.2f} to {hi:.2f})')
    if phi > .95:
        reasons.append('Smooth surface: wettability is not modelled; on REF-H only about 62% of impacts fell inside the 90% interval')
    return dict(beta_max=float(np.exp(m[0])), interval_90=[float(np.exp(m[0] - E.Z90 * s[0])), float(np.exp(m[0] + E.Z90 * s[0]))],
                phi_from_image=phi, Re=float(d.Re.iloc[0]), We=float(d.We.iloc[0]),
                status='within_measured_support' if not reasons else 'extrapolation', reasons=reasons,
                model='image_only GP (ln Re, ln We, ln D0, phi from image)',
                note=f"Leave one surface out: RMSE {b['loso']['rmse']:.4f}, 90% interval coverage {b['loso']['coverage']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--image', required=True)
    ap.add_argument('--um-per-px', type=float, help='pixel size; read from JEOL metadata if omitted')
    ap.add_argument('--footer-y', type=int, default=E.FOOTER_Y, help='first row of the annotation bar (0 = none)')
    ap.add_argument('--force', action='store_true')
    for k, v in [('D_mm', 2.5), ('V', 1.5), ('rho', 997.2375), ('mu', .000943923), ('sigma', .07246)]:
        ap.add_argument('--' + k, type=float, default=v)
    a = ap.parse_args()
    phi, um = read_phi(a.image, a.um_per_px, a.footer_y or None, a.force)
    out = predict(phi, a.D_mm, a.V, a.rho, a.mu, a.sigma)
    out['um_per_px'] = um
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
