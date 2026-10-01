"""Export the image_only GP (step 1 extension) as a portable bundle for the browser.

    python -m droplet.export_image_model   ->  models/image_only.json, models/image_only.L.bin
"""
import hashlib, json
import numpy as np
import pandas as pd
from .data import ROOT
from .nested import order_quantile
from . import extensions as E
from .predict_image import trained


def run():
    b = trained(); g = b['gp']; k = g.kernel_
    pred = pd.read_csv(E.OUT / 'loso_predictions.csv'); sub = pred[pred.model == 'image_only']
    q = order_quantile(np.abs(np.log(sub.beta) - sub.mu_log) / np.maximum(sub.sd_log, 1e-8))
    cov_q = float(np.mean(np.abs(np.log(sub.beta) - sub.mu_log) <= q * sub.sd_log))
    s = json.loads((E.OUT / 'summary.json').read_text())
    m = dict(kind='image_only', label='Image GP (phi read from SEM)', features=b['cols'],
             kernel=dict(type='matern', nu=1.5, length_scale=np.asarray(k.k1.k2.length_scale).tolist(), amplitude=float(k.k1.k1.constant_value)),
             X=g.X_train_.tolist(), alpha=g.alpha_.tolist(), xmean=b['scaler'].mean_.tolist(), xscale=b['scaler'].scale_.tolist(),
             ymean=float(g._y_train_mean), yscale=float(g._y_train_std), noise=float(k.k2.noise_level), n=len(g.X_train_),
             q=q, loso_coverage_with_q=cov_q, loso=s['candidates']['image_only']['loso'] | {'per_surface': None},
             refh=s['candidates']['image_only']['refh'] | {'per_surface': None},
             reader=dict(threshold=b['threshold'], window_um=E.WINDOW_UM, um_per_px_43x=E.UM_PER_PX_AT_MAG1 / 43,
                         um_per_px_at_mag1=E.UM_PER_PX_AT_MAG1, footer_y=E.FOOTER_Y, gaussian_sigma_px=1, tolerance=.25),
             phi_range=b['phi_range'], support=b['support'],
             L_format='lower triangular row-major float64 little-endian')
    L = g.L_; packed = L[np.tril_indices(len(L))].astype('<f8')
    (ROOT / 'models/image_only.L.bin').write_bytes(packed.tobytes())
    m['L_sha256'] = hashlib.sha256(packed.tobytes()).hexdigest()
    (ROOT / 'models/image_only.json').write_text(json.dumps(m, separators=(',', ':')))
    print('q', q, 'LOSO coverage with q', cov_q, 'n', m['n'])


if __name__ == '__main__':
    run()
