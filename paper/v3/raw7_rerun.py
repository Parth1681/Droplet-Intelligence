"""Re-run the Phase 2 dimensional-input GP (D, V, rho, sigma, mu, spacing, depth) on the v2.1 harness,
so it can be compared with the baseline on identical rows with the paired surface bootstrap."""
import os, sys, json
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../Droplet_Intelligence_v2'))
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from droplet.data import load
from droplet.models import optimizer
from droplet.benchmark import bootstrap

RAW = ['D', 'V', 'rho', 'sigma', 'mu', 'spacing', 'depth']

def fit(task):
    held = task
    with threadpool_limits(limits=1):
        t, r = load(); r = r.assign(spacing=0.0, depth=0.0)
        tr, te = (t, r) if held == 'REF-H' else (t[t.surface != held], t[t.surface == held])
        sc = StandardScaler().fit(tr[RAW])
        k = ConstantKernel(1.) * Matern(np.ones(7), nu=1.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(.001, (1e-6, .1))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            gp = GaussianProcessRegressor(k, normalize_y=True, optimizer=optimizer, random_state=23).fit(sc.transform(tr[RAW]), np.log(tr.beta.values))
        mu, sd = gp.predict(sc.transform(te[RAW]), return_std=True)
    return held, te.row_id.tolist(), mu.tolist(), sd.tolist()

if __name__ == '__main__':
    t, r = load(); S = sorted(t.surface.unique())
    with ProcessPoolExecutor(4) as p: res = list(p.map(fit, S + ['REF-H']))
    mu = np.zeros(len(t)); sd = np.zeros(len(t))
    for h, ids, m, s in res:
        if h != 'REF-H': mu[ids] = m; sd[ids] = s
    base = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../Droplet_Intelligence_v2/results/loso_predictions.csv'))
    bp = base[base.model == 'baseline'].sort_values('row_id').prediction.values
    p = np.exp(mu); y = t.beta.values
    h, ids, m, s = next(x for x in res if x[0] == 'REF-H'); pr = np.exp(m); yr = r.beta.values[ids]
    out = dict(model='GP, dimensional inputs D,V,rho,sigma,mu,spacing,depth (REF-H spacing=depth=0)',
               loso_rmse=float(np.sqrt(np.mean((p - y) ** 2))), loso_mae=float(np.mean(np.abs(p - y))),
               loso_r2=float(1 - np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2)),
               loso_cov90=float(np.mean(np.abs(np.log(y) - mu) <= 1.6448536269514722 * sd)),
               delta_vs_baseline_ci95=bootstrap(y, p, bp, t.surface.values),
               refh_rmse=float(np.sqrt(np.mean((pr - yr) ** 2))),
               per_surface={s_: float(np.sqrt(np.mean((p[t.surface.values == s_] - y[t.surface.values == s_]) ** 2))) for s_ in S})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), 'raw7_rerun.json'), 'w'), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_surface'}, indent=2))
