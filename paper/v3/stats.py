"""Bootstrap CIs, dataset ranges and a measured latency, from saved predictions and the released model.
Writes paper/v3/stats.json."""
import json, os, sys, time
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(R + '/../..')
sys.path.insert(0, ROOT + '/Droplet_Intelligence_v2')
from droplet.data import load
from droplet.benchmark import bootstrap
A = ROOT + '/archive/v1_droplet/results'; V2 = ROOT + '/Droplet_Intelligence_v2/results'
t, r = load(); y = t.beta.values; g = t.surface.values
out = {}

def rmse_ci(p, draws=4000):
    u = np.unique(g); sq = np.array([np.sum((y[g == s] - p[g == s])**2) for s in u]); n = np.array([(g == s).sum() for s in u])
    ix = np.random.default_rng(23).integers(len(u), size=(draws, len(u)))
    return np.quantile(np.sqrt(sq[ix].sum(1) / n[ix].sum(1)), [.025, .975]).tolist()

gp = np.load(A + '/final/pred_GPR.npz')['loso']
out['loso_rmse_ci95_surface_bootstrap'] = {}; out['delta_vs_GP_ci95'] = {}
for k in ['GPR', 'XGB', 'M1', 'M2', 'M3', 'M5', 'M6']:
    p = np.load(f'{A}/final/pred_{k}.npz')['loso']
    out['loso_rmse_ci95_surface_bootstrap'][k] = rmse_ci(p)
    if k != 'GPR': out['delta_vs_GP_ci95'][k] = bootstrap(y, p, gp, g)   # positive = worse than GP
lo = pd.read_csv(V2 + '/loso_predictions.csv')
for k in lo.model.unique():
    out['loso_rmse_ci95_surface_bootstrap']['v2_' + k] = rmse_ci(lo[lo.model == k].sort_values('row_id').prediction.values)

# dataset table
def rng(d, c, f=1): return [float(d[c].min() * f), float(d[c].max() * f)]
fl = pd.cut(t.mu, [0, .0012, .003, .02, .08, 1], labels=[0, 20, 60, 78, 91]).astype(int)
out['dataset'] = dict(
    textured=dict(rows=len(t), surfaces=int(t.surface.nunique()), conditions=int(t.condition.nunique()),
                  D_mm=rng(t, 'D', 1e3), V=rng(t, 'V'), rho=rng(t, 'rho'), mu_mPas=rng(t, 'mu', 1e3), sigma_mNm=rng(t, 'sigma', 1e3),
                  spacing_um=rng(t, 'spacing'), depth_um=rng(t, 'depth'), Re=rng(t, 'Re'), We=rng(t, 'We'), Oh=rng(t, 'Oh'),
                  phi=rng(t, 'phi'), texvol_um=rng(t, 'texvol'), beta=rng(t, 'beta'),
                  rows_per_fluid={int(k): int(v) for k, v in fl.value_counts().sort_index().items()},
                  rows_per_surface={k: int(v) for k, v in t.surface.value_counts().sort_index().items()}),
    refh=dict(rows=len(r), conditions=int(r.condition.nunique()), D_mm=rng(r, 'D', 1e3), V=rng(r, 'V'), Re=rng(r, 'Re'),
              We=rng(r, 'We'), Oh=rng(r, 'Oh'), beta=rng(r, 'beta')))
out['surfaces'] = t.groupby('surface')[['spacing', 'depth', 'phi', 'texvol']].first().round(4).to_dict('index')
# Oh redundancy: ln Oh = 0.5 ln We - ln Re exactly
out['oh_identity_max_abs_error'] = float(np.max(np.abs(np.log(t.Oh) - (.5 * np.log(t.We) - np.log(t.Re)))))

# replicate scatter (std of beta within a condition) and REF-H bias, by fluid
flr = pd.cut(r.mu, [0, .0012, .003, .02, .08, 1], labels=[0, 20, 60, 78, 91]).astype(int)
out['replicate_sd_by_fluid'] = dict(
    textured={int(k): float(v) for k, v in t.assign(fl=fl).groupby('condition').agg(sd=('beta', 'std'), fl=('fl', 'first')).groupby('fl').sd.mean().items()},
    refh={int(k): float(v) for k, v in r.assign(fl=flr).groupby('condition').agg(sd=('beta', 'std'), fl=('fl', 'first')).groupby('fl').sd.mean().items()})
rp = pd.read_csv(V2 + '/refh_predictions.csv'); rp = rp[rp.model == 'baseline'].sort_values('row_id')
out['refh_bias_by_fluid'] = {int(k): float(v) for k, v in pd.Series(rp.prediction.values - rp.beta.values).groupby(flr.values).mean().items()}
out['loso_bias_by_surface'] = {s_: float((gp - y)[g == s_].mean()) for s_ in np.unique(g)}

# latency of the released Python predictor (single query, warm)
from droplet.predict import Predictor
P = Predictor(); q = dict(D_mm=2.5, V=1.5, surface='D200', **P.release['fluids']['0'])
P.predict(q); ts = []
for _ in range(200):
    a = time.perf_counter(); P.predict(q); ts.append(time.perf_counter() - a)
out['latency_python_single_query_ms'] = dict(median=float(np.median(ts) * 1e3), p95=float(np.quantile(ts, .95) * 1e3), n=200,
                                             machine='cloud container CPU, warm model, includes input validation')
json.dump(out, open(R + '/stats.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('surfaces',)}, indent=1))
