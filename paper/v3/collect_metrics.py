"""Collect every verified metric (RMSE, MAE, R2 on beta_max) from the saved prediction files.
Writes paper/v3/verified_metrics.json. Nothing here is retrained except where a file says so."""
import json, glob, os, re
import numpy as np, pandas as pd
R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(R + '/../..')
A = ROOT + '/archive/v1_droplet/results'; V2 = ROOT + '/Droplet_Intelligence_v2/results'
NA = 'Not available from the supplied evidence'

def m(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float); e = p - y
    return dict(rmse=float(np.sqrt(np.mean(e**2))), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e**2) / np.sum((y - y.mean())**2)), bias=float(e.mean()), n=int(len(y)))

lo = pd.read_csv(V2 + '/loso_predictions.csv'); rf = pd.read_csv(V2 + '/refh_predictions.csv')
b = lo[lo.model == 'baseline'].sort_values('row_id'); y = b.beta.values; groups = b.surface.values
yr = rf[rf.model == 'baseline'].sort_values('row_id').beta.values
out = dict(source_note='beta_max scale; LOSO = leave one textured surface out (12 folds); ID = GroupKFold(5) by condition; REF-H = smooth plate never used in training', models={})

# v1 final models (per-row predictions saved)
names = dict(GPR='GP Matern 3/2 ARD (final)', XGB='XGBoost (same 5 inputs)', M1='XGBoost, raw 7 inputs',
             M2='XGBoost, Re/We/Oh + spacing/depth', M3='XGBoost, hybrid raw + Re/We/Oh',
             M5='Power-law backbone + XGBoost residual', M6='Laan backbone + GP residual')
for k, n in names.items():
    d = np.load(f'{A}/final/pred_{k}.npz')
    r = dict(label=n, source=f'archive/v1_droplet/results/final/pred_{k}.npz', id=m(y, d['id']), loso=m(y, d['loso']), refh=m(yr, d['ref']))
    if 'loso_sd' in d.files and np.any(d['loso_sd'] > 0) and k in ('GPR', 'M6'):
        r['has_uncertainty'] = True
    out['models']['v1_' + k] = r

# v2 candidates (LOSO + REF-H per row)
for k in lo.model.unique():
    s = lo[lo.model == k].sort_values('row_id'); t = rf[rf.model == k].sort_values('row_id')
    r = dict(label=k, source='Droplet_Intelligence_v2/results/{loso,refh}_predictions.csv', loso=m(s.beta, s.prediction), id=NA)
    r['refh'] = m(t.beta, t.prediction) if len(t) else NA
    if len(t) and 'lower' in t: r['refh']['coverage90'] = float(((t.beta >= t.lower) & (t.beta <= t.upper)).mean())
    out['models']['v2_' + k] = r
bj = json.load(open(V2 + '/benchmark.json'))
for e in bj['metrics']:
    out['models']['v2_' + e['model']].update(loso_coverage90=e['nominal_coverage'], delta_ci95=e['delta_vs_baseline_ci95'], worst=e['worst_rmse'], per_surface=e['per_surface'])

# extensions
ex = pd.read_csv(V2 + '/extensions/loso_predictions.csv'); es = json.load(open(V2 + '/extensions/summary.json'))
for k in ex.model.unique():
    s = ex[ex.model == k].sort_values('row_id'); c = es['candidates'].get(k, {})
    out['models']['ext_' + k] = dict(refh=dict(rmse=c.get('refh', {}).get('rmse')) if isinstance(c.get('refh'), dict) else NA, label=k, source='Droplet_Intelligence_v2/results/extensions/', loso=m(s.beta, s.prediction), id=NA,
                                     summary=c)

# zoo: LOSO RMSE only (table), REF-H from saved per-row predictions
z = pd.read_csv(A + '/zoo_table.csv')
for _, row in z.iterrows():
    key = re.sub('_+', '_', ''.join(ch if ch.isalnum() else '_' for ch in row.Model.lower()))
    p = A + f'/zoo/{key}_ref.npy'
    if not os.path.exists(p):
        c = [g for g in glob.glob(A + '/zoo/*_ref.npy') if key.startswith(os.path.basename(g)[:-8])]
        p = c[0] if len(c) == 1 else p
    refh = m(yr, np.load(p)) if os.path.exists(p) else NA
    out['models']['zoo_' + key] = dict(label=row.Model, source='archive/v1_droplet/results/zoo_table.csv' + (' + zoo/*_ref.npy' if os.path.exists(p) else ''),
        id=dict(rmse=float(row['ID RMSE']), r2=float(row['ID R2']), mae=NA), loso=dict(rmse=float(row['LOSO RMSE']), mae=NA, r2=NA, worst=row['Worst LOSO']),
        refh=refh, refh_table_rmse=float(row['REF-H RMSE']))

out['models']['raw7_gp'] = dict(label='GP on dimensional inputs (D,V,rho,sigma,mu,spacing,depth)', source='paper/v3/raw7_rerun.json',
                                **json.load(open(R + '/raw7_rerun.json')))
# sanity: archive GPR LOSO equals v2 baseline LOSO
out['checks'] = dict(v1_GPR_vs_v2_baseline_loso_rmse=[out['models']['v1_GPR']['loso']['rmse'], out['models']['v2_baseline']['loso']['rmse']],
                     y_range=[float(y.min()), float(y.max())], refh_range=[float(yr.min()), float(yr.max())])
json.dump(out, open(R + '/verified_metrics.json', 'w'), indent=1, default=str)
for k, v in out['models'].items():
    f = lambda d, q: (f"{d[q]:.4f}" if isinstance(d, dict) and isinstance(d.get(q), float) else '  NA  ')
    print(f"{k:42s} ID {f(v.get('id'),'rmse')}  LOSO {f(v['loso'],'rmse') if 'loso' in v else '':8s} {f(v.get('loso'),'mae')} {f(v.get('loso'),'r2')}  REFH {f(v.get('refh'),'rmse')} {f(v.get('refh'),'mae')} {f(v.get('refh'),'r2')}")
print(out['checks'])
