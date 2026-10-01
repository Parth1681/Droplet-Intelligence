from pathlib import Path as _P; _ROOT = _P(__file__).resolve().parents[1]  # repo root (droplet-intelligence/)
import json,numpy as np,pandas as pd
V=str(_ROOT/'Droplet_Intelligence_v2')
J=lambda f:json.load(open(f))
bm={m['model']:m for m in J(f'{V}/results/benchmark.json')['metrics']}; ab=J('ablation.json'); sem=J(f'{V}/results/sem/summary.json')
nest=J(f'{V}/results/nested.json'); lg=J(f'{V}/results/legacy_gpr_vs_xgb.json'); rel=J(f'{V}/models/release.json')
d=pd.read_csv(f'{V}/data/experiments.csv'); t=d[d.split=='textured']; r=d[d.split=='REF-H'].reset_index(drop=True)
rf=pd.read_csv(f'{V}/results/refh_predictions.csv'); lo=pd.read_csv(f'{V}/results/loso_predictions.csv')
rm=lambda e:float(np.sqrt(np.mean(np.square(e))))
def refh(m):
    x=rf[rf.model==m]; return rm(x.prediction-x.beta), float(((x.beta>=x.lower)&(x.beta<=x.upper)).mean())
def mape(m):
    x=lo[lo.model==m]; return float(((x.prediction-x.beta).abs()/x.beta).mean()*100)
def ps(p): v=list(p.values()); return float(np.mean(v)), float(max(v)), max(p,key=p.get)
rows=[]
for k,lab in [('baseline','Baseline GP, Matérn 3/2 (selected)'),('matern52','GP, Matérn 5/2'),('structured','GP, fluid–surface interaction kernel'),('physics_residual','Laan backbone + residual GP')]:
    m=bm[k]; rr,rc=refh(k); rows.append(dict(key=k,label=lab,loso=m['loso_rmse'],macro=m['macro_rmse'],worst=m['worst_rmse'],worst_s=max(m['per_surface'],key=m['per_surface'].get),refh=rr,refh_cov=rc,loso_cov=m['nominal_coverage'],mape=mape(k)))
mc,w,ws=ps(sem['per_surface']); rows.append(dict(key='sem',label='SEM CNN descriptors + GP (experimental)',loso=sem['loso_rmse'],macro=sem['macro_rmse'],worst=w,worst_s=ws,refh=0.064789,refh_cov=0.592,loso_cov=sem['nominal_gp_coverage']))
mc,w,ws=ps(ab['blind3']['per_surface']); rows.append(dict(key='blind',label='GP without surface descriptors (ablation)',loso=ab['blind3']['loso_rmse'],macro=mc,worst=w,worst_s=ws,refh=ab['blind3']['refh_rmse']))
mc,w,ws=ps(lg['XGB']['per_surface']); rows.append(dict(key='xgb',label='XGBoost, same five inputs',loso=lg['XGB']['LOSO'],macro=mc,worst=w,worst_s=ws,refh=lg['XGB']['REFH']))
mc,w,ws=ps(ab['laan']['per_surface']); rows.append(dict(key='laan',label='Laan scaling law only (A fitted)',loso=ab['laan']['loso_rmse'],macro=mc,worst=w,worst_s=ws,refh=ab['laan']['refh_rmse'],mape=ab['laan']['loso_mape']))
N=dict(rows=rows,forest=J('forest_rows.json'),by_fluid=J('by_fluid.json'),
 per_surface_cov={s:float(v) for s,v in lo[lo.model=='baseline'].assign(c=lambda x:(x.beta>=np.exp(x.mu_log-1.6448536*x.sd_log))&(x.beta<=np.exp(x.mu_log+1.6448536*x.sd_log))).groupby('surface').c.mean().items()},
 nested=dict(rmse=nest['nested_rmse'],ci=nest['delta_vs_baseline_ci95'],cov=nest['coverage'],width=nest['mean_width'],q=nest['deployment_q'],fold_cov=[min(f['coverage'] for f in nest['folds']),max(f['coverage'] for f in nest['folds'])],
   selected={f['outer']:f['selected'] for f in nest['folds']}),
 legacy=dict(gpr_id=lg['GPR']['ID'],xgb_id=lg['XGB']['ID'],id_ci=lg['ID_GPR_minus_XGB_CI'],within05=lg.get('LOSO_within_0.05'),wins=lg.get('GPR_wins_surfaces'),refh_ci=lg['REFH_GPR_minus_XGB_CI']),
 laan=dict(A=ab['laan']['A_all'],A_range=ab['laan']['A_range'],textbook=ab['laan']['A_textbook_1.24_all_rmse']),
 physA=J(f'{V}/models/physics_residual.json')['A'],
 ls=dict(zip(J(f'{V}/models/baseline.json')['features'],J(f'{V}/models/baseline.json')['kernel']['length_scale'])),
 noise=J(f'{V}/models/baseline.json')['noise'],
 data=dict(n_t=len(t),n_r=len(r),cond_t=int(t.condition.nunique()),cond_r=int(r.condition.nunique()),beta=[float(t.beta.min()),float(t.beta.max())],beta_r=[float(r.beta.min()),float(r.beta.max())],
   Re=[float(d.Re.min()),float(d.Re.max())],We=[float(d.We.min()),float(d.We.max())],D=[float(d.D.min()*1e3),float(d.D.max()*1e3)],V=[float(d.V.min()),float(d.V.max())],Oh=[float(d.Oh.min()),float(d.Oh.max())]),
 surfaces=d.groupby('surface')[['spacing','depth','phi','texvol']].first().reset_index().to_dict('records'),
 fluids=d.groupby('glycerol').agg(rho=('rho','mean'),mu=('mu','mean'),sigma=('sigma','mean'),n_t=('split',lambda x:(x=='textured').sum()),n_r=('split',lambda x:(x=='REF-H').sum()),mulo=('mu','min'),muhi=('mu','max')).reset_index().to_dict('records'),  # handoff: means + viscosity range, as in paper Table 1
 warnings_outer='12/48',warnings_inner='48/144',release=rel.get('version'),python=rel.get('python'),sklearn=rel.get('packages',rel).get('scikit-learn') if isinstance(rel.get('packages'),dict) else None,
 sem_nominal_cov=sem['nominal_gp_coverage'],sem_wins=sum(sem['per_surface'][s]<bm['baseline']['per_surface'][s] for s in sem['per_surface']),
 blind_wins=sum(ab['blind3']['per_surface'][s]>bm['baseline']['per_surface'][s] for s in sem['per_surface']))
# Added for the handoff: these two keys were merged into numbers.json by hand on 24 Sept; now regenerated here.
# 13-learner screen: computed by archive/v1_droplet/src/models_extra.py, table written by archive/v1_droplet/poster/make_figures.py.
Z=pd.read_csv(str(_ROOT/'archive/v1_droplet/results/zoo_table.csv'))
N['zoo']=[dict(name=r['name'],id=float(r['ID RMSE']),loso=float(r['LOSO RMSE']),worst=r['Worst LOSO'],refh=float(r['REF-H RMSE'])) for _,r in Z.iterrows()]
_mu=d.groupby('glycerol').mu.mean(); N['mu_ratio']=float(_mu.max()/_mu.min())
json.dump(N,open('numbers.json','w'),indent=1,default=float)
for x in rows: print({k:(round(v,5) if isinstance(v,float) else v) for k,v in x.items()})
print(N['legacy'],N['laan'],N['physA'],N['nested']['fold_cov'],N['sem_wins'],N['blind_wins'],N['release'],N['python'],N['sklearn'])
print(json.dumps(rel)[:600])
