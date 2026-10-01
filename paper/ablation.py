from pathlib import Path as _P; _ROOT = _P(__file__).resolve().parents[1]  # repo root (droplet-intelligence/)
import sys, json, warnings, numpy as np, pandas as pd
sys.path.insert(0,str(_ROOT/'Droplet_Intelligence_v2'))
from droplet.models import optimizer
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')
d=pd.read_csv(str(_ROOT/'Droplet_Intelligence_v2/data/experiments.csv'))
t=d[d.split=='textured'].reset_index(drop=True); r=d[d.split=='REF-H'].reset_index(drop=True)
rm=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
def laan(df,A):
    p=np.sqrt(df.We.values*df.Re.values**-.4); return df.Re.values**.2*p/(A+p)
def fitA(df):
    y=np.log(df.beta.values); p=np.sqrt(df.We.values*df.Re.values**-.4)
    return float(least_squares(lambda a:np.log(df.Re.values**.2*p/(a[0]+p))-y,[1.24],bounds=(.01,20)).x[0])
out={}
# Laan only
pl=np.zeros(len(t)); As={}
for s in sorted(t.surface.unique()):
    m=t.surface==s; A=fitA(t[~m]); As[s]=A; pl[m]=laan(t[m],A)
Aall=fitA(t)
out['laan']={'loso_rmse':rm(pl,t.beta.values),'per_surface':{s:rm(pl[t.surface==s],t.beta.values[t.surface==s]) for s in sorted(t.surface.unique())},
 'refh_rmse':rm(laan(r,Aall),r.beta.values),'A_all':Aall,'A_range':[min(As.values()),max(As.values())],
 'loso_mape':float(np.mean(np.abs(pl/t.beta.values-1))*100),'A_textbook_1.24_all_rmse':rm(laan(t,1.24),t.beta.values)}
print('laan',out['laan']['loso_rmse'],out['laan']['refh_rmse'],Aall,flush=True)
F3=['logRe','logWe','logD']
def gp_fit_pred(tr,te,F):
    sc=StandardScaler().fit(tr[F]); k=ConstantKernel(1.)*Matern(np.ones(len(F)),nu=1.5,length_scale_bounds=(1e-2,1e3))+WhiteKernel(.001,(1e-6,.1))
    g=GaussianProcessRegressor(k,normalize_y=True,optimizer=optimizer,random_state=23).fit(sc.transform(tr[F]),np.log(tr.beta.values))
    return np.exp(g.predict(sc.transform(te[F])))
surfs=sorted(t.surface.unique())
res=Parallel(n_jobs=4)(delayed(gp_fit_pred)(t[t.surface!=s],t[t.surface==s],F3) for s in surfs)
pb=np.zeros(len(t))
for s,p in zip(surfs,res): pb[(t.surface==s).values]=p
out['blind3']={'loso_rmse':rm(pb,t.beta.values),'per_surface':{s:rm(pb[t.surface==s],t.beta.values[t.surface==s]) for s in surfs}}
print('blind LOSO',out['blind3']['loso_rmse'],flush=True)
out['blind3']['refh_rmse']=rm(gp_fit_pred(t,r,F3),r.beta.values)
print('blind REFH',out['blind3']['refh_rmse'],flush=True)
# paired surface bootstrap vs baseline
lo=pd.read_csv(str(_ROOT/'Droplet_Intelligence_v2/results/loso_predictions.csv'))
b=lo[lo.model=='baseline'].sort_values('row_id'); assert np.allclose(b.beta.values,t.beta.values)
pbase=b.prediction.values; g=t.surface.values; rng=np.random.default_rng(23); idx={s:np.where(g==s)[0] for s in surfs}
def ci(pa,pb_):
    v=[]
    for _ in range(4000):
        ii=np.concatenate([idx[s] for s in rng.choice(surfs,len(surfs))]); v.append(rm(pa[ii],t.beta.values[ii])-rm(pb_[ii],t.beta.values[ii]))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
out['baseline_minus_blind3_ci']=ci(pbase,pb); out['baseline_minus_laan_ci']=ci(pbase,pl)
np.savez(str(_ROOT/'paper/ablation_preds.npz'),laan=pl,blind3=pb)
json.dump(out,open(str(_ROOT/'paper/ablation.json'),'w'),indent=1); print(json.dumps({k:v for k,v in out.items() if k not in('laan','blind3')}))
