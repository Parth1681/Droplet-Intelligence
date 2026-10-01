"""Numerical inference without sklearn, joblib or untrusted model deserialization."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.linalg import solve_triangular
from .data import ROOT,features,FEATURES
import pandas as pd

def kernel(X,Y,spec):
    ls=np.asarray(spec['length_scale']);d=(X[:,None,:]-Y[None,:,:])/ls
    def m32(r):return (1+np.sqrt(3)*r)*np.exp(-np.sqrt(3)*r)
    if spec['type']=='structured':
        f=m32(np.sqrt((d[:,:,:3]**2).sum(-1)));s=m32(np.sqrt((d[:,:,3:]**2).sum(-1)));a,b,c=spec['amplitudes'];return a*f+b*s+c*f*s
    r=np.sqrt((d*d).sum(-1))
    if spec['nu']==1.5:return spec['amplitude']*m32(r)
    return spec['amplitude']*(1+np.sqrt(5)*r+5*r*r/3)*np.exp(-np.sqrt(5)*r)

def validate_input(payload,release):
    if not isinstance(payload,dict):raise ValueError('Input must be a JSON object')
    allowed={'D_mm','V','rho','sigma','mu','surface','spacing','depth','phi','texvol','model'}
    if set(payload)-allowed:raise ValueError('Unknown fields: '+', '.join(sorted(set(payload)-allowed)))
    d={}
    for key in ['D_mm','V','rho','sigma','mu']:
        v=payload.get(key)
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0:raise ValueError(f'{key} must be a finite positive number')
        d[key]=float(v)
    surface=payload.get('surface','D200')
    if surface in release['surfaces']:
        if any(k in payload for k in ['spacing','depth','phi','texvol']):raise ValueError('Use surface=custom when supplying geometry')
        d.update(release['surfaces'][surface])
    elif surface=='custom':
        for key in ['spacing','depth','phi','texvol']:
            v=payload.get(key)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:raise ValueError(f'Custom {key} must be finite and nonnegative')
            d[key]=float(v)
        if d['phi']>1:raise ValueError('phi must be at most 1')
    else:raise ValueError('Unknown surface')
    d['surface']=surface;d['D']=d.pop('D_mm')*.001
    with np.errstate(all='ignore'):df=features(pd.DataFrame([d]))
    if not np.isfinite(df[FEATURES+['Re','We','Oh','P']].values).all():raise ValueError('Inputs exceed numerical limits')
    return df

class Predictor:
    def __init__(self,folder=ROOT/'models'):
        self.folder=Path(folder);self.release=json.loads((self.folder/'release.json').read_text());self.loaded={}
    def load(self,kind):
        if kind not in self.release['candidates']:raise ValueError('Unknown model')
        if kind not in self.loaded:
            m=json.loads((self.folder/f'{kind}.json').read_text());raw=(self.folder/f'{kind}.L.bin').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=m['L_sha256']:raise ValueError('Model integrity check failed')
            n=m['n'];L=np.zeros((n,n));L[np.tril_indices(n)]=np.frombuffer(raw,dtype='<f8')
            m['L']=L;m['X']=np.array(m['X']);m['alpha']=np.array(m['alpha']);self.loaded[kind]=m
        return self.loaded[kind]
    def predict_frame(self,d,kind):
        m=self.load(kind);X=(d[FEATURES].values-m['xmean'])/m['xscale'];K=kernel(X,m['X'],m['kernel'])
        mu=K@m['alpha']*m['yscale']+m['ymean']
        if m['A'] is not None:
            p=np.sqrt(d.We.values*d.Re.values**(-.4));mu+=np.log(d.Re.values**.2*p/(m['A']+p))
        v=solve_triangular(m['L'],K.T,lower=True,check_finite=False)
        diag=sum(m['kernel']['amplitudes']) if m['kernel']['type']=='structured' else m['kernel']['amplitude']
        sd=np.sqrt(np.maximum(diag+m['noise']-(v*v).sum(0),1e-12))*m['yscale']
        return mu,sd
    def predict(self,payload):
        kind=payload.get('model',self.release['selected']) if isinstance(payload,dict) else None
        d=validate_input(payload,self.release);m=self.load(kind)
        if kind=='sem_gp' and d.iloc[0].surface in m['surface_descriptors']:
            desc=m['surface_descriptors'][d.iloc[0].surface]
            d['phi']=desc['phi'];d['texvol']=desc['texvol']
        mu,sd=self.predict_frame(d,kind);u,s=float(mu[0]),float(sd[0]);row=d.iloc[0]
        reasons=[]
        if kind=='sem_gp':reasons.append('Experimental SEM branch: nominal GP interval excludes image-encoder uncertainty and is not conformal-calibrated')
        for key in ['D','V','Re','We','rho','sigma','mu','phi','texvol']:
            lo,hi=self.release['support'][key]
            if row[key]<lo-1e-12 or row[key]>hi+1e-12:reasons.append(f'{key} outside measured training range')
        if row.surface=='REF-H' or row.phi>=.999:reasons.append('Smooth surface: historical interval undercoverage; wettability is not modeled')
        if row.surface=='custom':reasons.append('Custom surface: supplied descriptors are not independently validated')
        od=self.release['ood'];z=(row[FEATURES].values.astype(float)-od['mean'])/od['scale'];distance=float(np.sqrt(z@np.array(od['precision'])@z));pct=float(np.searchsorted(od['sorted_distances'],distance)/len(od['sorted_distances'])*100)
        if pct>=95:reasons.append('Feature-space anomaly score at or above training 95th percentile')
        band=m['q']*s;wide=max(band,m['abs_width_log'])
        with np.errstate(over='ignore'):
            values=np.exp([u,u-band,u+band,u-wide,u+wide])
        if not np.isfinite(values).all():raise ValueError('Prediction exceeds numerical limits')
        return {'model_version':self.release['version'],'model':kind,'beta_max':float(values[0]),'point_estimator':'lognormal median',
          'interval':{'lower':float(values[1]),'upper':float(values[2]),'nominal_level':.9,'method':'nominal GP, conditional on CNN descriptors' if kind=='sem_gp' else 'empirical CV calibration'},
          'wide_band':{'lower':float(values[3]),'upper':float(values[4]),'method':'empirical sensitivity band, no coverage guarantee'},
          'mu_log':u,'sd_log':s,'physics':{k:float(row[k]) for k in ['Re','We','Oh','P','phi','texvol']},
          'reliability':{'status':('extrapolation' if len(reasons)>1 else 'experimental') if kind=='sem_gp' else ('extrapolation' if reasons else 'within_measured_support'),'reasons':reasons,'ood_percentile':pct}}

if __name__=='__main__':
    import sys
    print(json.dumps(Predictor().predict(json.load(sys.stdin)),indent=2))
