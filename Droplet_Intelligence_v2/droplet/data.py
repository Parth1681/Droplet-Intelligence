from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['logRe', 'logWe', 'logD', 'phi', 'texvol']
TEX = ['surface','D','V','rho','sigma','mu','spacing','depth','Re','We','tc','eta','beta','vlam']
REF = ['surface','D','V','rho','sigma','mu','Re','We','beta','vlam']

def features(d):
    d = d.copy()
    if 'spacing' not in d: d['spacing'] = 0.
    if 'depth' not in d: d['depth'] = 0.
    d['Re'] = d.rho*d.V*d.D/d.mu
    d['We'] = d.rho*d.V**2*d.D/d.sigma
    d['Oh'] = np.sqrt(d.We)/d.Re
    d['P'] = d.We*d.Re**(-.4)
    width = np.where(d.depth >= 15,45.,30.)
    if 'phi' not in d:
        d['phi'] = np.where((d.spacing>0)&(d.depth>0), (np.maximum(d.spacing-width,0)/np.maximum(d.spacing,1e-30))**2,1.)
    if 'texvol' not in d: d['texvol'] = (1-d.phi)*d.depth
    d['logRe'],d['logWe'],d['logD'] = np.log(d.Re),np.log(d.We),np.log(d.D*1000)
    return d

def condition_ids(d):
    ids = pd.Series(index=d.index,dtype=object)
    gly = pd.cut(d.mu,[0,.0012,.003,.02,.08,1],labels=[0,20,60,78,91]).astype(int)
    for (surface,fluid), sub in d.assign(gly=gly).groupby(['surface','gly']):
        v=sub.V.sort_values(); c=(v.diff()>.10).cumsum()
        ids.loc[v.index]=[f'{surface}|{fluid}|{x}' for x in c]
    return ids

def load():
    frames=[]
    for name,cols in [('01 - Rebound Data.csv',TEX),('02 - Rebound Data - REF-H.csv',REF)]:
        d=pd.read_csv(ROOT/'data/raw'/name,encoding='latin1');d.columns=cols
        d=features(d);d['condition']=condition_ids(d);d['row_id']=np.arange(len(d))
        frames.append(d)
    return frames

def audit():
    t,r=load(); raw=pd.read_csv(ROOT/'data/raw/01 - Rebound Data.csv',encoding='latin1')
    return {'textured_rows':len(t),'reference_rows':len(r),'surfaces':sorted(t.surface.unique()),
      'textured_conditions':int(t.condition.nunique()),'reference_conditions':int(r.condition.nunique()),
      'target_range':[float(t.beta.min()),float(t.beta.max())],
      'missing_cells':int(t.isna().sum().sum()+r.isna().sum().sum()),
      'duplicate_rows':int(raw.duplicated().sum()),
      're_relative_recompute_error':float(np.max(np.abs(t.Re/raw.iloc[:,8]-1))),
      'we_relative_recompute_error':float(np.max(np.abs(t.We/raw.iloc[:,9]-1))),
      'feature_support':{k:[float(t[k].min()),float(t[k].max())] for k in FEATURES+['D','V','Re','We','mu','rho','sigma']},
      'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data/raw').glob('*.csv'))},
      'descriptor_note':'Existing fixed geometry proxy: width=45 um at depth>=15 um, else 30 um. No new SEM extraction. Conditional on these supplied descriptors.',
      'contact_angles':'Water advancing angles on 12 surfaces; glycerol-mixture advancing angles on 6 surfaces; REF-H measured advancing angle not supplied in CSV files. Excluded from predictive candidates.'}
