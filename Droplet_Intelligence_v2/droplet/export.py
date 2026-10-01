"""Fit deployment candidates and export portable numerical bundles (no pickle)."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,sys,platform,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import ROOT,load,FEATURES,audit
from .models import Model,CANDIDATES,LABELS
from .nested import order_quantile

def portable(model):
    g=model.gp;k=g.kernel_
    if model.kind=='structured':spec={'type':'structured','length_scale':np.asarray(k.k1.length_scale).tolist(),'amplitudes':np.asarray(k.k1.amplitudes).tolist()}
    else:spec={'type':'matern','nu':2.5 if model.kind=='matern52' else 1.5,'length_scale':np.asarray(k.k1.k2.length_scale).tolist(),'amplitude':float(k.k1.k1.constant_value)}
    return {'kind':model.kind,'label':LABELS[model.kind],'features':FEATURES,'kernel':spec,'A':model.A,
      'X':g.X_train_.tolist(),'alpha':g.alpha_.tolist(),'xmean':model.scaler.mean_.tolist(),'xscale':model.scaler.scale_.tolist(),
      'ymean':float(g._y_train_mean),'yscale':float(g._y_train_std),'noise':float(k.k2.noise_level),
      'n':len(g.X_train_),'L_format':'lower triangular row-major float64 little-endian','warnings':model.warnings}

def run():
    t,r=load();out=ROOT/'results';nested=json.loads((out/'nested.json').read_text());pred=pd.read_csv(out/'loso_predictions.csv');a=audit()
    modeldir=ROOT/'models';modeldir.mkdir(exist_ok=True);ref_rows=[];meta={}
    with threadpool_limits(limits=1):
        for kind in CANDIDATES:
            start=time.time();model=Model(kind).fit(t);m=portable(model);subset=pred[pred.model==kind]
            m['q']=order_quantile(np.abs(np.log(subset.beta)-subset.mu_log)/np.maximum(subset.sd_log,1e-8))
            m['abs_width_log']=order_quantile(np.abs(np.log(subset.beta)-subset.mu_log))
            mu,sd=model.predict(r);p=np.exp(mu);lo=np.exp(mu-m['q']*sd);hi=np.exp(mu+m['q']*sd)
            m['refh']={'rmse':float(np.sqrt(np.mean((r.beta-p)**2))),'coverage':float(((r.beta>=lo)&(r.beta<=hi)).mean()),'mean_width':float(np.mean(hi-lo))}
            for i,row in r.iterrows():ref_rows.append({'model':kind,'row_id':i,'surface':row.surface,'condition':row.condition,'beta':row.beta,'prediction':p[i],'mu_log':mu[i],'sd_log':sd[i],'lower':lo[i],'upper':hi[i]})
            L=model.gp.L_;packed=L[np.tril_indices(len(L))].astype('<f8');packed.tofile(modeldir/f'{kind}.L.bin')
            m['L_sha256']=hashlib.sha256((modeldir/f'{kind}.L.bin').read_bytes()).hexdigest()
            (modeldir/f'{kind}.json').write_text(json.dumps(m,separators=(',',':')));meta[kind]={'label':m['label'],'refh':m['refh']}
            print(f'Exported {kind} in {time.time()-start:.1f}s',flush=True)
    # Shrinkage covariance over standardized supplied features, fitted on training inputs.
    X=t[FEATURES].values;xm=X.mean(0);xs=X.std(0);Z=(X-xm)/xs;cov=np.cov(Z,rowvar=False);cov=.95*cov+.05*np.eye(5);prec=np.linalg.inv(cov)
    distances=np.sqrt(np.einsum('ij,jk,ik->i',Z,prec,Z))
    surfaces={s:{'spacing':float(d.spacing.iloc[0]),'depth':float(d.depth.iloc[0]),'phi':float(d.phi.iloc[0]),'texvol':float(d.texvol.iloc[0])} for s,d in t.groupby('surface')}
    surfaces['REF-H']={'spacing':0.,'depth':0.,'phi':1.,'texvol':0.}
    gly=pd.cut(t.mu,[0,.0012,.003,.02,.08,1],labels=[0,20,60,78,91]).astype(int)
    fluids={str(int(g)):{'rho':float(d.rho.median()),'sigma':float(d.sigma.median()),'mu':float(d.mu.median())} for g,d in t.assign(gly=gly).groupby('gly')}
    # Representative presets are medians, not exact properties of every observation.
    release={'version':'2.0.0','selected':nested['deployment_model'],'candidates':meta,'features':FEATURES,'surfaces':surfaces,'fluids':fluids,
      'fluid_preset_note':'Median measured properties within each glycerol family; individual experimental rows retain their original properties.',
      'support':a['feature_support'],'ood':{'mean':xm.tolist(),'scale':xs.tolist(),'precision':prec.tolist(),'sorted_distances':np.sort(distances).tolist()},
      'interval_method':'Empirical cross-validated log-residual calibration; nominal 90% target, no guaranteed shifted-surface coverage.',
      'data_hashes':a['files_sha256'],'python':sys.version.split()[0],'platform':platform.platform()}
    (modeldir/'release.json').write_text(json.dumps(release,indent=2));pd.DataFrame(ref_rows).to_csv(out/'refh_predictions.csv',index=False)
    data=pd.concat([t.assign(split='textured'),r.assign(split='REF-H')]);data['glycerol']=pd.cut(data.mu,[0,.0012,.003,.02,.08,1],labels=[0,20,60,78,91]).astype(int)
    data.to_csv(ROOT/'data/experiments.csv',index=False)
    rows=data[['surface','glycerol','D','V','Re','We','beta','split']].to_dict(orient='records')
    (out/'experiments.json').write_text(json.dumps(rows,separators=(',',':')))
    print('Selected:',release['selected'],flush=True)

if __name__=='__main__':run()
