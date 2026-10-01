"""Nested surface model selection and empirical cross-validated calibration.

For each outer surface: partition remaining surfaces into 3 inner folds. Each
candidate is fitted only on inner-training surfaces. Select with inner macro
RMSE, then use existing independent outer fit. No outer labels enter selection
or calibration. This is CV calibration, not a split-conformal coverage theorem.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,time,argparse,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import ROOT,load
from .models import Model,CANDIDATES
from .benchmark import bootstrap

TOLERANCE=.001  # predefined absolute macro-RMSE simplicity tolerance

def choose(scores):
    best=min(scores.values())
    return next(k for k in CANDIDATES if scores[k]<=best+TOLERANCE)

def order_quantile(values,level=.9):
    a=np.sort(np.asarray(values));rank=int(np.ceil((len(a)+1)*level))
    return float(a[rank-1]) if rank<=len(a) else float('inf')

def inner_splits(outer,surfaces):
    remaining=np.array([s for s in surfaces if s!=outer]);np.random.default_rng(23).shuffle(remaining)
    return [list(x) for x in np.array_split(remaining,3)]

def fit_inner(task):
    kind,outer,fold,held=task
    with threadpool_limits(limits=1):
        t,_=load();test=t[t.surface.isin(held)];train=t[~t.surface.isin(held+[outer])]
        start=time.time();model=Model(kind).fit(train);mu,sd=model.predict(test)
        assert outer not in model.train_surfaces and not set(held)&set(model.train_surfaces)
        return {'model':kind,'outer':outer,'inner_fold':fold,'held':held,'train_surfaces':model.train_surfaces,
          'row_id':test.row_id.tolist(),'mu':mu.tolist(),'sd':sd.tolist(),'seconds':time.time()-start,'warnings':model.warnings}

def run(workers):
    t,_=load();surfaces=sorted(t.surface.unique());out=ROOT/'results';cache=out/'inner_folds';cache.mkdir(exist_ok=True)
    training_hash=hashlib.sha256(b''.join((ROOT/'droplet'/f).read_bytes() for f in ['data.py','models.py','nested.py'])+b''.join(p.read_bytes() for p in sorted((ROOT/'data/raw').glob('*.csv')))).hexdigest()
    pending=[];results=[]
    for outer in surfaces:
        for fold,held in enumerate(inner_splits(outer,surfaces)):
            for kind in CANDIDATES:
                task=(kind,outer,fold,held);p=cache/f'{kind}_{outer}_{fold}.json'
                if p.exists() and json.loads(p.read_text()).get('training_hash')==training_hash: results.append(json.loads(p.read_text()))
                else:pending.append(task)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs=[pool.submit(fit_inner,x) for x in pending]
        for future in as_completed(futs):
            z=future.result();z['training_hash']=training_hash;results.append(z)
            (cache/f"{z['model']}_{z['outer']}_{z['inner_fold']}.json").write_text(json.dumps(z))
            print(f"inner {len(results)}/144 {z['model']} {z['outer']} fold {z['inner_fold']} {z['seconds']:.1f}s",flush=True)
    outer_rows=pd.read_csv(out/'loso_predictions.csv');fold_report=[];pred_rows=[]
    for outer in surfaces:
        scores={};prepared={}
        for kind in CANDIDATES:
            rows=[]
            for z in results:
                if z['model']==kind and z['outer']==outer:
                    d=t.iloc[z['row_id']].copy();d['mu']=z['mu'];d['sd']=z['sd'];rows.append(d)
            d=pd.concat(rows);prepared[kind]=d
            per=[np.sqrt(np.mean((g.beta-np.exp(g.mu))**2)) for _,g in d.groupby('surface')]
            scores[kind]=float(np.mean(per))
        selected=choose(scores);cal=prepared[selected]
        # Equal-row scores approximate this dataset's near-balanced surfaces.
        # Report empirical intervals; no finite-sample grouped guarantee asserted.
        z=np.abs(np.log(cal.beta)-cal.mu)/np.maximum(cal.sd,1e-8)
        q=order_quantile(z);absolute=order_quantile(np.abs(np.log(cal.beta)-cal.mu))
        test=outer_rows[(outer_rows.surface==outer)&(outer_rows.model==selected)].copy()
        test['selected']=selected;test['q']=q;test['lower']=np.exp(test.mu_log-q*test.sd_log);test['upper']=np.exp(test.mu_log+q*test.sd_log)
        test['wide_lower']=np.exp(test.mu_log-np.maximum(q*test.sd_log,absolute));test['wide_upper']=np.exp(test.mu_log+np.maximum(q*test.sd_log,absolute))
        pred_rows.append(test)
        fold_report.append({'outer':outer,'selected':selected,'inner_macro_rmse':scores,'q':q,'calibration_rows':len(cal),
          'calibration_surface_count':int(cal.surface.nunique()),'coverage':float(((test.beta>=test.lower)&(test.beta<=test.upper)).mean()),
          'rmse':float(np.sqrt(np.mean((test.beta-test.prediction)**2))),'interval_width':float((test.upper-test.lower).mean())})
    p=pd.concat(pred_rows).sort_values('row_id');base=outer_rows[outer_rows.model=='baseline'].sort_values('row_id')
    full_scores={k:float(np.mean([np.sqrt(np.mean((g.beta-g.prediction)**2)) for _,g in outer_rows[outer_rows.model==k].groupby('surface')])) for k in CANDIDATES}
    deployment=choose(full_scores)
    d=outer_rows[outer_rows.model==deployment]
    q=order_quantile(np.abs(np.log(d.beta)-d.mu_log)/np.maximum(d.sd_log,1e-8));absolute=order_quantile(np.abs(np.log(d.beta)-d.mu_log))
    summary={'protocol':'12 outer surfaces; three inner surface folds; four fixed GP candidates; all fitting, scaling, selection and empirical CV calibration exclude the outer surface.',
      'calibration_method':'Nested cross-validated normalized log residuals. Empirical 90% target; no distribution-free guarantee for unseen surfaces.',
      'selection_rule':'First candidate in baseline, matern52, structured, physics_residual order within 0.001 macro RMSE of inner best.',
      'nested_rmse':float(np.sqrt(np.mean((p.beta-p.prediction)**2))),'baseline_rmse':float(np.sqrt(np.mean((base.beta-base.prediction)**2))),
      'delta_vs_baseline_ci95':bootstrap(p.beta.values,p.prediction.values,base.prediction.values,p.surface.values),
      'coverage':float(((p.beta>=p.lower)&(p.beta<=p.upper)).mean()),'mean_width':float((p.upper-p.lower).mean()),
      'folds':fold_report,'deployment_model':deployment,'deployment_macro_scores':full_scores,'deployment_q':q,'deployment_absolute_log_width':absolute,
      'limitation':'Conditional on supplied geometry proxies and fixed candidate set. REF-H is a previously inspected stress test, not fresh blind data. CV calibration does not imply per-surface guaranteed coverage.'}
    p.to_csv(out/'nested_predictions.csv',index=False);(out/'nested.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);run(p.parse_args().workers)
