"""Fixed-candidate LOSO development benchmark; no global warm starts.

No nested model-family selection is claimed. Candidate comparison is development
evidence; selected-model metrics need a prospective external test.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, json, time, hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import ROOT,load,audit
from .models import Model,CANDIDATES,LABELS

def fingerprint():
    h=hashlib.sha256()
    for p in sorted((ROOT/'droplet').glob('*.py'))+sorted((ROOT/'data/raw').glob('*.csv')):h.update(p.name.encode()+p.read_bytes())
    return h.hexdigest()

def fit_fold(task):
    kind,surface=task
    with threadpool_limits(limits=1):
        t,_=load();train=t[t.surface!=surface];test=t[t.surface==surface]
        start=time.time();model=Model(kind).fit(train);mu,sd=model.predict(test)
        result={'model':kind,'heldout':surface,'train_surfaces':model.train_surfaces,'seconds':time.time()-start,
          'warnings':model.warnings,'kernel':str(model.gp.kernel_),'row_id':test.row_id.tolist(),'mu':mu.tolist(),'sd':sd.tolist()}
    assert surface not in result['train_surfaces']
    return result

def bootstrap(y,pa,pb,groups,draws=4000):
    uniq=np.unique(groups);sq=np.array([[np.sum((y[groups==s]-p[groups==s])**2) for s in uniq] for p in [pa,pb]])
    sizes=np.array([(groups==s).sum() for s in uniq]);ix=np.random.default_rng(23).integers(len(uniq),size=(draws,len(uniq)))
    delta=np.sqrt(sq[0][ix].sum(1)/sizes[ix].sum(1))-np.sqrt(sq[1][ix].sum(1)/sizes[ix].sum(1))
    return np.quantile(delta,[.025,.975]).tolist()

def run(workers=2):
    t,r=load();out=ROOT/'results';out.mkdir(exist_ok=True)
    (out/'data_audit.json').write_text(json.dumps(audit(),indent=2))
    run_id=fingerprint();cache=out/'folds';cache.mkdir(exist_ok=True)
    tasks=[(k,s) for k in CANDIDATES for s in sorted(t.surface.unique())];results=[];pending=[]
    for k,s in tasks:
        p=cache/f'{k}_{s}.json'
        if p.exists() and json.loads(p.read_text()).get('run_id')==run_id:results.append(json.loads(p.read_text()))
        else:pending.append((k,s))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs={pool.submit(fit_fold,task):task for task in pending}
        for future in as_completed(futs):
            result=future.result();result['run_id']=run_id;results.append(result)
            (cache/f"{result['model']}_{result['heldout']}.json").write_text(json.dumps(result,indent=2))
            print(f"{len(results)}/{len(tasks)} {result['model']} {result['heldout']} {result['seconds']:.1f}s",flush=True)
    records=[];metrics=[];predictions={}
    for kind in CANDIDATES:
        mu=np.zeros(len(t));sd=mu.copy()
        for z in results:
            if z['model']==kind:mu[z['row_id']]=z['mu'];sd[z['row_id']]=z['sd']
        p=np.exp(mu);predictions[kind]=p
        for i,row in t.iterrows():records.append({'model':kind,'row_id':int(i),'surface':row.surface,'condition':row.condition,'beta':row.beta,'prediction':p[i],'mu_log':mu[i],'sd_log':sd[i]})
        per={s:float(np.sqrt(np.mean((t.beta.values[t.surface==s]-p[t.surface==s])**2))) for s in sorted(t.surface.unique())}
        metrics.append({'model':kind,'label':LABELS[kind],'loso_rmse':float(np.sqrt(np.mean((t.beta-p)**2))),
          'macro_rmse':float(np.mean(list(per.values()))),'worst_rmse':max(per.values()),'per_surface':per,
          'mae':float(np.mean(np.abs(t.beta-p))),'bias':float(np.mean(p-t.beta)),
          'nominal_coverage':float(np.mean(np.abs(np.log(t.beta)-mu)<=1.6448536269514722*sd))})
    for m in metrics:
        m['delta_vs_baseline_ci95']=bootstrap(t.beta.values,predictions[m['model']],predictions['baseline'],t.surface.values)
        m['improvement_pct']=100*(1-m['loso_rmse']/metrics[0]['loso_rmse'])
    # Select by LOSO only. Conservative promotion needs paired CI excluding zero.
    best=min(metrics,key=lambda x:x['loso_rmse']);selected=best['model'] if best['delta_vs_baseline_ci95'][1]<0 else 'baseline'
    summary={'run_id':run_id,'protocol':'Fixed-candidate LOSO development comparison, independent fold initialization; conditional on supplied geometry descriptors. No nested model-family selection claim.',
      'selected':selected,'lowest_rmse_candidate':best['model'],'selection_rule':'Lowest LOSO RMSE, promoted only if paired surface-bootstrap 95% difference interval excludes zero; otherwise retain baseline.',
      'selection_limitation':'Selection and ranking share LOSO results: descriptive development estimates, not unbiased post-selection performance.',
      'metrics':metrics,'bootstrap_draws':4000,'seed':23}
    pd.DataFrame(records).to_csv(out/'loso_predictions.csv',index=False)
    (out/'benchmark.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);run(p.parse_args().workers)
