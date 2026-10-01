"""Real SEM pixels -> trained multi-view CNN -> predicted descriptors -> GP.

All preprocessing excludes the annotation footer. A whole surface (all views and
augmentations) is the test unit. Geometry-derived descriptors are weak targets,
not independently measured image labels. No test image enters optimizer updates.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,hashlib,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from threadpoolctl import threadpool_limits
from .data import ROOT,load,FEATURES
from .models import Model
from .benchmark import bootstrap
from .export import portable

MAGS=[43,100,350]
SIDE=128
EPOCHS=250
SEED=23

class SEMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image=nn.Sequential(nn.Conv2d(1,8,5,stride=2,padding=2),nn.SiLU(),
          nn.Conv2d(8,12,3,stride=2,padding=1),nn.SiLU(),
          nn.Conv2d(12,16,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)))
        self.head=nn.Sequential(nn.Linear(3*64,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,2),nn.Sigmoid())
    def forward(self,x):
        b=x.shape[0];z=self.image(x.reshape(-1,1,SIDE,SIDE)).reshape(b,-1)
        return self.head(z)

def prepare():
    folder=ROOT/'data/sem_original';out=ROOT/'data/sem';out.mkdir(exist_ok=True)
    names=sorted({p.stem.rsplit('_',1)[0] for p in folder.glob('*.tif')})
    arrays=[];records=[]
    for name in names:
        views=[]
        for mag in MAGS:
            p=folder/f'{name}_{mag}.tif'
            with Image.open(p) as im:
                if im.size!=(2560,2048):raise ValueError(f'Unexpected SEM dimensions: {p}')
                # Annotation/scale-bar footer is below this fixed crop.
                crop=im.convert('L').crop((0,0,2560,1880))
                a=np.array(crop.resize((SIDE,SIDE),Image.Resampling.LANCZOS),dtype=np.uint8)
                thumb=crop.copy();thumb.thumbnail((720,529));thumb.save(out/f'{name}_{mag}.jpg',quality=88)
            views.append(a)
            records.append({'surface':name,'magnification':mag,'filename':p.name,'original_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'original_shape':[2048,2560],'crop_xyxy':[0,0,2560,1880],'training_shape':[SIDE,SIDE]})
        arrays.append(views)
    np.savez_compressed(out/'training_pixels.npz',images=np.array(arrays),surfaces=np.array(names),magnifications=np.array(MAGS))
    (out/'manifest.json').write_text(json.dumps({'records':records,'independent_surfaces':len(names),'image_count':len(records),'note':'Training arrays and display thumbnails are derived from the same footer-free crop; no microscope text enters the encoder.'},indent=2))
    return names,np.array(arrays)

def get_images():
    z=np.load(ROOT/'data/sem/training_pixels.npz');a=z['images'].astype(np.float32)/255
    means=a.mean((-2,-1),keepdims=True);std=a.std((-2,-1),keepdims=True)
    return z['surfaces'].tolist(),(a-means)/np.maximum(std,1e-3)

def targets(t,names):
    d=t.groupby('surface')[['phi','texvol']].first();return np.array([[d.loc[n,'phi'],d.loc[n,'texvol']/25] for n in names],dtype=np.float32)

def train_encoder(images,labels,seed=SEED):
    torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net=SEMEncoder();opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    X=torch.from_numpy(images);Y=torch.from_numpy(labels);curve=[]
    for epoch in range(EPOCHS):
        net.train();batch=[]
        for i in range(len(X)):
            x=torch.rot90(X[i],int(rng.integers(4)),dims=(-2,-1))
            if rng.random()<.5:x=torch.flip(x,[-1])
            batch.append(x*rng.uniform(.95,1.05)+rng.normal(0,.025))
        aug=torch.stack(batch);pred=net(aug);loss=nn.functional.mse_loss(pred,Y)
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
        if epoch%10==0 or epoch==EPOCHS-1:curve.append({'epoch':epoch+1,'loss':float(loss.detach())})
    net.eval()
    return net,curve

@torch.no_grad()
def describe(net,images):
    # Deterministic four-rotation ensemble, also used at deployment.
    x=torch.from_numpy(images);out=[]
    for k in range(4):out.append(net(torch.rot90(x,k,(-2,-1))).numpy())
    return np.mean(out,axis=0)

def with_descriptors(d,descriptors):
    d=d.copy();d['phi']=[float(descriptors[s][0]) for s in d.surface];d['texvol']=[float(descriptors[s][1])*25 for s in d.surface]
    return d

def run_fold(surface):
    start=time.time()
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);names,images=get_images();t,_=load();train_names=sorted(s for s in t.surface.unique() if s!=surface)
        ix=[names.index(s) for s in train_names];net,curve=train_encoder(images[ix],targets(t,train_names));desc=describe(net,images)
        mapping=dict(zip(names,desc.tolist()));train=with_descriptors(t[t.surface!=surface],mapping);test=with_descriptors(t[t.surface==surface],mapping)
        gp=Model('baseline').fit(train);mu,sd=gp.predict(test)
        result={'heldout':surface,'cnn_train_surfaces':train_names,'gp_train_surfaces':gp.train_surfaces,
          'cnn_train_images':[f'{s}_{m}.tif' for s in train_names for m in MAGS],
          'row_id':test.row_id.tolist(),'mu':mu.tolist(),'sd':sd.tolist(),'descriptor_predictions':mapping,'loss_curve':curve,
          'epochs':EPOCHS,'seed':SEED,'gp_warnings':gp.warnings,'seconds':time.time()-start}
        assert surface not in train_names and 'REF-H' not in train_names
        return result

def run(workers=2):
    if not (ROOT/'data/sem/training_pixels.npz').exists():prepare()
    t,r=load();out=ROOT/'results/sem';out.mkdir(exist_ok=True);(ROOT/'models/sem').mkdir(exist_ok=True)
    signature=hashlib.sha256(Path(__file__).read_bytes()+(ROOT/'data/sem/training_pixels.npz').read_bytes()+(ROOT/'droplet/models.py').read_bytes()).hexdigest()
    surfaces=sorted(t.surface.unique());folds=[];pending=[]
    for s in surfaces:
        p=out/f'fold_{s}.json'
        if p.exists() and json.loads(p.read_text()).get('signature')==signature:folds.append(json.loads(p.read_text()))
        else:pending.append(s)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_fold,s) for s in pending]
        for future in as_completed(futures):
            z=future.result();z['signature']=signature;folds.append(z);(out/f"fold_{z['heldout']}.json").write_text(json.dumps(z,indent=2));print(f"SEM LOSO {len(folds)}/12 {z['heldout']} {z['seconds']:.1f}s",flush=True)
    mu=np.zeros(len(t));sd=mu.copy()
    for f in folds:mu[f['row_id']]=f['mu'];sd[f['row_id']]=f['sd']
    predictions=t[['row_id','surface','condition','beta']].copy();predictions['prediction']=np.exp(mu);predictions['mu_log']=mu;predictions['sd_log']=sd
    predictions.to_csv(out/'loso_predictions.csv',index=False)
    baseline=pd.read_csv(ROOT/'results/loso_predictions.csv');baseline=baseline[baseline.model=='baseline'].sort_values('row_id')
    y=t.beta.values;p=np.exp(mu);per={s:float(np.sqrt(np.mean((y[t.surface==s]-p[t.surface==s])**2))) for s in surfaces}
    summary={'model':'sem_gp','images':39,'textured_images':36,'reference_images':3,'magnifications':MAGS,'encoder':'Shared 3-layer CNN, ordered 3-view fusion, 32-unit head, two sigmoid descriptor outputs','epochs':EPOCHS,'seed':SEED,
      'training_targets':'Weak supervision: geometry-derived phi and texture-volume-per-area / 25. These are proxy labels, not new SEM measurements.',
      'protocol':'12 LOSO folds. All images/rotations of the outer surface and REF-H excluded from CNN and GP fitting. Each GP uses descriptors predicted by its own training-only CNN for both training and test inputs.',
      'loso_rmse':float(np.sqrt(np.mean((y-p)**2))),'macro_rmse':float(np.mean(list(per.values()))),'per_surface':per,
      'baseline_rmse':float(np.sqrt(np.mean((y-baseline.prediction.values)**2))),
      'delta_vs_baseline_ci95':bootstrap(y,p,baseline.prediction.values,t.surface.values),
      'nominal_gp_coverage':float(np.mean(abs(np.log(y)-mu)<=1.6448536269514722*sd)),
      'nominal_intervals_note':'Conditional GP uncertainty only; does not propagate CNN descriptor uncertainty. No calibrated 90% image-model guarantee.',
      'fold_descriptors':{f['heldout']:f['descriptor_predictions'][f['heldout']] for f in folds}}
    print('SEM LOSO RMSE',summary['loso_rmse'],flush=True)
    with threadpool_limits(limits=1):
        names,images=get_images();ix=[names.index(s) for s in surfaces];net,curve=train_encoder(images[ix],targets(t,surfaces));desc=describe(net,images);mapping=dict(zip(names,desc.tolist()))
        np.savez_compressed(ROOT/'models/sem/encoder_weights.npz',**{k:v.detach().numpy() for k,v in net.state_dict().items()})
        torch.save(net.state_dict(),ROOT/'models/sem/encoder.pt')
        train=with_descriptors(t,mapping);ref=with_descriptors(r,mapping);gp=Model('baseline').fit(train);refmu,refsd=gp.predict(ref)
        m=portable(gp);m.update({'kind':'sem_gp','label':'SEM CNN + GP','q':1.6448536269514722,'abs_width_log':0.,'surface_descriptors':{s:{'phi':v[0],'texvol':v[1]*25} for s,v in mapping.items()},'interval_note':summary['nominal_intervals_note']})
        m['refh']={'rmse':float(np.sqrt(np.mean((r.beta-np.exp(refmu))**2))),'coverage':float(np.mean(abs(np.log(r.beta)-refmu)<=m['q']*refsd))}
        packed=gp.gp.L_[np.tril_indices(len(t))].astype('<f8');packed.tofile(ROOT/'models/sem_gp.L.bin');m['L_sha256']=hashlib.sha256((ROOT/'models/sem_gp.L.bin').read_bytes()).hexdigest()
        (ROOT/'models/sem_gp.json').write_text(json.dumps(m,separators=(',',':')))
        refout=r[['row_id','surface','condition','beta']].copy();refout['prediction']=np.exp(refmu);refout['mu_log']=refmu;refout['sd_log']=refsd;refout.to_csv(out/'refh_predictions.csv',index=False)
        meta={'magnifications':MAGS,'side':SIDE,'crop_xyxy':[0,0,2560,1880],'training_surfaces':surfaces,'excluded_surfaces':['REF-H'],'epochs':EPOCHS,'seed':SEED,'parameters':sum(p.numel() for p in net.parameters()),'loss_curve':curve,'surface_descriptors':m['surface_descriptors'],'torch_version':torch.__version__,'weights_sha256':hashlib.sha256((ROOT/'models/sem/encoder_weights.npz').read_bytes()).hexdigest(),'targets':summary['training_targets']}
        (ROOT/'models/sem/encoder.json').write_text(json.dumps(meta,indent=2));summary['refh']=m['refh'];summary['encoder_parameters']=meta['parameters']
    summary['status']='experimental image branch; baseline remains default pending independent confirmation'
    release_path=ROOT/'models/release.json'
    release=json.loads(release_path.read_text());release['version']='2.1.0'
    release['candidates']['sem_gp']={'label':'SEM CNN + GP','refh':summary['refh'],'experimental':True}
    release['sem_encoder']='models/sem/encoder_weights.npz'
    release['sem_note']='CNN trained on real SEM images with geometry-proxy supervision; known-surface CNN descriptors are precomputed for live GP inference.'
    release_path.write_text(json.dumps(release,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);p.add_argument('--prepare',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    else:run(args.workers)
