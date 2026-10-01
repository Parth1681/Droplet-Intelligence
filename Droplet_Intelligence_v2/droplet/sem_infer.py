"""Predict from a new 43x/100x/350x SEM triplet using the trained CNN and GP."""
import argparse,json,hashlib
import numpy as np
from PIL import Image
import torch
from .sem import SEMEncoder,describe,SIDE
from .data import ROOT
from .predict import Predictor

def encode(paths):
    if len(paths)!=3:raise ValueError('Exactly three views are required, in 43x, 100x, 350x order')
    meta=json.loads((ROOT/'models/sem/encoder.json').read_text());weights=ROOT/'models/sem/encoder_weights.npz'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('Encoder integrity check failed')
    net=SEMEncoder();z=np.load(weights,allow_pickle=False);net.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files});net.eval();torch.set_num_threads(1)
    a=[]
    for p in paths:
        with Image.open(p) as im:
            if im.size!=(2560,2048):raise ValueError('This release expects the supplied SEM format: 2560×2048 with the annotation footer below y=1880. Validate other imaging formats separately.')
            v=np.array(im.convert('L').crop((0,0,2560,1880)).resize((SIDE,SIDE),Image.Resampling.LANCZOS),dtype=np.float32)/255
            a.append((v-v.mean())/max(float(v.std()),1e-3))
    result=describe(net,np.array([a],dtype=np.float32))[0]
    return {'phi':float(result[0]),'texvol':float(result[1]*25)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--images',nargs=3,required=True,metavar=('SEM43','SEM100','SEM350'))
    for key,default in [('D_mm',2.5),('V',1.5),('rho',997.2375),('mu',.000943923),('sigma',.07246)]:p.add_argument('--'+key,type=float,default=default)
    args=p.parse_args();d=encode(args.images);payload={k:getattr(args,k) for k in ['D_mm','V','rho','mu','sigma']}
    # Geometry is not an additional predictor in this five-feature image branch.
    payload.update({'surface':'custom','spacing':0.,'depth':0.,'model':'sem_gp',**d})
    result=Predictor().predict(payload);result['image_descriptors']=d;result['input_images']=[str(x) for x in args.images];print(json.dumps(result,indent=2))
if __name__=='__main__':main()
