import warnings
import numpy as np
from scipy.optimize import minimize, least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, Matern, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from .data import FEATURES

CANDIDATES = ['baseline','matern52','structured','physics_residual']
LABELS = {'baseline':'Baseline Matérn 3/2 GP','matern52':'Matérn 5/2 GP','structured':'Fluid–surface interaction GP','physics_residual':'Laan + residual GP'}

class FluidSurfaceKernel(Kernel):
    """PSD sum of fluid, surface and product kernels, with shared ARD scales.

    Parameters are positive and optimized in log space. Gradients are analytic.
    sklearn orders hyperparameters alphabetically: amplitudes then length_scale.
    """
    def __init__(self, amplitudes=(1.,.1,.1), length_scale=(1.,1.,1.,1.,1.)):
        self.amplitudes=amplitudes; self.length_scale=length_scale
    @property
    def hyperparameter_amplitudes(self):
        return Hyperparameter('amplitudes','numeric',(1e-5,1e3),3)
    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter('length_scale','numeric',(1e-2,1e3),5)
    def __call__(self,X,Y=None,eval_gradient=False):
        a,b,c=self.amplitudes; ls=np.asarray(self.length_scale)
        if eval_gradient:
            if Y is not None: raise ValueError('Gradient requires Y=None')
            f,df=Matern(ls[:3],nu=1.5)(X[:,:3],eval_gradient=True)
            s,ds=Matern(ls[3:],nu=1.5)(X[:,3:],eval_gradient=True)
            K=a*f+b*s+c*f*s
            grad=np.concatenate([np.stack([a*f,b*s,c*f*s],axis=2),df*(a+c*s[:,:,None].squeeze(-1))[:,:,None],ds*(b+c*f)[:,:,None]],axis=2)
            return K,grad
        f=Matern(ls[:3],nu=1.5)(X[:,:3],None if Y is None else Y[:,:3])
        s=Matern(ls[3:],nu=1.5)(X[:,3:],None if Y is None else Y[:,3:])
        return a*f+b*s+c*f*s
    def diag(self,X): return np.full(len(X),sum(self.amplitudes))
    def is_stationary(self): return True

def optimizer(obj,theta,bounds):
    result=minimize(obj,theta,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-9,'gtol':1e-5})
    if not result.success: warnings.warn(str(result.message),ConvergenceWarning)
    return result.x,result.fun

class Model:
    def __init__(self,kind='baseline'): self.kind=kind
    def backbone(self,d):
        if self.kind!='physics_residual': return np.zeros(len(d))
        p=np.sqrt(d.We.values*d.Re.values**(-.4))
        return np.log(d.Re.values**.2*p/(self.A+p))
    def fit(self,d):
        self.train_surfaces=sorted(d.surface.unique().tolist())
        y=np.log(d.beta.values)
        self.A=None
        if self.kind=='physics_residual':
            p=np.sqrt(d.We.values*d.Re.values**(-.4))
            self.A=float(least_squares(lambda a:np.log(d.Re.values**.2*p/(a[0]+p))-y,[1.24],bounds=(.01,20)).x[0])
        self.scaler=StandardScaler().fit(d[FEATURES])
        x=self.scaler.transform(d[FEATURES])
        if self.kind=='structured': k=FluidSurfaceKernel()+WhiteKernel(.001,(1e-6,.1))
        else:k=ConstantKernel(1.)*Matern(np.ones(5),nu=2.5 if self.kind=='matern52' else 1.5,length_scale_bounds=(1e-2,1e3))+WhiteKernel(.001,(1e-6,.1))
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always',ConvergenceWarning)
            self.gp=GaussianProcessRegressor(k,normalize_y=True,optimizer=optimizer,random_state=23).fit(x,y-self.backbone(d))
        self.warnings=[str(w.message) for w in records]
        return self
    def predict(self,d):
        mu,sd=self.gp.predict(self.scaler.transform(d[FEATURES]),return_std=True)
        return mu+self.backbone(d),sd
