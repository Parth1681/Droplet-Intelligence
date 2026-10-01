import unittest,json
import numpy as np
import pandas as pd
from droplet.data import load,ROOT,audit
from droplet.models import FluidSurfaceKernel
from droplet.predict import Predictor
from droplet.nested import inner_splits,choose,order_quantile

class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.p=Predictor();cls.t,cls.r=load()
    def sample(self):return {'D_mm':2.5,'V':1.5,'rho':997.2375,'mu':.000943923,'sigma':.07246,'surface':'D200'}
    def test_data_integrity(self):
        a=audit();self.assertEqual((a['textured_rows'],a['reference_rows']),(1498,125));self.assertEqual(a['missing_cells'],0);self.assertEqual(a['duplicate_rows'],0);self.assertLess(a['re_relative_recompute_error'],1e-6)
    def test_physics_identity(self):
        r=self.p.predict(self.sample());q=r['physics'];self.assertAlmostEqual(q['Oh'],np.sqrt(q['We'])/q['Re']);self.assertGreater(r['beta_max'],0)
    def test_invalid_inputs(self):
        for k,v in [('D_mm',0),('V',-1),('mu',float('nan')),('rho',True),('sigma','0.07'),('surface','UNKNOWN')]:
            with self.subTest(k=k),self.assertRaises(ValueError):self.p.predict({**self.sample(),k:v})
        with self.assertRaises(ValueError):self.p.predict({**self.sample(),'typo':5})
    def test_smooth_warning(self):
        r=self.p.predict({**self.sample(),'surface':'REF-H'});self.assertEqual(r['reliability']['status'],'extrapolation');self.assertTrue(any('Smooth surface' in x for x in r['reliability']['reasons']))
    def test_interval_order(self):
        for kind in self.p.release['candidates']:
            r=self.p.predict({**self.sample(),'model':kind});self.assertLessEqual(r['interval']['lower'],r['beta_max']);self.assertGreaterEqual(r['interval']['upper'],r['beta_max'])
    def test_kernel_gradient(self):
        x=np.random.default_rng(2).normal(size=(8,5));k=FluidSurfaceKernel();K,G=k(x,eval_gradient=True)
        self.assertGreater(np.linalg.eigvalsh(K).min(),-1e-10)
        for i in range(len(k.theta)):
            th=k.theta.copy();th[i]+=1e-6;a=k.clone_with_theta(th)(x);th[i]-=2e-6;b=k.clone_with_theta(th)(x);np.testing.assert_allclose((a-b)/2e-6,G[:,:,i],atol=1e-7)
    def test_nested_isolation(self):
        for p in (ROOT/'results/inner_folds').glob('*.json'):
            d=json.loads(p.read_text());self.assertNotIn(d['outer'],d['train_surfaces']);self.assertNotIn(d['outer'],d['held']);self.assertFalse(set(d['held'])&set(d['train_surfaces']))
    def test_outer_isolation(self):
        for p in (ROOT/'results/folds').glob('*.json'):
            d=json.loads(p.read_text());self.assertNotIn(d['heldout'],d['train_surfaces'])
    def test_quantile_edge(self):
        self.assertTrue(np.isinf(order_quantile([1,2,3])));self.assertEqual(order_quantile(range(9)),8)
    def test_export_against_sklearn_reference(self):
        golden=pd.read_csv(ROOT/'results/refh_predictions.csv')
        for kind in ['baseline','matern52','structured','physics_residual']:
            mu,sd=self.p.predict_frame(self.r,kind);ref=golden[golden.model==kind].sort_values('row_id');np.testing.assert_allclose(mu,ref.mu_log,atol=2e-9,rtol=0);np.testing.assert_allclose(sd,ref.sd_log,atol=2e-8,rtol=0)
    def test_metrics_recompute(self):
        pred=pd.read_csv(ROOT/'results/nested_predictions.csv');metrics=json.loads((ROOT/'results/nested.json').read_text());self.assertAlmostEqual(float(np.sqrt(np.mean((pred.beta-pred.prediction)**2))),metrics['nested_rmse'],places=12)
    def test_partition_completeness(self):
        ss=sorted(self.t.surface.unique())
        for outer in ss:
            held=sum(inner_splits(outer,ss),[]);self.assertEqual(len(held),len(set(held)));self.assertEqual(set(held),set(ss)-{outer})

if __name__=='__main__':unittest.main()
