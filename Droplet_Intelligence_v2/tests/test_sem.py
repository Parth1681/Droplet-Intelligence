import unittest,json
import numpy as np
import pandas as pd
from droplet.data import ROOT,load
from droplet.predict import Predictor

class SEMTests(unittest.TestCase):
    def test_all_views_present(self):
        z=np.load(ROOT/'data/sem/training_pixels.npz');self.assertEqual(z['images'].shape,(13,3,128,128));self.assertEqual(z['magnifications'].tolist(),[43,100,350]);self.assertEqual(len(set(z['surfaces'])),13)
    def test_footer_removed(self):
        meta=json.loads((ROOT/'data/sem/manifest.json').read_text());self.assertEqual(len(meta['records']),39)
        for im in meta['records']:self.assertEqual(im['crop_xyxy'],[0,0,2560,1880])
    def test_every_image_split_isolated(self):
        files=list((ROOT/'results/sem').glob('fold_*.json'));self.assertEqual(len(files),12)
        for p in files:
            d=json.loads(p.read_text());self.assertEqual(len(d['cnn_train_images']),33);self.assertNotIn(d['heldout'],d['cnn_train_surfaces']);self.assertNotIn('REF-H',d['cnn_train_surfaces']);self.assertEqual(d['cnn_train_surfaces'],d['gp_train_surfaces'])
            self.assertTrue(all(not n.startswith(d['heldout']+'_') and not n.startswith('REF-H_') for n in d['cnn_train_images']))
    def test_reference_export(self):
        _,r=load();m=json.loads((ROOT/'models/sem_gp.json').read_text());desc=m['surface_descriptors']['REF-H'];r['phi']=desc['phi'];r['texvol']=desc['texvol'];mu,sd=Predictor().predict_frame(r,'sem_gp');g=pd.read_csv(ROOT/'results/sem/refh_predictions.csv');np.testing.assert_allclose(mu,g.mu_log,atol=2e-9,rtol=0);np.testing.assert_allclose(sd,g.sd_log,atol=2e-8,rtol=0)
    def test_pixels_change_encoder_output(self):
        import torch
        from droplet.sem import SEMEncoder,describe,get_images
        torch.set_num_threads(1);z=np.load(ROOT/'models/sem/encoder_weights.npz',allow_pickle=False);net=SEMEncoder();net.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files});net.eval();names,x=get_images();actual=describe(net,x);blank=describe(net,np.zeros_like(x));self.assertGreater(float(np.max(abs(actual-blank))),.01)
    def test_loso_metric(self):
        d=pd.read_csv(ROOT/'results/sem/loso_predictions.csv');s=json.loads((ROOT/'results/sem/summary.json').read_text());self.assertAlmostEqual(float(np.sqrt(np.mean((d.beta-d.prediction)**2))),s['loso_rmse'],places=12)

if __name__=='__main__':unittest.main()
