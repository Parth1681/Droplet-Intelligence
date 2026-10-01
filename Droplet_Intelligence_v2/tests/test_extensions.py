"""Checks for the image reader and the image prediction tool (droplet.extensions, droplet.predict_image)."""
import json, unittest
from droplet import extensions as E
from droplet.data import ROOT

SUMMARY = E.OUT / 'summary.json'


class ImageReaderTests(unittest.TestCase):
    def test_pixel_size_from_metadata(self):
        self.assertAlmostEqual(E.pixel_size(E.SEM / 'D200_43.tif'), 500 / 432, places=6)
        self.assertAlmostEqual(E.pixel_size(E.SEM / 'D200_350.tif'), 500 / 432 * 43 / 350, places=6)

    @unittest.skipUnless(SUMMARY.exists(), 'run python -m droplet.extensions first')
    def test_smooth_plate_reads_as_smooth(self):
        thr = json.loads(SUMMARY.read_text())['image_reader']['threshold_all_textured']
        self.assertGreater(E.image_phi('REF-H', thr), .99)
        self.assertLess(abs(E.image_phi('D800', thr) - .891), .05)

    @unittest.skipUnless(SUMMARY.exists(), 'run python -m droplet.extensions first')
    def test_wrong_magnification_refused(self):
        from droplet.predict_image import read_phi
        with self.assertRaises(SystemExit):
            read_phi(E.SEM / 'D200_350.tif')

    @unittest.skipUnless(SUMMARY.exists(), 'run python -m droplet.extensions first')
    def test_baseline_reproduced(self):
        c = json.loads(SUMMARY.read_text())['candidates']['baseline']
        b = json.loads((ROOT / 'results/benchmark.json').read_text())['metrics'][0]
        self.assertAlmostEqual(c['loso']['rmse'], b['loso_rmse'], places=6)


if __name__ == '__main__':
    unittest.main()
