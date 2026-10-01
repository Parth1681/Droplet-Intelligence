"""The video pipeline recovers known D0, V and beta_max from synthetic backlit impacts."""
import tempfile, unittest
from pathlib import Path
from droplet.synthetic_video import render, encode
from droplet.video import read_frames, analyse


class VideoTests(unittest.TestCase):
    def test_known_answers(self):
        for k, (D0, V, b) in enumerate([(2.5, 1.2, 2.6), (2.3, .6, 1.7), (2.6, 1.6, 3.0)]):
            f, _ = render(D0, V, b, seed=k)
            with tempfile.TemporaryDirectory() as d:
                p = str(Path(d) / 'v.mp4'); encode(f, p); m = analyse(read_frames(p), 5000, .03)
            self.assertAlmostEqual(m['D0_mm'], D0, delta=.02)
            self.assertAlmostEqual(m['V'], V, delta=.01)
            self.assertAlmostEqual(m['beta_max'], b, delta=.03 * b)


if __name__ == '__main__':
    unittest.main()
