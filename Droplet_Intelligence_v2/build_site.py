"""Build the public website from showcase/ (one source for Vercel and a self-contained copy).

    python build_site.py                      # -> ../site  (Vercel service "site"; binary model factors)
    python build_site.py --artifact OUTDIR    # -> artifact folder (page fragment, base64 model factors)

The page runs every model in the browser: the five GP candidates, the image GP (φ read from an
uploaded SEM image), the evaluation tables, the 1,623 impacts and the 39 SEM views. All numbers shown
come from result files; stats.json is generated here from them. Re-run and commit ../site after
changing showcase/, models/ or results/.
"""
import argparse, base64, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ['baseline', 'matern52', 'structured', 'physics_residual', 'sem_gp', 'image_only']
SAMPLES = ['D200_43', 'S800_43', 'REF-H_43']
HEAD = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def stats():
    n = json.loads((ROOT.parent / 'paper/numbers.json').read_text())
    rel = json.loads((ROOT / 'models/release.json').read_text())
    ext = json.loads((ROOT / 'results/extensions/summary.json').read_text())
    r = pd.read_csv(ROOT / 'results/refh_predictions.csv'); b = r[r.model == 'baseline']
    return {'baseline': next(x for x in n['rows'] if x['key'] == 'baseline'),
            'zoo': [{'name': z['name'], 'loso': z['loso'], 'refh': z['refh']} for z in n['zoo']],
            'by_fluid': n['by_fluid'], 'data': n['data'], 'legacy': n['legacy'], 'laan_textbook': n['laan']['textbook'],
            'per_surface_cov': n['per_surface_cov'],
            'surfaces': {k: {f: v.get(f) for f in ('phi', 'texvol', 'spacing', 'depth')} for k, v in rel['surfaces'].items()},
            'refh_mape': float(np.mean(np.abs(b.prediction - b.beta) / b.beta) * 100),
            'image_phi': {k: v['image_phi'] for k, v in ext['image_reader']['per_surface'].items()}}


def build(out, artifact):
    shutil.rmtree(out, ignore_errors=True); (out / 'models/sem').mkdir(parents=True); (out / 'results/sem').mkdir(parents=True)
    (out / 'sem').mkdir(); (out / 'samples').mkdir()
    page = (ROOT / 'showcase/index.html').read_text()
    (out / 'index.html').write_text(page if artifact else HEAD + page.replace('<link rel="preconnect"', '</head>\n<body>\n<link rel="preconnect"', 1) + '\n</body>\n</html>\n')
    shutil.copy(ROOT / 'showcase/imagereader.mjs', out); shutil.copy(ROOT / 'showcase/video.mjs', out); shutil.copy(ROOT / 'web/engine.mjs', out)
    (out / 'stats.json').write_text(json.dumps(stats()))
    shutil.copy(ROOT / 'models/release.json', out / 'models'); shutil.copy(ROOT / 'models/sem/encoder.json', out / 'models/sem')
    for k in MODELS:
        shutil.copy(ROOT / f'models/{k}.json', out / 'models')
        L = (ROOT / f'models/{k}.L.bin').read_bytes()
        if artifact: (out / f'models/{k}.L.txt').write_bytes(base64.b64encode(L))
        else: (out / f'models/{k}.L.bin').write_bytes(L)
    for f in ['benchmark.json', 'nested.json', 'experiments.json']: shutil.copy(ROOT / 'results' / f, out / 'results')
    shutil.copy(ROOT / 'results/sem/summary.json', out / 'results/sem')
    for p in sorted((ROOT / 'data/sem').glob('*.jpg')): shutil.copy(p, out / 'sem')
    for s in SAMPLES: Image.open(ROOT / f'data/sem_original/{s}.tif').convert('L').save(out / f'samples/{s}.png', optimize=True)
    from droplet.synthetic_video import render, encode   # known-answer impact for the video section
    frames, truth = render(seed=0); encode(frames, str(out / 'samples/impact_sample.webm'))
    (out / 'samples/impact_sample.truth.json').write_text(json.dumps(truth))
    print('built', out, sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) // 2**20, 'MB')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--artifact', help='build a self-contained copy (base64 model files) here instead')
    a = ap.parse_args()
    build(Path(a.artifact).resolve() if a.artifact else (ROOT.parent / 'site'), bool(a.artifact))
