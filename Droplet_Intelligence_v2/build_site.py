"""Assemble the static website (browser-side GP inference) into public/ for Vercel or any static host.

    python build_site.py

Mirrors droplet/serve.py: web/ at the site root, plus models/, results/ and data/sem/ beside it,
and the downloadable package zip (without the 200 MB SEM originals), as the local server provides.
"""
import shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'public'

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(ROOT / 'web', OUT)
for d in ('models', 'results', 'data/sem'):
    shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns('__pycache__'))
(OUT / 'downloads').mkdir()
with zipfile.ZipFile(OUT / 'downloads' / 'Droplet_Intelligence_v2.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if p.is_file() and rel.parts[0] not in ('public', 'node_modules') and '__pycache__' not in rel.parts \
                and 'sem_original' not in rel.parts and not p.name.endswith(('.log', '.zip')):
            z.write(p, 'Droplet_Intelligence_v2/' + str(rel))
print('built', OUT, sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) // 2**20, 'MB')
