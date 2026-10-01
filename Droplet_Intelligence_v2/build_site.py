"""Assemble the static website (browser-side GP inference) into public/ for Vercel or any static host.

    python build_site.py                 # -> ../site (deployed on Vercel as service "site")
    python build_site.py --zip --out public   # also bundle the package zip, for self-hosting

Mirrors droplet/serve.py: web/ at the site root, plus models/, results/ and data/sem/ beside it.
Re-run and commit ../site after changing web/, models/ or results/.
"""
import argparse, shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_URL = 'https://github.com/Parth1681/iisc-droplet'
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=str(ROOT.parent / 'site'))
ap.add_argument('--zip', action='store_true', help='include downloads/Droplet_Intelligence_v2.zip (49 MB)')
args = ap.parse_args()
OUT = Path(args.out).resolve()

shutil.rmtree(OUT, ignore_errors=True)
shutil.copytree(ROOT / 'web', OUT)
for d in ('models', 'results', 'data/sem'):
    shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns('__pycache__'))
if not args.zip:
    # Without the bundled zip, point the "download package" links at the GitHub repository.
    html = OUT / 'index.html'
    html.write_text(html.read_text().replace('href="downloads/Droplet_Intelligence_v2.zip" download', f'href="{REPO_URL}" target="_blank" rel="noopener"'))
    print('built', OUT, sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) // 2**20, 'MB')
    raise SystemExit
(OUT / 'downloads').mkdir()
with zipfile.ZipFile(OUT / 'downloads' / 'Droplet_Intelligence_v2.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if p.is_file() and rel.parts[0] not in ('public', 'node_modules') and OUT not in p.parents and '__pycache__' not in rel.parts \
                and 'sem_original' not in rel.parts and not p.name.endswith(('.log', '.zip')):
            z.write(p, 'Droplet_Intelligence_v2/' + str(rel))
print('built', OUT, sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) // 2**20, 'MB')
