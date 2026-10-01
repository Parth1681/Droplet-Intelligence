set -e
python3 build_poster.py && python3 render.py
python3 - <<'PY'
import asyncio
from playwright.async_api import async_playwright
from pypdf import PdfWriter, PdfReader
async def m():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page()
        import pathlib; await pg.goto(pathlib.Path('cover.html').resolve().as_uri()); await pg.wait_for_timeout(500)
        await pg.pdf(path='cover_A0.pdf',width='841mm',height='1189mm',print_background=True,prefer_css_page_size=True); await b.close()
asyncio.run(m())
w=PdfWriter()
for f in ['cover_A0.pdf','poster_A0.pdf']:
    for pg in PdfReader(f).pages: w.add_page(pg)
w.write('Droplet_Poster_with_Cover_A0.pdf')
PY
