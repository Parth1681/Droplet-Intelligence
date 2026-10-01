import asyncio, sys
from playwright.async_api import async_playwright
MM = 96 / 25.4
async def main(pdf=True):
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": int(841 * MM), "height": int(1189 * MM)})
        await pg.goto("" + __import__("pathlib").Path("poster.html").resolve().as_uri() + ""); await pg.wait_for_timeout(1200)
        colh = await pg.evaluate("""() => { const c=document.querySelector('.cols').getBoundingClientRect().top; const b=document.querySelector('.bottom').getBoundingClientRect().top;
             const h=(b - c)/3.7795 - 36; document.documentElement.style.setProperty('--colh', h+'mm'); return Math.round(h); }""")
        print('colh', colh); await pg.wait_for_timeout(300)
        info = await pg.evaluate("""() => {
          const mm = v => Math.round(v / 3.7795);
          const bt = document.querySelector('.bottom').getBoundingClientRect().top;
          const cols = [...document.querySelectorAll('.col')].map(c => mm(bt - c.lastElementChild.getBoundingClientRect().bottom) + 'mm free');
          const secs = [...document.querySelectorAll('.col')].map(c => { const s=[...c.children]; const tot=s.reduce((a,e)=>a+e.getBoundingClientRect().height,0); return mm(c.getBoundingClientRect().height - tot) + 'mm slack'; });
          const bot = [...document.querySelectorAll('.bottom > div')].map(d => mm(document.querySelector('.bottom').getBoundingClientRect().bottom - d.getBoundingClientRect().bottom) + 'mm');
          const fonts = [...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family+f.weight).length;
          const over = [...document.querySelectorAll('.col, .bottom > div, .band, .strip')].filter(e => e.scrollHeight > e.clientHeight + 2).length;
          return {cols, secs, bottom: bot, fontsLoaded: fonts, overflowing: over}; }""")
        print(info)
        await pg.screenshot(path="check_poster.png")
        if pdf:
            await pg.pdf(path="poster_A0.pdf", width="841mm", height="1189mm", print_background=True, prefer_css_page_size=True)
        await b.close()
asyncio.run(main('--nopdf' not in sys.argv))
