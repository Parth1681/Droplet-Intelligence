import asyncio,sys,pathlib
from playwright.async_api import async_playwright
MM=96/25.4
async def main():
  async with async_playwright() as p:
    b=await p.chromium.launch();pg=await b.new_page(viewport={'width':int(841*MM),'height':int(1189*MM)})
    await pg.goto(pathlib.Path('poster.html').resolve().as_uri());await pg.wait_for_timeout(1500)
    info=await pg.evaluate("""()=>{const mm=v=>Math.round(v/3.7795);
      const cols=[...document.querySelectorAll('.col')].map(c=>{const tot=[...c.children].reduce((a,e)=>a+e.getBoundingClientRect().height,0);return mm(c.clientHeight-tot)+'mm slack, scroll '+mm(c.scrollHeight-c.clientHeight)});
      return {cols,body:mm(document.body.scrollHeight-document.body.clientHeight),hdr:mm(document.querySelector('header').offsetHeight),kt:mm(document.querySelector('.ktband').offsetHeight)}}""")
    print(info);await pg.screenshot(path='check.png')
    if '--pdf' in sys.argv: await pg.pdf(path='Parth_Sharma_Droplet_Poster_A0_Portrait.pdf',width='841mm',height='1189mm',print_background=True,prefer_css_page_size=True)
    await b.close()
asyncio.run(main())
