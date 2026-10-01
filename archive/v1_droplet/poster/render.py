import asyncio
from playwright.async_api import async_playwright
MM = 96 / 25.4
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": int(841 * MM), "height": int(1189 * MM)})
        await pg.goto("file://archive/v1_droplet/poster/poster.html"); await pg.wait_for_timeout(800)
        # overflow check: each column's bottom vs footer top
        info = await pg.evaluate("""() => { const f=document.querySelector('.footer').getBoundingClientRect().top;
            return [...document.querySelectorAll('.col')].map(c=>Math.round((f - c.lastElementChild.getBoundingClientRect().bottom)/3.7795)+'mm free') }""")
        print(info)
        await pg.screenshot(path="check/poster_preview.png", full_page=False)
        await pg.pdf(path="poster_A0.pdf", width="841mm", height="1189mm", print_background=True, prefer_css_page_size=True)
        await b.close()
asyncio.run(main())
