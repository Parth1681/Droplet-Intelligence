import sys, asyncio
from playwright.async_api import async_playwright
async def main(files):
    async with async_playwright() as p:
        b = await p.chromium.launch(); pg = await b.new_page(viewport={"width": 1000, "height": 800})
        for f in files:
            open("check/_t.html","w").write(f'<html><body style="margin:0;background:#fff"><img src="../assets/{f}.svg" style="width:960px"></body></html>')
            await pg.goto("file://archive/v1_droplet/poster/check/_t.html")
            await pg.wait_for_timeout(300)
            await pg.screenshot(path=f"check/{f}.png", full_page=True)
        await b.close()
asyncio.run(main(sys.argv[1:]))
