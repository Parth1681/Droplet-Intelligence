"""Screenshots of the interface and an in-browser latency measurement (Chromium, headless)."""
import json, os, subprocess, time, sys
from playwright.sync_api import sync_playwright
R = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.abspath(R + '/../../site'); PORT = 8791
srv = subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '-d', SITE], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5); out = {}
try:
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        pg = b.new_page(viewport={'width': 1280, 'height': 900}, device_scale_factor=2)
        pg.goto(f'http://localhost:{PORT}/index.html'); pg.wait_for_timeout(4000)
        pg.add_style_tag(content='.header{display:none !important}')
        pg.locator('[data-sample="D200_43"]').click(); pg.wait_for_timeout(9000)
        pg.locator('#vsample').click(); pg.wait_for_timeout(12000)
        for sec in ['lab', 'upload', 'video']:
            pg.locator(f'#{sec}').scroll_into_view_if_needed(); pg.wait_for_timeout(800)
            pg.locator(f'#{sec}').screenshot(path=f'{R}/figs/ui_{sec}.png')
        # in-browser latency: posterior mean + variance for the released baseline GP (n = 1498 training rows)
        out = pg.evaluate("""async () => {
          const { GP } = await import('./engine.mjs');
          const meta = await fetch('./models/baseline.json').then(r => r.json());
          const L = new Float64Array(await fetch('./models/baseline.L.bin').then(r => r.arrayBuffer()));
          const rel = await fetch('./models/release.json').then(r => r.json());
          const gp = new GP(meta, L), f = rel.fluids['0'], q = {D_mm: 2.5, V: 1.5, rho: f.rho, mu: f.mu, sigma: f.sigma, surface: 'D200'};
          gp.predict(q, rel); const ts = [];
          for (let i = 0; i < 200; i++) { const a = performance.now(); gp.predict(q, rel); ts.push(performance.now() - a); }
          ts.sort((a, b) => a - b);
          return {median_ms: ts[100], p95_ms: ts[190], n: 200, beta: gp.predict(q, rel).beta_max, training_rows: meta.n};
        }""")
        b.close()
finally:
    srv.terminate()
out['machine'] = 'headless Chromium in the cloud build container; timing excludes model download'
json.dump(out, open(R + '/browser_latency.json', 'w'), indent=1); print(out)
