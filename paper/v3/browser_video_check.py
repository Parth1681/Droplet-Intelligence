"""Run the browser video pipeline (site/video.mjs) on the bundled synthetic clip five times in headless Chromium."""
import subprocess, sys, time, json
from playwright.sync_api import sync_playwright
srv = subprocess.Popen([sys.executable, '-m', 'http.server', '8793', '-d', '/home/user/iisc-droplet/site'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); time.sleep(1.5)
res = []
try:
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        for k in range(5):
            pg = b.new_page(); pg.goto('http://localhost:8793/index.html'); pg.wait_for_timeout(3000)
            r = pg.evaluate("""async () => { const {grabFrames, analyse} = await import('./video.mjs');
              const blob = await fetch('./samples/impact_sample.webm').then(r => r.blob());
              const d = await grabFrames(new File([blob], 'a.webm')); const m = analyse(d, 5000, 0.03);
              return {frames: d.frames.length, skipped: d.skipped, D0: m.D0_mm, V: m.V, beta: m.beta_max}; }""")
            res.append(r); pg.close()
        b.close()
finally: srv.terminate()
json.dump(res, open('/home/user/iisc-droplet/paper/v3/browser_video_check.json', 'w'), indent=1)
for r in res: print(r)
