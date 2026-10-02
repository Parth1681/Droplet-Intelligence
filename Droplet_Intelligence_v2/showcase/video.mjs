// Drop-impact video measurement in the browser, a port of droplet/video.py.
// Backlit side view: drop and substrate dark, background bright.

// Collect every frame of a video file as greyscale arrays (downscaled to maxW pixels wide).
export async function grabFrames(file, { maxW = 640, maxFrames = 3000, onProgress } = {}) {
  const v = document.createElement('video');
  v.muted = true; v.playsInline = true; v.preload = 'auto'; v.src = URL.createObjectURL(file);
  await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = () => rej(Error('This browser cannot decode the video. Convert it to H.264 MP4 or WebM, e.g. ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4')); });
  const s = Math.min(1, maxW / v.videoWidth), W = Math.round(v.videoWidth * s), H = Math.round(v.videoHeight * s);
  const c = new OffscreenCanvas(W, H), ctx = c.getContext('2d', { willReadFrequently: true });
  const frames = [], seen = new Set();
  const grab = t => {
    if (seen.has(t) || frames.length >= maxFrames) return;
    seen.add(t); ctx.drawImage(v, 0, 0, W, H);
    const d = ctx.getImageData(0, 0, W, H).data, g = new Uint8Array(W * H);
    for (let i = 0; i < g.length; i++) g[i] = (d[4 * i] * 299 + d[4 * i + 1] * 587 + d[4 * i + 2] * 114) / 1000;
    frames.push({ t, g });
  };
  if ('requestVideoFrameCallback' in v) {
    v.playbackRate = .5;
    await new Promise((res, rej) => {
      const cb = (_, meta) => { grab(meta.mediaTime); onProgress?.(v.currentTime / v.duration); if (!v.ended && frames.length < maxFrames) v.requestVideoFrameCallback(cb); };
      v.requestVideoFrameCallback(cb); v.onended = res; v.onerror = rej; v.play().catch(rej);
    });
  } else {
    const fpsFile = 30;
    for (let t = .5 / fpsFile; t < v.duration && frames.length < maxFrames; t += 1 / fpsFile) {
      await new Promise(r => { v.onseeked = r; v.currentTime = t; }); grab(t); onProgress?.(t / v.duration);
    }
  }
  URL.revokeObjectURL(v.src);
  frames.sort((a, b) => a.t - b.t);
  // Frame index from the media timestamp, so a frame the browser skipped leaves a gap instead of shifting time.
  const dts = frames.slice(1).map((f, i) => f.t - frames[i].t).filter(d => d > 1e-6).sort((a, b) => a - b);
  let step = dts.length ? dts[dts.length >> 1] : 1;                    // median spacing, then refit over the whole clip
  if (frames.length > 1) { const span = frames[frames.length - 1].t - frames[0].t; step = span / Math.max(1, Math.round(span / step)); }
  const idx = frames.map(f => Math.round((f.t - frames[0].t) / step));
  return { W, H, scale: s, frames: frames.map(f => f.g), idx, skipped: idx.length ? idx[idx.length - 1] + 1 - idx.length : 0 };
}

function otsu(hist) {
  let tot = 0, sum = 0; for (let i = 0; i < 256; i++) { tot += hist[i]; sum += i * hist[i]; }
  let w0 = 0, s0 = 0, best = 0, thr = 128;
  for (let i = 0; i < 256; i++) { w0 += hist[i]; if (!w0) continue; const w1 = tot - w0; if (!w1) break; s0 += i * hist[i];
    const m0 = s0 / w0, m1 = (sum - s0) / w1, v = w0 * w1 * (m0 - m1) ** 2; if (v > best) { best = v; thr = i + .5; } }
  return thr;
}

function largestBlob(mask, W, H) {
  const lab = new Int32Array(W * H), stack = new Int32Array(W * H); let best = null, id = 0;
  for (let p = 0; p < W * H; p++) {
    if (!mask[p] || lab[p]) continue;
    id++; let n = 0, top = 0; stack[top++] = p; lab[p] = id;
    let area = 0, sy = 0, minY = H, maxY = 0, minX = W, maxX = 0;
    while (top) { const q = stack[--top], y = (q / W) | 0, x = q - y * W; area++; sy += y;
      if (y < minY) minY = y; if (y > maxY) maxY = y; if (x < minX) minX = x; if (x > maxX) maxX = x;
      for (const r of [q - 1, q + 1, q - W, q + W]) { if (r < 0 || r >= W * H || lab[r] || !mask[r]) continue; if ((r === q - 1 || r === q + 1) && ((r / W) | 0) !== y) continue; lab[r] = id; stack[top++] = r; } n++; }
    if (!best || area > best.area) best = { area, cy: sy / area, top: minY, bottom: maxY, left: minX, right: maxX };
  }
  return best && best.area >= 20 ? best : null;
}

export function analyse({ W, H, frames, idx }, fps, mmPerPx, { baseline = null, fitFrames = 8 } = {}) {
  const N = frames.length, hist = new Float64Array(256), T = idx || frames.map((_, i) => i);
  for (const g of frames) for (let i = 0; i < g.length; i += 4) hist[g[i]]++;
  const thr = otsu(hist);
  if (baseline == null) {
    const rowDark = Array.from({ length: H }, (_, y) => { const fr = frames.map(g => { let c = 0; for (let x = 0; x < W; x++) if (g[y * W + x] < thr) c++; return c / W; }).sort((a, b) => a - b); return fr[fr.length >> 1]; });
    let y = H - 1; while (y >= 0 && rowDark[y] < .8) y--;
    if (y < 0) throw Error('No substrate found. Enter the substrate row (pixels from the top).');
    while (y > 0 && rowDark[y - 1] >= .8) y--;
    baseline = y;
  }
  const recs = frames.map(g => { const m = new Uint8Array(W * baseline); for (let i = 0; i < m.length; i++) m[i] = g[i] < thr; return largestBlob(m, W, baseline); });
  const contact = recs.findIndex(r => r && r.bottom >= baseline - 2);
  if (contact < 0) throw Error('The drop never reaches the substrate in this video.');
  const pre = []; for (let i = 0; i < contact; i++) if (recs[i] && recs[i].top > 0) pre.push(i);
  const fit = pre.slice(-fitFrames);
  if (fit.length < 3) throw Error('Fewer than 3 frames of the falling drop before contact.');
  const d0s = fit.map(i => 2 * Math.sqrt(recs[i].area / Math.PI)).sort((a, b) => a - b), d0px = d0s[d0s.length >> 1];
  const xm = fit.reduce((a, i) => a + T[i], 0) / fit.length, ym = fit.reduce((a, i) => a + recs[i].cy, 0) / fit.length;
  const slope = fit.reduce((a, i) => a + (T[i] - xm) * (recs[i].cy - ym), 0) / fit.reduce((a, i) => a + (T[i] - xm) ** 2, 0);
  const series = []; let best = 0, iBest = contact;
  for (let i = contact; i < N; i++) if (recs[i]) { const w = recs[i].right - recs[i].left + 1; series.push({ t_ms: (T[i] - T[contact]) / fps * 1e3, D_mm: w * mmPerPx }); if (w > best) { best = w; iBest = i; } }
  return { D0_mm: d0px * mmPerPx, V: slope * mmPerPx * 1e-3 * fps, beta_max: best / d0px, t_max_ms: (T[iBest] - T[contact]) / fps * 1e3,
           contact, baseline, threshold: thr, frames: N, series, recs };
}
