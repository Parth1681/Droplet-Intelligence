// SEM image reader + image GP, a browser port of droplet/extensions.py and droplet/predict_image.py.
// phi = share of pixels whose local intensity standard deviation (window fixed in micrometres) is at or
// below a threshold calibrated on the 12 textured training surfaces.

// ---------- decoding ----------
export function parseTiff(buf) {
  const v = new DataView(buf), le = v.getUint16(0) === 0x4949;
  if (!le && v.getUint16(0) !== 0x4d4d) throw Error('Not a TIFF file');
  const u16 = o => v.getUint16(o, le), u32 = o => v.getUint32(o, le);
  if (u16(2) !== 42) throw Error('Unsupported TIFF variant (BigTIFF)');
  const ifd = u32(4), n = u16(ifd), tags = {};
  const size = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 16: 8 };
  for (let i = 0; i < n; i++) {
    const e = ifd + 2 + i * 12, tag = u16(e), type = u16(e + 2), count = u32(e + 4), bytes = (size[type] || 1) * count;
    const off = bytes <= 4 ? e + 8 : u32(e + 8), vals = [];
    if (type === 3) for (let k = 0; k < count; k++) vals.push(u16(off + 2 * k));
    else if (type === 4) for (let k = 0; k < count; k++) vals.push(u32(off + 4 * k));
    else if (type === 1 || type === 7 || type === 2) tags[tag + '_bytes'] = new Uint8Array(buf, off, count);
    tags[tag] = vals.length === 1 ? vals[0] : vals;
  }
  const W = tags[256], H = tags[257], bps = Array.isArray(tags[258]) ? tags[258][0] : (tags[258] || 1), spp = tags[277] || 1;
  if ((tags[259] || 1) !== 1) throw Error('Compressed TIFF. Save it as an uncompressed TIFF or as PNG and try again');
  if (bps !== 8 && bps !== 16) throw Error(`Unsupported bit depth ${bps}`);
  const offs = [].concat(tags[273]), counts = [].concat(tags[279]), raw = new Uint8Array(W * H * spp * bps / 8);
  let p = 0; offs.forEach((o, i) => { raw.set(new Uint8Array(buf, o, counts[i]).subarray(0, raw.length - p), p); p += counts[i]; });
  const gray = new Uint8Array(W * H), invert = tags[262] === 0;
  for (let i = 0; i < W * H; i++) {
    let g;
    if (bps === 8) g = spp === 1 ? raw[i] : Math.round(0.299 * raw[i * spp] + 0.587 * raw[i * spp + 1] + 0.114 * raw[i * spp + 2]);
    else { const k = i * spp * 2, s = le ? raw[k] | raw[k + 1] << 8 : raw[k] << 8 | raw[k + 1]; g = s >> 8; }
    gray[i] = invert ? 255 - g : g;
  }
  let mag = null;
  const meta = tags['40094_bytes'];
  if (meta) { const txt = new TextDecoder('utf-16le').decode(meta); const m = /Mag\/(\d+(\.\d+)?)/.exec(txt); if (m) mag = Number(m[1]); }
  return { width: W, height: H, gray, mag, source: 'tiff' };
}

export async function decodeFile(file) {
  const buf = await file.arrayBuffer(), head = new Uint8Array(buf, 0, 4);
  if ((head[0] === 0x49 && head[1] === 0x49) || (head[0] === 0x4d && head[1] === 0x4d)) return parseTiff(buf);
  const bmp = await createImageBitmap(new Blob([buf]));
  const c = new OffscreenCanvas(bmp.width, bmp.height), ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data, gray = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < gray.length; i++) gray[i] = Math.round(0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]);
  return { width: bmp.width, height: bmp.height, gray, mag: null, source: 'bitmap' };
}

// ---------- roughness map (scipy reflect boundaries) ----------
const refl = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };
function conv1d(src, dst, W, H, w, horizontal) {
  const r = (w.length - 1) / 2;
  if (horizontal) for (let y = 0; y < H; y++) { const o = y * W; for (let x = 0; x < W; x++) { let s = 0; for (let k = -r; k <= r; k++) s += w[k + r] * src[o + refl(x + k, W)]; dst[o + x] = s; } }
  else for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { let s = 0; for (let k = -r; k <= r; k++) s += w[k + r] * src[refl(y + k, H) * W + x]; dst[y * W + x] = s; }
}
function boxMean(src, dst, W, H, size, horizontal) {
  const r = (size - 1) / 2;
  if (horizontal) for (let y = 0; y < H; y++) { const o = y * W; let s = 0; for (let k = -r; k <= r; k++) s += src[o + refl(k, W)];
    for (let x = 0; x < W; x++) { dst[o + x] = s / size; s += src[o + refl(x + r + 1, W)] - src[o + refl(x - r, W)]; } }
  else for (let x = 0; x < W; x++) { let s = 0; for (let k = -r; k <= r; k++) s += src[refl(k, H) * W + x];
    for (let y = 0; y < H; y++) { dst[y * W + x] = s / size; s += src[refl(y + r + 1, H) * W + x] - src[refl(y - r, H) * W + x]; } }
}
export function roughness(img, umPerPx, footerY, windowUm) {
  const W = img.width, H = Math.min(img.height, footerY > 0 ? footerY : img.height), N = W * H;
  const g = Array.from({ length: 9 }, (_, i) => Math.exp(-((i - 4) ** 2) / 2)), gs = g.reduce((a, b) => a + b, 0), gw = g.map(x => x / gs);
  const a = new Float64Array(N); for (let i = 0; i < N; i++) a[i] = img.gray[i];
  const t = new Float64Array(N); conv1d(a, t, W, H, gw, false); conv1d(t, a, W, H, gw, true);   // a = blurred
  const win = Math.max(3, Math.round(windowUm / umPerPx) | 1);
  const m = new Float64Array(N); boxMean(a, t, W, H, win, false); boxMean(t, m, W, H, win, true);
  for (let i = 0; i < N; i++) t[i] = a[i] * a[i];
  boxMean(t, a, W, H, win, false); boxMean(a, t, W, H, win, true);                                 // t = E[a^2]
  const sd = new Float32Array(N); for (let i = 0; i < N; i++) sd[i] = Math.sqrt(Math.max(t[i] - m[i] * m[i], 0));
  return { sd, width: W, height: H, win };
}
export function phiFrom(map, threshold) { let c = 0; for (let i = 0; i < map.sd.length; i++) if (map.sd[i] <= threshold) c++; return c / map.sd.length; }

// ---------- image GP ----------
export function loadImageGP(meta, L) {
  const d = meta.features.length, s = meta.kernel;
  function posterior(xraw) {
    const x = xraw.map((v, i) => (v - meta.xmean[i]) / meta.xscale[i]), ks = new Float64Array(meta.n); let sum = 0;
    for (let j = 0; j < meta.n; j++) { let r2 = 0; for (let i = 0; i < d; i++) { const z = (x[i] - meta.X[j][i]) / s.length_scale[i]; r2 += z * z; }
      const z3 = Math.sqrt(3 * r2), k = s.amplitude * (1 + z3) * Math.exp(-z3); ks[j] = k; sum += k * meta.alpha[j]; }
    const v = new Float64Array(meta.n); let norm = 0, off = 0;
    for (let i = 0; i < meta.n; i++) { let a = ks[i]; for (let j = 0; j < i; j++) a -= L[off + j] * v[j]; v[i] = a / L[off + i]; norm += v[i] * v[i]; off += i + 1; }
    return { mu: sum * meta.yscale + meta.ymean, sd: Math.sqrt(Math.max(s.amplitude + meta.noise - norm, 1e-12)) * meta.yscale };
  }
  return {
    predict({ phi, D_mm, V, rho, mu, sigma }) {
      const D = D_mm / 1000, Re = rho * V * D / mu, We = rho * V * V * D / sigma;
      const p = posterior([Math.log(Re), Math.log(We), Math.log(D_mm), phi]), reasons = [];
      for (const [k, val] of [['Re', Re], ['We', We], ['D', D], ['V', V]]) { const [lo, hi] = meta.support[k]; if (val < lo || val > hi) reasons.push(`${k} outside measured range`); }
      const [plo, phi_hi] = meta.phi_range;
      if (phi < plo || phi > phi_hi) reasons.push(`φ ${phi.toFixed(3)} outside the training surfaces (${plo.toFixed(2)} to ${phi_hi.toFixed(2)})`);
      if (phi > .95) reasons.push('Smooth surface: wettability is not modelled; on the smooth plate only about 62% of impacts fell inside the interval');
      return { beta: Math.exp(p.mu), lo: Math.exp(p.mu - meta.q * p.sd), hi: Math.exp(p.mu + meta.q * p.sd), Re, We, mu_log: p.mu, sd_log: p.sd, reasons };
    }
  };
}
