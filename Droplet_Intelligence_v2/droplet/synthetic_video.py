"""Render a synthetic backlit drop-impact video with known D0, V and beta_max (for testing droplet.video).

    python -m droplet.synthetic_video --out test.mp4 --D0 2.5 --V 1.2 --beta 2.6

Bright background, dark drop and substrate, like a backlit high-speed recording. The drop falls at V,
touches the substrate, and its base spreads to beta*D0 along a sine-shaped curve, then recedes.
Gaussian blur and sensor noise are added. The ground truth is written next to the video as JSON.
"""
import argparse, json, subprocess
import numpy as np
from scipy import ndimage as ndi


def render(D0=2.5, V=1.2, beta=2.6, fps=5000, mm_per_px=.03, W=400, H=300, substrate_row=250, pre_frames=14,
           t_spread_ms=2.4, post_frames=40, noise=4.0, seed=0):
    rng = np.random.default_rng(seed)
    R = D0 / 2 / mm_per_px; step = V * 1e3 / fps / mm_per_px                 # px per frame
    yy, xx = np.mgrid[0:H, 0:W].astype(float); cx = W / 2
    frames = []
    n_spread = t_spread_ms * 1e-3 * fps
    for i in range(-pre_frames, post_frames):
        img = np.full((H, W), 215.0)
        img[substrate_row:] = 25
        cy = substrate_row - R + i * step                                      # contact at i = 0
        drop = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
        if i > 0:
            s = min(i / n_spread, 1.0) if i <= n_spread else max(1 - (i - n_spread) / (2.5 * n_spread), 0)
            half = R * (1 + (beta - 1) * np.sin(np.pi / 2 * s)) if i <= n_spread else R * (1 + (beta - 1) * s)
            h = max(R * 2 * (R / half) ** 2, 6)                                  # thin lamella as it spreads
            lam = ((xx - cx) / half) ** 2 + ((yy - substrate_row) / h) ** 2 <= 1
            cap_c = substrate_row - R + min(i * step, R * .9)
            drop = ((xx - cx) ** 2 + (yy - cap_c) ** 2 <= (R * .95) ** 2) & (yy <= substrate_row) | lam
        img[drop & (yy < substrate_row)] = 35
        img = ndi.gaussian_filter(img, 1.2) + rng.normal(0, noise, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    truth = dict(D0_mm=D0, V=V, beta_max=beta, fps=fps, mm_per_px=mm_per_px, substrate_row=substrate_row,
                 note='beta_max is the rendered base width at the end of spreading divided by D0')
    return np.stack(frames), truth


def encode(frames, out, file_fps=30):
    N, H, W = frames.shape
    codec = ['-c:v', 'libvpx-vp9', '-b:v', '0', '-crf', '20'] if out.endswith('.webm') else ['-c:v', 'libx264', '-crf', '12', '-pix_fmt', 'yuv420p']
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(file_fps),
                    '-i', '-', *codec, out], input=frames.tobytes(), check=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    for k, v in [('D0', 2.5), ('V', 1.2), ('beta', 2.6), ('fps', 5000.0), ('mm_per_px', .03), ('seed', 0)]:
        ap.add_argument('--' + k, type=type(v), default=v)
    a = ap.parse_args()
    f, t = render(a.D0, a.V, a.beta, a.fps, a.mm_per_px, seed=a.seed)
    encode(f, a.out); open(a.out.rsplit('.', 1)[0] + '.truth.json', 'w').write(json.dumps(t, indent=2))
    print('wrote', a.out, f.shape)
