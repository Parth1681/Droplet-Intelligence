"""CNN surface encoder: SEM image -> physical surface descriptors (phi, texvol) -> droplet model.

Why a CNN here (and not a TCN): the impact data is one row per experiment, with no time series, so there's
nothing for a TCN to convolve over. The SEM images are real 2-D data, and they exist for REF-H too.
The CNN learns to read texture descriptors from an image, so a NEW surface can be described from its SEM
image alone.

Honesty constraints:
  * REF-H images are never used in training; its descriptor is predicted blind.
  * Leave-one-surface-out: the CNN for surface s never sees images of s.
  * Only 12 training surfaces -> the CNN learns a descriptor regression, not beta directly.
"""
import glob, os, json
import numpy as np
import torch, torch.nn as nn
from PIL import Image
from src.surface import phi_smooth

torch.set_num_threads(2)
SEM = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "06 - SEM images of the test surfaces")
DS, P = 8, 128          # downsample 8x (9.3 um/px) and 128 px patches (~1.19 mm field of view)


def load_images():
    ims = {}
    for f in glob.glob(os.path.join(SEM, "*_43.tif")):
        n = os.path.basename(f)[:-7]
        im = Image.open(f).crop((0, 0, 2560, 1880))
        a = np.asarray(im.resize((2560 // DS, 1880 // DS), Image.BILINEAR), np.float32)
        ims[n] = (a - a.mean()) / (a.std() + 1e-6)
    return ims


def target(n):
    if n == "REF-H": return np.array([1.0, 0.0], np.float32)
    s, dep = int(n[1:]), 25 if n[0] == "D" else 6
    phi = float(phi_smooth([s], [dep])[0])
    return np.array([phi, (1 - phi) * dep / 25.0], np.float32)     # texvol scaled to [0, 1]


def crops(a, k, rng):
    H, W = a.shape; out = []
    for _ in range(k):
        y, x = rng.integers(0, H - P), rng.integers(0, W - P)
        c = a[y:y + P, x:x + P]
        c = np.rot90(c, rng.integers(4))
        if rng.random() < 0.5: c = c[:, ::-1]
        out.append(c * rng.uniform(0.85, 1.15) + rng.normal(0, 0.1))
    return np.stack(out)[:, None].copy()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 64), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 2), nn.Sigmoid())
    def forward(self, x): return self.h(self.f(x))
    def embed(self, x): return self.f(x).flatten(1)


def train(ims, names, epochs=12, per=24, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    for _ in range(epochs):
        X = np.concatenate([crops(ims[n], per, rng) for n in names])
        Y = np.concatenate([np.repeat(target(n)[None], per, 0) for n in names])
        idx = rng.permutation(len(X)); net.train()
        for b in range(0, len(X), 32):
            j = idx[b:b + 32]
            loss = nn.functional.mse_loss(net(torch.from_numpy(X[j])), torch.from_numpy(Y[j]))
            opt.zero_grad(); loss.backward(); opt.step()
    return net


@torch.no_grad()
def describe(net, a, k=48, seed=123):
    net.eval(); p = net(torch.from_numpy(crops(a, k, np.random.default_rng(seed)))).numpy()
    return p.mean(0), p.std(0)


def run():
    ims = load_images()
    tex = sorted([n for n in ims if n != "REF-H"], key=lambda n: (n[0], int(n[1:])))
    out = {}
    for s in tex:                                  # leave-one-surface-out CNN
        net = train(ims, [n for n in tex if n != s])
        m, sd = describe(net, ims[s])
        out[s] = dict(true=target(s).tolist(), pred=m.tolist(), sd=sd.tolist())
        print(s, np.round(target(s), 3), "->", np.round(m, 3), flush=True)
    net = train(ims, tex)                           # all 12 textured -> REF-H blind
    m, sd = describe(net, ims["REF-H"])
    out["REF-H"] = dict(true=target("REF-H").tolist(), pred=m.tolist(), sd=sd.tolist())
    print("REF-H", target("REF-H"), "->", np.round(m, 3))
    torch.save(net.state_dict(), "results/cnn_surface.pt")
    json.dump(out, open("results/cnn_descriptors.json", "w"), indent=1)


if __name__ == "__main__":
    run()
