"""FT-Transformer (Gorishniy et al., NeurIPS 2021) for the droplet table, same features and protocol as the zoo.
Each numeric feature becomes a token (x_j * W_j + b_j); a [CLS] token is read out after self-attention.
Kept small (d=32, 2 layers) because there are only ~300 independent conditions."""
import sys, json
import numpy as np
import torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from src.data import load
from src.models_extra import feats, F
from src.benchmark import metrics, boot_ci

torch.set_num_threads(2)


class FTT(nn.Module):
    def __init__(self, n, d=32, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.W, self.b = nn.Parameter(torch.randn(n, d) * 0.1), nn.Parameter(torch.zeros(n, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, x):
        t = x[..., None] * self.W + self.b
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        return self.out(self.tr(t)[:, 0]).squeeze(-1)


class FTTRegressor:
    def __init__(self, epochs=300, seed=0): self.epochs, self.seed = epochs, seed
    def fit(self, X, y):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.ym, self.ys = y.mean(), y.std()
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        yt = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
        self.m = FTT(X.shape[1]); opt = torch.optim.AdamW(self.m.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        for _ in range(self.epochs):
            self.m.train(); idx = rng.permutation(len(Xt))
            for b in range(0, len(idx), 128):
                j = idx[b:b + 128]
                loss = nn.functional.mse_loss(self.m(Xt[j]), yt[j]); opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
        return self
    @torch.no_grad()
    def predict(self, X):
        self.m.eval()
        return self.m(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)).numpy() * self.ys + self.ym


def run():
    t, r = load(); t = feats(t); r = feats(r.assign(spacing=0.0, depth=0.0))
    y, yr, X, Xr = t.beta.values, r.beta.values, t[F].values, r[F].values
    fit = lambda tr: FTTRegressor().fit(X[tr], np.log(y[tr]))
    p_id = np.zeros(len(t))
    for tr, te in GroupKFold(5).split(t, y, t.cond): p_id[te] = np.exp(fit(tr).predict(X[te]))
    p_lo, per = np.zeros(len(t)), {}
    for s in t["sample"].unique():
        te = (t["sample"] == s).values
        p_lo[te] = np.exp(fit(~te).predict(X[te])); per[s] = float(np.sqrt(np.mean((p_lo[te] - y[te]) ** 2)))
    p_ref = np.exp(fit(np.ones(len(t), bool)).predict(Xr))
    mi, ml, mr = metrics(y, p_id), metrics(y, p_lo), metrics(yr, p_ref); worst = max(per, key=per.get)
    res = [{"Model": "FT-Transformer (d=32, 2 layers)", "ID RMSE": mi["RMSE"], "ID R2": mi["R2"],
            "LOSO RMSE": ml["RMSE"], "Worst LOSO": f"{per[worst]:.3f} ({worst})", "REF-H RMSE": mr["RMSE"],
            "REF-H CI95": boot_ci(yr, p_ref, r.cond), "REF-H Bias": mr["Bias"], "REF-H MaxErr": mr["MaxErr"],
            "REF-H R2": mr["R2"]}]
    json.dump(res, open("results/zoo/ft_transformer.json", "w")); np.save("results/zoo/ft_transformer_ref.npy", p_ref)
    print(res)


if __name__ == "__main__":
    run()
