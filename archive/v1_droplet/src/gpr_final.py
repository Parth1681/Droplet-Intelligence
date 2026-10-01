"""Final model (GPR, chosen on LOSO): per-row LOSO and REF-H predictions with predictive intervals,
empirical coverage, and ARD length-scales (a model-native explanation: short length-scale = influential)."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.data import load
from src.models_extra import feats, F, gpr

Z90 = norm.ppf(0.95)
t, r = load(); t = feats(t); r = feats(r.assign(spacing=0.0, depth=0.0))


def fit_pred(tr, te):
    m = gpr().fit(tr[F].values, np.log(tr.beta.values))
    mu, sd = m.predict(te[F].values, return_std=True)
    # pipeline(StandardScaler, GPR): return_std passes through to the final step
    return mu, sd, m


rows = []
for s in sorted(t["sample"].unique()):
    tr, te = t[t["sample"] != s], t[t["sample"] == s]
    mu, sd, _ = fit_pred(tr, te)
    rows.append(pd.DataFrame({"set": "LOSO", "surface": s, "gly": te.gly, "We": te.We, "Re": te.Re,
                              "beta": te.beta, "pred": np.exp(mu), "lo": np.exp(mu - Z90 * sd),
                              "hi": np.exp(mu + Z90 * sd)}))
    print(s, flush=True)
mu, sd, m = fit_pred(t, r)
rows.append(pd.DataFrame({"set": "REF-H", "surface": "REF-H", "gly": r.gly, "We": r.We, "Re": r.Re, "beta": r.beta,
                          "pred": np.exp(mu), "lo": np.exp(mu - Z90 * sd), "hi": np.exp(mu + Z90 * sd)}))
P = pd.concat(rows); P.to_csv("results/gpr_final_predictions.csv", index=False)
cov = P.assign(inside=(P.beta >= P.lo) & (P.beta <= P.hi)).groupby("set").inside.mean()
k = m[-1].kernel_
ls = dict(zip(F, np.atleast_1d(k.k1.k2.length_scale).tolist()))
out = {"coverage_90": cov.to_dict(), "ard_length_scales": ls, "kernel": str(k),
       "mean_interval_width": P.assign(w=P.hi - P.lo).groupby("set").w.mean().to_dict()}
json.dump(out, open("results/gpr_final.json", "w"), indent=1); print(json.dumps(out, indent=1))
