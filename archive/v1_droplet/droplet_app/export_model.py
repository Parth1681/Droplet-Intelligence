"""Fit the model chosen in Phase 7 on all textured rows and export it as plain arrays (model/bundle.npz + meta.json).
Run from the repo root:  python droplet_app/export_model.py
Checks the numpy predictor in core.py against sklearn before writing."""
import json, os, sys, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from sklearn.covariance import MinCovDet
from src.phases import build, GPR, LaanBackbone, FINAL
from src.surface import W_TRACK

APP = os.path.join(ROOT, "droplet_app"); OUT = os.path.join(APP, "model"); os.makedirs(OUT, exist_ok=True)
p7 = json.load(open("results/final/p7.json")); p5 = json.load(open("results/final/p5.json"))
key = p7["final"]
t, r = build()
m = GPR(FINAL, LaanBackbone if key == "M6" else __import__("src.phases", fromlist=["NoBackbone"]).NoBackbone).fit(t)
gp = m.m; k = gp.kernel_
X = t[FINAL].values
mcd = MinCovDet(random_state=0).fit(X)
np.savez(os.path.join(OUT, "bundle.npz"),
         Xtr=gp.X_train_, alpha=gp.alpha_, L=gp.L_, mu_x=m.sc.mean_, sd_x=m.sc.scale_,
         ls=np.atleast_1d(k.k1.k2.length_scale), c=k.k1.k1.constant_value, noise=k.k2.noise_level,
         y_mean=np.atleast_1d(gp._y_train_mean)[0], y_std=np.atleast_1d(gp._y_train_std)[0],
         mcd_loc=mcd.location_, mcd_prec=mcd.precision_, mah_sorted=np.sort(mcd.mahalanobis(X) ** 0.5))

fl = {}
for g, sub in t.groupby("gly"):
    fl[f"{g} wt.% glycerol"] = {"rho": float(sub.rho.median()), "sigma": float(sub.sigma.median()), "mu": float(sub.mu.median())}
sf = {s: {"spacing_um": float(sub.spacing.iloc[0]), "depth_um": float(sub.depth.iloc[0]),
          "phi": float(sub.phi.iloc[0]), "texvol_um": float(sub.texvol.iloc[0])}
      for s, sub in sorted(t.groupby("sample"), key=lambda kv: (kv[0][0], int(kv[0][1:])))}
sf["REF-H"] = {"spacing_um": 0.0, "depth_um": 0.0, "phi": 1.0, "texvol_um": 0.0}
meta = {"model_key": key, "model_name": p7["rule"] and next(d["model"] for d in p7["table"] if d["id"] == key),
        "features": FINAL, "backbone": {"type": "laan", "A": float(m.b.A)} if key == "M6" else None,
        "conformal_q": p5[key]["q_conformal"], "surface_halfwidth_log": p5[key]["surface_conformal_halfwidth_log"],
        "refh_coverage": p5[key]["cov_conformal_REFH"],
        "track_width_um": {str(a): b for a, b in W_TRACK.items()},
        "train_ranges": {"Re": [float(t.Re.min()), float(t.Re.max())], "We": [float(t.We.min()), float(t.We.max())],
                         "D_mm": [float(t.D.min() * 1e3), float(t.D.max() * 1e3)]},
        "fluids": fl, "surfaces": sf}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)

# ---- check numpy predictor == sklearn
sys.path.insert(0, APP)
from core import Predictor
P = Predictor()
for d in (t.sample(200, random_state=0), r):
    mu_sk, sd_sk = m.predict(d, std=True)
    mu_np, sd_np = P.predict_log(d[FINAL].values, d.Re.values, d.We.values)
    print("max |mean diff|", np.abs(mu_sk - mu_np).max(), " max |std diff|", np.abs(sd_sk - sd_np).max())
    assert np.abs(mu_sk - mu_np).max() < 1e-6 and np.abs(sd_sk - sd_np).max() < 1e-6
# Re used in the app is recomputed from rho V D / mu: check it matches the dataset's Re
row = t.iloc[0]
out = P.predict(row.D * 1e3, row.V, row.rho, row.sigma, row.mu, row.spacing, row.depth)
print("example", row["sample"], "beta", round(row.beta, 3), "->", {k: out[k] for k in ["beta_max", "interval90", "Re"]}, "dataset Re", round(row.Re, 1))

# copy results the dashboard shows
DD = os.path.join(APP, "data"); os.makedirs(DD, exist_ok=True)
for f in ["results/final/p34.json", "results/final/p5.json", "results/final/p6.json", "results/final/p7.json",
          "results/phase2_benchmark.json", "results/phase3.json", "results/phase4.json", "results/zoo_table.csv",
          "results/cnn_descriptors.json", "results/final/shap.npz"]:
    if os.path.exists(f): shutil.copy(f, DD)
for s in ["D200_43.jpg", "REF-H_43.jpg", "S800_43.jpg", "D200_350.jpg"]:
    shutil.copy(os.path.join("poster/assets", s), DD)
# per-row LOSO predictions of the chosen model, for parity plots
pr = np.load(f"results/final/pred_{key}.npz")
import pandas as pd
pd.DataFrame({"surface": t["sample"], "gly": t.gly, "We": t.We, "beta": t.beta, "pred": pr["loso"],
              "lo": np.exp(pr["loso_mu"] - p5[key]["q_conformal"] * pr["loso_sd"]),
              "hi": np.exp(pr["loso_mu"] + p5[key]["q_conformal"] * pr["loso_sd"])}).to_csv(os.path.join(DD, "loso_pred.csv"), index=False)
pd.DataFrame({"surface": "REF-H", "gly": r.gly, "We": r.We, "beta": r.beta, "pred": pr["ref"],
              "lo": np.exp(pr["ref_mu"] - p5[key]["q_conformal"] * pr["ref_sd"]),
              "hi": np.exp(pr["ref_mu"] + p5[key]["q_conformal"] * pr["ref_sd"])}).to_csv(os.path.join(DD, "refh_pred.csv"), index=False)
print("exported", key)
