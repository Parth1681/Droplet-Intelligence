"""Non-XGBoost model zoo under the same leakage-safe protocols (grouped CV, LOSO, REF-H once).

Common feature set F for every model, so differences come from the learner, not the inputs:
  log Re, log We, log D0, smooth-plateau fraction phi, texture volume tex*depth (src/surface.py).
Both surface descriptors are physically defined for REF-H (phi = 1, texvol = 0), so no arbitrary encoding.
Sensitivity to the measured track width (+/-10 um) is reported instead.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.data import load
from src.surface import add_surface, W_TRACK
from src.benchmark import metrics, boot_ci, XGB, SEED

warnings.filterwarnings("ignore")
F = ["logRe", "logWe", "logD", "phi", "texvol"]


def feats(d, w=W_TRACK):
    d = add_surface(d, w)
    d["logD"] = np.log(d.D * 1e3)
    return d


def scaled(m): return make_pipeline(StandardScaler(), m)


def gpr():
    k = C(1.0) * Matern(length_scale=np.ones(len(F)), nu=1.5) + WhiteKernel(1e-3)
    return scaled(GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=SEED))


ZOO = {
    "Ridge, quadratic in log-features": lambda: make_pipeline(
        StandardScaler(), PolynomialFeatures(2), RidgeCV(np.logspace(-4, 2, 20))),
    "kNN (k=10, distance-weighted)": lambda: scaled(KNeighborsRegressor(10, weights="distance")),
    "SVR (RBF)": lambda: scaled(SVR(C=10, epsilon=0.005, gamma="scale")),
    "Kernel ridge (RBF)": lambda: scaled(KernelRidge(alpha=1e-3, kernel="rbf", gamma=0.2)),
    "Gaussian process (Matern 3/2, ARD)": gpr,
    "Random forest": lambda: RandomForestRegressor(500, min_samples_leaf=3, n_jobs=4, random_state=SEED),
    "Extra trees": lambda: ExtraTreesRegressor(500, min_samples_leaf=3, n_jobs=4, random_state=SEED),
    "LightGBM": lambda: LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=15,
                                      min_child_samples=10, subsample=0.8, subsample_freq=1,
                                      colsample_bytree=0.8, random_state=SEED, verbose=-1),
    "CatBoost": lambda: CatBoostRegressor(iterations=800, depth=5, learning_rate=0.05,
                                          random_seed=SEED, verbose=0),
    "MLP (64-64, strong L2)": lambda: scaled(MLPRegressor(hidden_layer_sizes=(64, 64), alpha=0.1, max_iter=5000,
                                                          early_stopping=True, random_state=SEED)),
    "Deep ensemble (5 x MLP)": lambda: DeepEnsemble(),
    "XGBoost (reference, same features)": lambda: XGBRegressor(**XGB),
}

WIDTHS = {"w nominal": W_TRACK, "w -10um": {25: 35.0, 6: 20.0}, "w +10um": {25: 55.0, 6: 40.0}}


class DeepEnsemble:
    """5 MLPs with different seeds; mean = prediction, spread = epistemic uncertainty."""
    def __init__(self, k=5): self.k = k
    def fit(self, X, y):
        self.ms = [scaled(MLPRegressor(hidden_layer_sizes=(64, 64), alpha=0.1, max_iter=5000, early_stopping=True,
                                       random_state=s)).fit(X, y) for s in range(self.k)]
        return self
    def predict(self, X): return np.mean([m.predict(X) for m in self.ms], 0)


def fitpred(mk, tr, te):
    m = mk().fit(tr[F].values, np.log(tr.beta.values))
    return np.exp(m.predict(te[F].values)), m


def run(names=None):
    t, r = load()
    tw = {k: feats(t, w) for k, w in WIDTHS.items()}
    t = tw["w nominal"]
    r = feats(r.assign(spacing=0.0, depth=0.0))        # smooth plate: phi = 1, texvol = 0
    y, yr = t.beta.values, r.beta.values
    rows, refpred = [], {}
    for name, mk in ZOO.items():
        if names and name not in names: continue
        p_id = np.zeros(len(t))
        for tr, te in GroupKFold(5).split(t, y, t.cond):
            p_id[te], _ = fitpred(mk, t.iloc[tr], t.iloc[te])
        p_lo, per = np.zeros(len(t)), {}
        for s in t["sample"].unique():
            te = (t["sample"] == s).values
            p_lo[te], _ = fitpred(mk, t[~te], t[te])
            per[s] = np.sqrt(np.mean((p_lo[te] - y[te]) ** 2))
        enc = {}
        for k, tk in tw.items():
            mk_ = mk().fit(tk[F].values, np.log(y))
            enc[k] = np.sqrt(np.mean((np.exp(mk_.predict(r[F].values)) - yr) ** 2))
            if k == "w nominal": m = mk_
        p_ref = np.exp(m.predict(r[F].values)); refpred[name] = p_ref
        mi, ml, mr = metrics(y, p_id), metrics(y, p_lo), metrics(yr, p_ref)
        worst = max(per, key=per.get)
        rows.append({"Model": name, "ID RMSE": mi["RMSE"], "ID R2": mi["R2"], "LOSO RMSE": ml["RMSE"],
                     "Worst LOSO": f"{per[worst]:.3f} ({worst})", "REF-H RMSE": mr["RMSE"],
                     "REF-H CI95": boot_ci(yr, p_ref, r.cond), "REF-H Bias": mr["Bias"],
                     "REF-H MaxErr": mr["MaxErr"], "REF-H R2": mr["R2"],
                     "REF-H RMSE over w +/-10um": (min(enc.values()), max(enc.values()))})
        print(f"done {name}", flush=True)
    return pd.DataFrame(rows), refpred, r


if __name__ == "__main__":
    import sys, json, re
    if len(sys.argv) > 1:                      # one model per process: results/zoo/<slug>.json
        name = sys.argv[1]; res, refpred, r = run([name])
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
        res.to_json(f"results/zoo/{slug}.json", orient="records")
        np.save(f"results/zoo/{slug}_ref.npy", refpred[name]); sys.exit()
    res, refpred, r = run()
    pd.set_option("display.width", 250)
    res = res.sort_values("LOSO RMSE")
    print(res.round(4).to_string(index=False))
    res.to_json("results/phase2b_model_zoo.json", orient="records", indent=1)
    out = r[["cond", "gly", "We", "beta"]].copy()
    for k, v in refpred.items(): out[k] = v
    out.to_csv("results/refh_predictions_zoo.csv", index=False)
