"""Phase 2 benchmark: physics laws vs XGBoost/MLP under ID, leave-one-surface-out and REF-H protocols.
No hyperparameter is tuned on REF-H; REF-H is predicted once by models trained on all textured data."""
import json
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.data import load

SEED = 0
XGB = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
           colsample_bytree=0.8, min_child_weight=5, random_state=SEED, n_jobs=4)

RAW = ["D", "V", "rho", "sigma", "mu", "spacing", "depth"]
PHYS = ["Re", "We", "Oh", "spacing", "depth"]
PHYS_NOGEO = ["logRe", "logWe"]
HYB = RAW + ["logRe", "logWe", "logOh", "P", "sp_D", "dep_sp"]


def feats(d):
    d = d.copy()
    d["sp_D"] = d.spacing * 1e-6 / d.D
    d["dep_sp"] = np.where(d.spacing > 0, d.depth / d.spacing.replace(0, np.nan), 0.0)
    return d


# ---------- physics backbones (fitted on the training fold only) ----------
class Laan:
    """beta = Re^(1/5) sqrt(P) / (A + sqrt(P)); A refitted (paper value 1.24)."""
    name = "Physics: Laan 2014 (A refit)"
    def fit(self, d, y):
        f = lambda a: self._f(d, a[0]) - y
        self.A = least_squares(f, [1.24]).x[0]; return self
    @staticmethod
    def _f(d, A):
        s = np.sqrt(d.P.values); return d.Re.values ** 0.2 * s / (A + s)
    def predict(self, d): return self._f(d, self.A)


class Power:
    """log beta = a + b log We + c log Re  (scaling-law regression)."""
    name = "Physics: power law in We, Re"
    def fit(self, d, y):
        X = np.c_[np.ones(len(d)), d.logWe, d.logRe]
        self.w = np.linalg.lstsq(X, np.log(y), rcond=None)[0]; return self
    def predict(self, d):
        return np.exp(np.c_[np.ones(len(d)), d.logWe, d.logRe] @ self.w)


class Tabular:
    def __init__(self, name, cols, kind="xgb", log_target=True):
        self.name, self.cols, self.kind, self.log = name, cols, kind, log_target
    def _m(self):
        if self.kind == "xgb": return XGBRegressor(**XGB)
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 32), alpha=1e-2, max_iter=3000,
            early_stopping=True, random_state=SEED))
    def fit(self, d, y):
        self.m = self._m().fit(d[self.cols], np.log(y) if self.log else y); return self
    def predict(self, d):
        p = self.m.predict(d[self.cols]); return np.exp(p) if self.log else p


class Residual:
    """beta = backbone(Re, We) * exp(ML residual(surface + physics features))."""
    def __init__(self, name, cols, backbone=Power):
        self.name, self.cols, self.bb = name, cols, backbone
    def fit(self, d, y):
        self.b = self.bb().fit(d, y)
        self.m = XGBRegressor(**XGB).fit(d[self.cols], np.log(y) - np.log(self.b.predict(d)))
        return self
    def predict(self, d): return self.b.predict(d) * np.exp(self.m.predict(d[self.cols]))


MODELS = [
    lambda: Laan(),
    lambda: Power(),
    lambda: Tabular("M1 XGB raw", RAW),
    lambda: Tabular("M2 XGB Re/We/Oh+geo", PHYS),
    lambda: Tabular("M2b XGB Re/We only", PHYS_NOGEO),
    lambda: Tabular("M3 XGB hybrid", HYB),
    lambda: Tabular("M4 MLP logRe/logWe+geo", ["logRe", "logWe", "spacing", "depth"], kind="mlp"),
    lambda: Residual("M5 Power-law + XGB residual (geo)", ["logRe", "logWe", "spacing", "depth"]),
    lambda: Residual("M5b Power-law + XGB residual (no geo)", ["logRe", "logWe"]),
]


def metrics(y, p):
    e = p - y
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MaxErr=float(np.max(np.abs(e))), Bias=float(np.mean(e)),
                R2=float(r2_score(y, p)), MAPE=float(np.mean(np.abs(e) / y) * 100))


def boot_ci(y, p, groups, n=2000, rng=np.random.default_rng(SEED)):
    g = pd.Series(groups).astype("category").cat.codes.values
    ug = np.unique(g); out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug))
        idx = np.concatenate([np.where(g == k)[0] for k in pick])
        out.append(np.sqrt(np.mean((p[idx] - y[idx]) ** 2)))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def run():
    t, r = load()
    t, r = feats(t), feats(r.assign(spacing=0.0, depth=0.0))   # smooth plate = no channels
    y, yr = t.beta.values, r.beta.values
    rows, preds = [], {}
    for mk in MODELS:
        name = mk().name
        # ID: grouped 5-fold on replicate conditions
        p_id = np.zeros(len(t))
        for tr, te in GroupKFold(5).split(t, y, t.cond):
            p_id[te] = mk().fit(t.iloc[tr], y[tr]).predict(t.iloc[te])
        # LOSO: 12 folds, one textured surface held out each time
        p_lo = np.zeros(len(t))
        for s in t["sample"].unique():
            te = (t["sample"] == s).values
            p_lo[te] = mk().fit(t[~te], y[~te]).predict(t[te])
        # REF-H: train on every textured experiment, predict once
        p_ref = mk().fit(t, y).predict(r)
        preds[name] = dict(id=p_id, loso=p_lo, ref=p_ref)
        mi, ml, mr = metrics(y, p_id), metrics(y, p_lo), metrics(yr, p_ref)
        rows.append({"Model": name, "ID RMSE": mi["RMSE"], "ID R2": mi["R2"],
                     "LOSO RMSE": ml["RMSE"], "LOSO worst-surface RMSE": max(
                         np.sqrt(np.mean((p_lo[t["sample"] == s] - y[t["sample"] == s]) ** 2))
                         for s in t["sample"].unique()),
                     "REF-H RMSE": mr["RMSE"], "REF-H CI95": boot_ci(yr, p_ref, r.cond),
                     "REF-H MAE": mr["MAE"], "REF-H MaxErr": mr["MaxErr"],
                     "REF-H Bias": mr["Bias"], "REF-H R2": mr["R2"], "REF-H MAPE": mr["MAPE"],
                     "ID->REF-H degradation %": (mr["RMSE"] / mi["RMSE"] - 1) * 100})
    res = pd.DataFrame(rows)
    return res, preds, t, r


if __name__ == "__main__":
    res, preds, t, r = run()
    pd.set_option("display.width", 250)
    print(res.drop(columns=["REF-H CI95"]).round(4).to_string(index=False))
    print(res[["Model", "REF-H CI95"]].to_string(index=False))
    res.to_json("results/phase2_benchmark.json", orient="records", indent=1)
    out = r[["cond", "gly", "V", "We", "Re", "beta"]].copy()
    for k, v in preds.items(): out[k] = v["ref"]
    out.to_csv("results/refh_predictions.csv", index=False)
    lo = t[["sample", "cond", "gly", "We", "Re", "beta"]].copy()
    for k, v in preds.items(): lo[k] = v["loso"]
    lo.to_csv("results/loso_predictions.csv", index=False)
