"""Phases 2-4 under one protocol.

Protocol (unchanged from earlier phases):
  ID    = GroupKFold(5) over replicate conditions (surface x fluid x velocity cluster)
  LOSO  = leave one textured surface out (12 folds)  <- selection criterion
  REF-H = train on all textured rows, predict the smooth surface once (never tuned on)

GPR is fitted row-level, exactly like the final model (an earlier condition-mean variant lost the per-impact
D0/V information and scored ~0.01 worse, so it was dropped). ID CV is run only where it is reported.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from xgboost import XGBRegressor
from src.data import load
from src.surface import add_surface

warnings.filterwarnings("ignore")
G = 9.81
SEED = 0
XGB = dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=5, random_state=SEED, n_jobs=2)

# measured advancing water contact angles per textured surface (file 03)
THETA_TEX = {"D50": 162.4, "D100": 166.4, "D200": 165.3, "D400": 166.6, "D600": 165.4, "D800": 163.6,
             "S50": 165.1, "S100": 166.5, "S200": 165.3, "S400": 165.3, "S600": 165.9, "S800": 164.9}


def beta0(theta_deg):
    """Spreading ratio at zero velocity from spherical-cap geometry (Lee et al. 2016)."""
    th = np.radians(theta_deg)
    return np.sin(th) * (4 / ((1 - np.cos(th)) ** 2 * (2 + np.cos(th)))) ** (1 / 3)


def laan(d, A=1.24):
    s = np.sqrt(d.P.values); return d.Re.values ** 0.2 * s / (A + s)


def build(theta_ref=114.5):
    t, r = load()
    t = add_surface(t); r = add_surface(r.assign(spacing=0.0, depth=0.0))   # REF-H: phi = 1, texvol = 0
    for d in (t, r):
        d["logD"] = np.log(d.D * 1e3)
        d["logP"] = np.log(d.P)
        d["logLaan"] = np.log(laan(d))
        d["Bo"] = d.rho * G * d.D ** 2 / d.sigma
        d["logmu"] = np.log(d.mu)
        d["depth_D"] = d.depth * 1e-6 / d.D
        d["texvol_D"] = d.texvol * 1e-6 / d.D
        d["sp_D"] = d.spacing * 1e-6 / d.D
        d["y"] = np.log(d.beta)
    t["theta"] = t["sample"].map(THETA_TEX); r["theta"] = theta_ref
    for d in (t, r): d["costh"] = np.cos(np.radians(d.theta)); d["beta0"] = beta0(d.theta)
    return t, r


# ------------------------------------------------------------------ backbones for residual learning
class NoBackbone:
    def fit(self, d): return self
    def __call__(self, d): return np.zeros(len(d))


class PowerBackbone:
    """log beta = a + b log We + c log Re"""
    def fit(self, d):
        X = np.c_[np.ones(len(d)), d.logWe, d.logRe]; self.w = np.linalg.lstsq(X, d.y, rcond=None)[0]; return self
    def __call__(self, d): return np.c_[np.ones(len(d)), d.logWe, d.logRe] @ self.w


class LaanBackbone:
    def fit(self, d):
        self.A = least_squares(lambda a: np.log(laan(d, a[0])) - d.y, [1.24]).x[0]; return self
    def __call__(self, d): return np.log(laan(d, self.A))


class LeeBackbone:
    """beta = sqrt( (Re^(1/5) f(P))^2 + beta0^2 ), f = Pade(A); A refit. Needs a contact angle per surface."""
    def fit(self, d):
        self.A = least_squares(lambda a: self._f(d, a[0]) - d.y, [1.24]).x[0]; return self
    @staticmethod
    def _f(d, A): return 0.5 * np.log(laan(d, A) ** 2 + d.beta0.values ** 2)
    def __call__(self, d): return self._f(d, self.A)


# ------------------------------------------------------------------ learners
def cond_means(d, cols):
    g = d.groupby("cond")
    agg = g[cols + ["y"]].mean()
    return agg[cols].values, agg["y"].values, g.size().loc[agg.index].values


class GPR:
    """Row-level GPR on log beta (same setup as the final model). Hyperparameters are re-optimised in every fold;
    `init` only sets the optimiser's starting point (warm start from the all-textured fit), which saves time."""
    def __init__(self, cols, backbone=NoBackbone, init=None):
        self.cols, self.bb, self.init = cols, backbone, init
    def fit(self, d):
        self.b = self.bb().fit(d)
        X, y = d[self.cols].values, (d.y - self.b(d)).values
        self.sc = StandardScaler().fit(X)
        k = self.init if self.init is not None else \
            C(1.0) * Matern(np.ones(len(self.cols)), nu=1.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-3, (1e-6, 1e-1))
        self.m = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=0, random_state=SEED).fit(self.sc.transform(X), y)
        return self
    def predict(self, d, std=False):
        mu, sd = self.m.predict(self.sc.transform(d[self.cols].values), return_std=True)
        mu = mu + self.b(d)
        return (mu, sd) if std else mu


class XGB_:
    def __init__(self, cols, backbone=NoBackbone, monotone=None):
        self.cols, self.bb, self.mono = cols, backbone, monotone
    def fit(self, d):
        self.b = self.bb().fit(d)
        kw = dict(XGB)
        if self.mono: kw["monotone_constraints"] = "(" + ",".join(str(self.mono.get(c, 0)) for c in self.cols) + ")"
        self.m = XGBRegressor(**kw).fit(d[self.cols].values, d.y - self.b(d)); return self
    def predict(self, d, std=False): return self.m.predict(d[self.cols].values) + self.b(d)


# ------------------------------------------------------------------ protocol
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def evaluate(name, make, t, r, intervals=False, do_id=False):
    bt, br = t.beta.values, r.beta.values
    m_full = make().fit(t)                                        # REF-H model (all textured rows)
    init = m_full.m.kernel_ if isinstance(m_full, GPR) else None  # warm start for fold fits
    def mk():
        m = make()
        if isinstance(m, GPR): m.init = init
        return m
    p_id = np.full(len(t), np.nan)
    if do_id:
        for tr, te in GroupKFold(5).split(t, t.y, t.cond):
            p_id[te] = np.exp(mk().fit(t.iloc[tr]).predict(t.iloc[te]))
    p_lo, lo_mu, lo_sd, per = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t)), {}
    for s in sorted(t["sample"].unique()):
        te = (t["sample"] == s).values
        m = mk().fit(t[~te])
        if intervals:
            mu, sd = m.predict(t[te], std=True); lo_mu[te], lo_sd[te] = mu, sd; p_lo[te] = np.exp(mu)
        else:
            p_lo[te] = np.exp(m.predict(t[te]))
        per[s] = rmse(p_lo[te], bt[te])
    m = m_full
    if intervals: rmu, rsd = m.predict(r, std=True); p_ref = np.exp(rmu)
    else: p_ref = np.exp(m.predict(r))
    hiW = (r.We >= r.We.quantile(2 / 3)).values
    worst = max(per, key=per.get)
    out = {"model": name, "ID": rmse(p_id, bt) if do_id else None, "LOSO": rmse(p_lo, bt), "worst": f"{per[worst]:.3f} {worst}",
           "REFH": rmse(p_ref, br), "REFH_bias": float(np.mean(p_ref - br)),
           "REFH_hiWe_bias": float(np.mean(p_ref[hiW] - br[hiW])),
           "REFH_MAPE": float(np.mean(np.abs(p_ref - br) / br) * 100)}
    if intervals:
        z = norm.ppf(0.95)
        yl, yr = np.log(bt), np.log(br)
        out["cov90_LOSO"] = float(np.mean(np.abs(yl - lo_mu) <= z * lo_sd))
        out["cov90_REFH"] = float(np.mean(np.abs(yr - rmu) <= z * rsd))
        out["width90_REFH"] = float(np.mean(np.exp(rmu + z * rsd) - np.exp(rmu - z * rsd)))
        # conformal: calibrate the LOSO normalised residuals, apply to REF-H
        q = np.quantile(np.abs(yl - lo_mu) / lo_sd, 0.9)
        out["conformal_q"] = float(q)
        out["cov90_REFH_conformal"] = float(np.mean(np.abs(yr - rmu) <= q * rsd))
        out["width90_REFH_conformal"] = float(np.mean(np.exp(rmu + q * rsd) - np.exp(rmu - q * rsd)))
        if hasattr(m, "m") and hasattr(m.m, "kernel_"):
            out["length_scales"] = dict(zip(m.cols, np.round(np.atleast_1d(m.m.kernel_.k1.k2.length_scale), 3).tolist()))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out


# ------------------------------------------------------------------ phases
RAW = ["D", "V", "rho", "sigma", "mu", "spacing", "depth"]
BASE = ["logRe", "logWe", "phi", "texvol"]
FINAL = ["logRe", "logWe", "logD", "phi", "texvol"]


def phase2(t, r):
    rows = []
    for lname, L in [("GPR", GPR), ("XGBoost", XGB_)]:
        rows.append(evaluate(f"{lname} | raw 7 (D,V,rho,sigma,mu,spacing,depth)", lambda L=L: L(RAW), t, r, do_id=True))
        rows.append(evaluate(f"{lname} | Re,We,Oh + spacing,depth (paper PIML set)", lambda L=L: L(["logRe", "logWe", "logOh", "spacing", "depth"]), t, r, do_id=True))
        rows.append(evaluate(f"{lname} | hybrid raw + Re,We,Oh", lambda L=L: L(RAW + ["logRe", "logWe", "logOh"]), t, r, do_id=True))
    return rows


def phase3(t, r):
    rows = [evaluate("base: logRe, logWe, phi, texvol", lambda: GPR(BASE), t, r)]
    for c in ["logD", "logOh", "logP", "logLaan", "Bo", "logmu", "depth_D", "sp_D", "costh"]:
        rows.append(evaluate(f"base + {c}", lambda c=c: GPR(BASE + [c]), t, r))
    for c in ["phi", "texvol"]:
        rows.append(evaluate(f"final - {c}", lambda c=c: GPR([x for x in FINAL if x != c]), t, r))
    return rows


def phase4(t, r):
    rows = []
    rows.append(evaluate("SEM descriptors phi, texvol  [final]", lambda: GPR(FINAL), t, r, intervals=True, do_id=True))
    rows.append(evaluate("surface-blind (logRe, logWe, logD)", lambda: GPR(["logRe", "logWe", "logD"]), t, r))
    rows.append(evaluate("raw geometry (spacing, depth; REF-H = 0/0)", lambda: GPR(["logRe", "logWe", "logD", "spacing", "depth"]), t, r))
    rows.append(evaluate("residual on power-law backbone", lambda: GPR(FINAL, PowerBackbone), t, r))
    rows.append(evaluate("residual on Laan backbone (A refit)", lambda: GPR(FINAL, LaanBackbone), t, r))
    rows.append(evaluate("XGBoost, SEM descriptors", lambda: XGB_(FINAL), t, r))
    rows.append(evaluate("XGBoost monotone (beta up in Re, We)", lambda: XGB_(FINAL, monotone={"logRe": 1, "logWe": 1}), t, r))
    for th in (110.0, 114.5, 120.0):
        t2, r2 = build(theta_ref=th)
        rows.append(evaluate(f"wettability: + cos(theta), REF-H theta={th:g}", lambda: GPR(FINAL + ["costh"]), t2, r2))
        rows.append(evaluate(f"Lee beta0 backbone + residual, REF-H theta={th:g}", lambda: GPR(FINAL, LeeBackbone), t2, r2))
    return rows


if __name__ == "__main__":
    ph = sys.argv[1]
    t, r = build()
    rows = {"2": phase2, "3": phase3, "4": phase4}[ph](t, r)
    json.dump(rows, open(f"results/phase{ph}.json", "w"), indent=1)
