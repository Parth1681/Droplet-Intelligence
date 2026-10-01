"""Phases 3-7 of the plan, one protocol (same as phases.py):
  ID    = GroupKFold(5) over replicate conditions
  LOSO  = leave one textured surface out (12 folds)   <- selection criterion
  REF-H = train on all textured rows, predict the smooth plate once (never tuned on)

Model ladder (names match src/benchmark.py and the plan):
  M1  XGBoost, raw 7 inputs
  M2  XGBoost, Re/We/Oh + spacing/depth               (reference for the REF-H comparison)
  M3  XGBoost, hybrid raw + Re/We/Oh
  M5  power-law backbone + XGBoost residual (geo)      [physics residual, tree learner]
  M6  Laan backbone + GPR residual, SEM descriptors    [physics residual, GP learner]
  GPR final: GPR on logRe, logWe, logD, phi, texvol    (chosen on LOSO earlier)

Stages (each writes results/final/<stage>.json and a status line to results/final/status.json):
  p34  M5/M6 vs M2 on REF-H with paired, condition-grouped bootstrap CI of the RMSE difference
  p5   conformal intervals (LOSO-calibrated) + OOD score vs error
  p6   SHAP (KernelExplainer) on the final GPR, plus ARD length-scales
  p7   benchmark table and the evidence-based final pick
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.phases import build, GPR, XGB_, PowerBackbone, LaanBackbone, FINAL, RAW
from src.ood import OODDetector

warnings.filterwarnings("ignore")
OUT = "results/final"; os.makedirs(OUT, exist_ok=True)
SEED = 0
Z90 = norm.ppf(0.95)

MODELS = {
    "M1": ("XGBoost, raw 7 inputs", lambda: XGB_(RAW)),
    "M2": ("XGBoost, Re/We/Oh + spacing/depth", lambda: XGB_(["logRe", "logWe", "logOh", "spacing", "depth"])),
    "M3": ("XGBoost, hybrid raw + Re/We/Oh", lambda: XGB_(RAW + ["logRe", "logWe", "logOh"])),
    "M5": ("Power-law backbone + XGBoost residual", lambda: XGB_(["logRe", "logWe", "spacing", "depth"], PowerBackbone)),
    "M6": ("Laan backbone + GPR residual (SEM descriptors)", lambda: GPR(FINAL, LaanBackbone)),
    "GPR": ("GPR, SEM descriptors", lambda: GPR(FINAL)),
}


def status(stage, state, note=""):
    p = f"{OUT}/status.json"
    s = json.load(open(p)) if os.path.exists(p) else {}
    s[stage] = {"state": state, "note": note, "time": time.strftime("%H:%M:%S")}
    json.dump(s, open(p, "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] {stage}: {state} {note}", flush=True)


def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))


def run_model(key, t, r):
    """ID, LOSO and REF-H predictions (+ std for GP models). Cached to disk."""
    f = f"{OUT}/pred_{key}.npz"
    if os.path.exists(f): return dict(np.load(f))
    make = MODELS[key][1]
    full = make().fit(t)
    init = full.m.kernel_ if isinstance(full, GPR) else None
    def mk():
        m = make()
        if isinstance(m, GPR): m.init = init
        return m
    gp = isinstance(full, GPR)
    out = {k: np.zeros(len(t)) for k in ["id", "loso", "loso_mu", "loso_sd"]}
    for tr, te in GroupKFold(5).split(t, t.y, t.cond):
        out["id"][te] = np.exp(mk().fit(t.iloc[tr]).predict(t.iloc[te]))
    for s in sorted(t["sample"].unique()):
        te = (t["sample"] == s).values
        m = mk().fit(t[~te])
        if gp:
            mu, sd = m.predict(t[te], std=True); out["loso_mu"][te], out["loso_sd"][te] = mu, sd
        else:
            mu = m.predict(t[te]); out["loso_mu"][te] = mu
        out["loso"][te] = np.exp(mu)
    if gp:
        mu, sd = full.predict(r, std=True); out["ref_mu"], out["ref_sd"] = mu, sd
    else:
        mu = full.predict(r); out["ref_mu"], out["ref_sd"] = mu, np.zeros(len(r))
    out["ref"] = np.exp(mu)
    if gp:
        out["ard"] = np.atleast_1d(full.m.kernel_.k1.k2.length_scale)
    np.savez(f, **out)
    return out


def grouped_boot_diff(y, pa, pb, groups, n=2000):
    """Paired bootstrap of RMSE(pa) - RMSE(pb), resampling whole groups. For LOSO the group is the surface
    (errors are correlated within a held-out surface); for REF-H it is the replicate condition."""
    rng = np.random.default_rng(SEED)
    g = pd.Series(groups).astype("category").cat.codes.values
    idx_by = [np.where(g == k)[0] for k in np.unique(g)]
    d = []
    for _ in range(n):
        idx = np.concatenate([idx_by[k] for k in rng.integers(0, len(idx_by), len(idx_by))])
        d.append(rmse(pa[idx], y[idx]) - rmse(pb[idx], y[idx]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float(np.mean(np.array(d) < 0))


def per_surface(t, p):
    return {s: rmse(p[(t["sample"] == s).values], t.beta.values[(t["sample"] == s).values]) for s in sorted(t["sample"].unique())}


# ---------------------------------------------------------------- stages
def p34(t, r, P):
    status("p34", "running", "M5/M6 vs M2 on REF-H")
    y, yr = t.beta.values, r.beta.values
    rows = {}
    for k in ["M2", "M5", "M6", "GPR"]:
        p = P[k]
        rows[k] = {"name": MODELS[k][0], "ID": rmse(p["id"], y), "LOSO": rmse(p["loso"], y),
                   "REFH": rmse(p["ref"], yr), "REFH_bias": float(np.mean(p["ref"] - yr)),
                   "LOSO_per_surface": per_surface(t, p["loso"])}
    for k in ["M5", "M6", "GPR"]:
        ci, pneg = grouped_boot_diff(yr, P[k]["ref"], P["M2"]["ref"], r.cond.values)
        rows[k]["REFH_minus_M2"] = rows[k]["REFH"] - rows["M2"]["REFH"]
        rows[k]["REFH_minus_M2_CI95"] = ci
        rows[k]["P(better than M2)"] = pneg
        lci, lp = grouped_boot_diff(y, P[k]["loso"], P["M2"]["loso"], t["sample"].values)
        rows[k]["LOSO_minus_M2"] = rows[k]["LOSO"] - rows["M2"]["LOSO"]
        rows[k]["LOSO_minus_M2_CI95"] = lci
    # does the Laan backbone help the GP? (M6 vs plain GPR)
    ci, pneg = grouped_boot_diff(yr, P["M6"]["ref"], P["GPR"]["ref"], r.cond.values)
    rows["M6_vs_GPR_REFH_CI95"] = ci
    lci, _ = grouped_boot_diff(y, P["M6"]["loso"], P["GPR"]["loso"], t["sample"].values)
    rows["M6_vs_GPR_LOSO_CI95"] = lci
    rows["M6_wins_surfaces"] = int(sum(rmse(P["M6"]["loso"][m], y[m]) < rmse(P["GPR"]["loso"][m], y[m])
                                       for m in [(t["sample"] == s).values for s in t["sample"].unique()]))
    json.dump(rows, open(f"{OUT}/p34.json", "w"), indent=1)
    status("p34", "done", f"M6 REF-H {rows['M6']['REFH']:.4f} vs M2 {rows['M2']['REFH']:.4f}")
    return rows


def p5(t, r, P):
    status("p5", "running", "conformal + OOD")
    out = {}
    yl, yr = np.log(t.beta.values), np.log(r.beta.values)
    for k in ["GPR", "M6"]:
        p = P[k]
        z = np.abs(yl - p["loso_mu"]) / p["loso_sd"]
        lev = lambda n: min(1.0, np.ceil((n + 1) * 0.9) / n)   # split-conformal finite-sample level
        q = float(np.quantile(z, lev(len(z))))                  # all 12 surfaces -> used for REF-H
        zr = np.abs(yr - p["ref_mu"]) / p["ref_sd"]
        # honest LOSO coverage: q for surface s is calibrated on the other 11 surfaces only
        cs = {}
        for s in sorted(t["sample"].unique()):
            m = (t["sample"] == s).values
            cs[s] = float(np.mean(z[m] <= np.quantile(z[~m], lev((~m).sum()))))
        out[k] = {"q_conformal": q,
                  "cov_nominal_LOSO": float(np.mean(z <= Z90)),
                  "cov_conformal_LOSO": float(np.average(list(cs.values()), weights=[(t["sample"] == s).sum() for s in cs])),
                  "cov_nominal_REFH": float(np.mean(zr <= Z90)), "cov_conformal_REFH": float(np.mean(zr <= q)),
                  "width_REFH_conformal": float(np.mean(np.exp(p["ref_mu"] + q * p["ref_sd"]) - np.exp(p["ref_mu"] - q * p["ref_sd"]))),
                  "cov_conformal_per_surface": cs,
                  "min_surface_coverage": min(cs.values()), "worst_surface": min(cs, key=cs.get)}
        # absolute (not normalised) conformal: error does not track GP std under shift
        a = np.abs(yl - p["loso_mu"]); qa = float(np.quantile(a, lev(len(a))))
        out[k]["abs_conformal_halfwidth_log"] = qa
        out[k]["cov_abs_conformal_REFH"] = float(np.mean(np.abs(yr - p["ref_mu"]) <= qa))
        # wide band: largest per-surface 90th percentile of |log error| (a worst-surface band, not a conformal quantile)
        per = [np.quantile(a[(t["sample"] == s).values], 0.9) for s in sorted(t["sample"].unique())]
        qs = float(np.max(per)); out[k]["surface_conformal_halfwidth_log"] = qs
        out[k]["cov_surface_conformal_REFH"] = float(np.mean(np.abs(yr - p["ref_mu"]) <= qs))
    # OOD: per held-out surface, fraction flagged vs model error
    F = FINAL; rows = []
    for s in sorted(t["sample"].unique()):
        te = (t["sample"] == s).values
        sc = OODDetector().fit(t.loc[~te, F].values).score(t.loc[te, F].values)
        rows.append({"surface": s, "flagged": float((sc.max(axis=1) >= 95).mean()),
                     "iso_med": float(sc.iso_pct.median()), "mah_med": float(sc.mah_pct.median()),
                     "rmse": rmse(P["GPR"]["loso"][te], t.beta.values[te])})
    sc = OODDetector().fit(t[F].values).score(r[F].values)
    rows.append({"surface": "REF-H", "flagged": float((sc.max(axis=1) >= 95).mean()),
                 "iso_med": float(sc.iso_pct.median()), "mah_med": float(sc.mah_pct.median()),
                 "rmse": rmse(P["GPR"]["ref"], r.beta.values)})
    O = pd.DataFrame(rows); tex = O[O.surface != "REF-H"]
    rho, pv = spearmanr(tex.flagged, tex.rmse)
    out["ood"] = rows
    out["ood_spearman_flag_vs_rmse"] = {"rho": float(rho), "p": float(pv), "n": int(len(tex))}
    json.dump(out, open(f"{OUT}/p5.json", "w"), indent=1)
    status("p5", "done", f"LOSO cov {out['GPR']['cov_conformal_LOSO']:.0%}, REF-H cov {out['GPR']['cov_conformal_REFH']:.0%}")
    return out


def p6(t, r, P):
    import shap
    key = json.load(open(f"{OUT}/p7.json"))["final"]
    status("p6", "running", f"SHAP on final model {key}")
    m = MODELS[key][1]().fit(t)
    def f(X):   # full log-beta prediction (backbone + GP) as a function of the model features only
        d = pd.DataFrame(X, columns=FINAL)
        Re, We = np.exp(d.logRe), np.exp(d.logWe)
        d["Re"], d["We"], d["P"] = Re, We, We * Re ** (-0.4)
        return m.predict(d)
    rng = np.random.default_rng(SEED)
    bg = shap.kmeans(t[FINAL].values, 20)
    ex = shap.KernelExplainer(f, bg)
    idx = rng.choice(len(t), 250, replace=False)
    sv = ex.shap_values(t[FINAL].values[idx], nsamples=200, silent=True)
    svr = ex.shap_values(r[FINAL].values, nsamples=200, silent=True)
    imp = dict(zip(FINAL, np.abs(sv).mean(0).tolist()))
    impr = dict(zip(FINAL, np.abs(svr).mean(0).tolist()))
    ard = dict(zip(FINAL, np.atleast_1d(m.m.kernel_.k1.k2.length_scale).tolist()))
    np.savez(f"{OUT}/shap.npz", sv=sv, X=t[FINAL].values[idx], svr=svr, Xr=r[FINAL].values)
    out = {"mean_abs_shap_textured": imp, "mean_abs_shap_REFH": impr, "ard_length_scales": ard,
           "model": key, "note": "SHAP on the full log(beta) prediction (incl. backbone, if the model has one); KernelExplainer, 20 k-means background points, 250 textured rows + all 125 REF-H rows"}
    json.dump(out, open(f"{OUT}/p6.json", "w"), indent=1)
    top = max(imp, key=imp.get)
    status("p6", "done", f"top feature {top}")
    return out


def p7(t, r, P):
    status("p7", "running", "benchmark table")
    y, yr = t.beta.values, r.beta.values
    rows = []
    for k, (name, _) in MODELS.items():
        p = P[k]; ps = per_surface(t, p["loso"]); w = max(ps, key=ps.get)
        rows.append({"id": k, "model": name, "ID": rmse(p["id"], y), "LOSO": rmse(p["loso"], y),
                     "LOSO_worst": f"{ps[w]:.3f} ({w})", "REFH": rmse(p["ref"], yr),
                     "REFH_MAPE": float(np.mean(np.abs(p["ref"] - yr) / yr) * 100),
                     "REFH_bias": float(np.mean(p["ref"] - yr))})
    B = pd.DataFrame(rows).sort_values("LOSO")
    # decision rule: lowest LOSO; tie (overlapping CI) broken by simplicity (fewer moving parts)
    best, second = B.iloc[0], B.iloc[1]
    ci, _ = grouped_boot_diff(y, P[best.id]["loso"], P[second.id]["loso"], t["sample"].values)
    tie = ci[0] < 0 < ci[1]
    simpler = {"M6": "GPR"}          # M6 = GPR + backbone; on a tie keep the one without the backbone
    final = simpler.get(best.id, best.id) if tie and simpler.get(best.id) == second.id else best.id
    B.to_csv(f"{OUT}/benchmark.csv", index=False)
    out = {"table": B.to_dict("records"), "lowest_LOSO": best.id, "runner_up": second.id, "best_minus_runner_up_LOSO_CI95": ci,
           "tie": bool(tie), "final": final,
           "rule": "lowest LOSO RMSE; if the LOSO difference to the runner-up has a surface-bootstrap CI that includes 0, keep the simpler model (no backbone)"}
    json.dump(out, open(f"{OUT}/p7.json", "w"), indent=1)
    status("p7", "done", f"final = {final}")
    return out


if __name__ == "__main__":
    t, r = build()
    P = {}
    for k in MODELS:
        status(f"fit_{k}", "running", MODELS[k][0])
        t0 = time.time(); P[k] = run_model(k, t, r)
        status(f"fit_{k}", "done", f"{time.time() - t0:.0f}s")
    if len(sys.argv) > 1 and sys.argv[1] == "p6": p6(t, r, P)
    else: p34(t, r, P); p5(t, r, P); p7(t, r, P); p6(t, r, P)
    status("all", "done")
