"""GPR vs XGBoost on identical inputs (logRe, logWe, logD, phi, texvol), same three protocols.
Writes results/final/pred_XGB.npz, results/final/gpr_vs_xgb.json and poster/assets/fig_gpr_vs_xgb.{svg,png}."""
import json, os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.final_pipeline as fp
from src.phases import build, XGB_, FINAL

fp.MODELS["XGB"] = ("XGBoost, SEM descriptors (same inputs as GPR)", lambda: XGB_(FINAL))
t, r = build(); y, yr = t.beta.values, r.beta.values
G, X = fp.run_model("GPR", t, r), fp.run_model("XGB", t, r)
rmse = fp.rmse


def boot_ci(p, yy, groups, n=2000):
    rng = np.random.default_rng(0)
    g = pd.Series(groups).astype("category").cat.codes.values
    idx = [np.where(g == k)[0] for k in np.unique(g)]
    v = [rmse(p[i], yy[i]) for i in (np.concatenate([idx[k] for k in rng.integers(0, len(idx), len(idx))]) for _ in range(n))]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {}
for key, P in [("GPR", G), ("XGB", X)]:
    out[key] = {"ID": rmse(P["id"], y), "ID_CI": boot_ci(P["id"], y, t.cond.values),
                "LOSO": rmse(P["loso"], y), "LOSO_CI": boot_ci(P["loso"], y, t["sample"].values),
                "REFH": rmse(P["ref"], yr), "REFH_CI": boot_ci(P["ref"], yr, r.cond.values),
                "per_surface": fp.per_surface(t, P["loso"])}
out["LOSO_GPR_minus_XGB_CI"], _ = fp.grouped_boot_diff(y, G["loso"], X["loso"], t["sample"].values)
out["REFH_GPR_minus_XGB_CI"], _ = fp.grouped_boot_diff(yr, G["ref"], X["ref"], r.cond.values)
out["ID_GPR_minus_XGB_CI"], _ = fp.grouped_boot_diff(y, G["id"], X["id"], t.cond.values)
out["GPR_wins_surfaces"] = int(sum(out["GPR"]["per_surface"][s] < out["XGB"]["per_surface"][s] for s in out["GPR"]["per_surface"]))
ae_g, ae_x = np.abs(G["loso"] - y), np.abs(X["loso"] - y)
out["LOSO_within_0.05"] = {"GPR": float(np.mean(ae_g <= 0.05)), "XGB": float(np.mean(ae_x <= 0.05))}
out["LOSO_p95_abs_err"] = {"GPR": float(np.percentile(ae_g, 95)), "XGB": float(np.percentile(ae_x, 95))}
json.dump(out, open("results/final/gpr_vs_xgb.json", "w"), indent=1)

# ------------------------------------------------------------------ figure (poster style, 1:1 at A0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in glob.glob("/root/.fonts/Inter-*.ttf"): fm.fontManager.addfont(f)
INK, INK2, MUTED, GRID = "#14213d", "#4a5268", "#8a90a0", "#e3e6ec"
NAVY, ORANGE = "#2f4574", "#eb6834"
plt.rcParams.update({"font.family": "Inter", "font.size": 24, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2, "axes.axisbelow": True,
                     "svg.fonttype": "path", "axes.titleweight": "semibold", "axes.titlesize": 24,
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.linewidth": 1.4})
W = 500 / 25.4
fig = plt.figure(figsize=(W, W * 0.42))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.0], wspace=0.38)

# A: RMSE with 95% CI under the three protocols
ax = fig.add_subplot(gs[0])
labs = [("ID", "Seen\nsurfaces"), ("LOSO", "Unseen\nsurface"), ("REFH", "Smooth\nplate")]
xx = np.arange(3); w = 0.36
for j, (key, c) in enumerate([("GPR", NAVY), ("XGB", ORANGE)]):
    v = [out[key][k] for k, _ in labs]; ci = [out[key][k + "_CI"] for k, _ in labs]
    xs = xx + (j - 0.5) * w
    ax.bar(xs, v, w * 0.92, color=c, label="Gaussian process" if key == "GPR" else "XGBoost", zorder=2)
    ax.errorbar(xs, v, yerr=[[a - b[0] for a, b in zip(v, ci)], [b[1] - a for a, b in zip(v, ci)]],
                fmt="none", ecolor=INK, elinewidth=1.8, capsize=5, zorder=3)
    for xi, vi in zip(xs, v): ax.text(xi, 0.0015, f"{vi:.3f}", ha="center", va="bottom", rotation=90, color="white", fontsize=19, weight="semibold", zorder=4)
ax.set_xticks(xx, [l for _, l in labs], fontsize=20); ax.grid(axis="x", visible=False)
ax.set_ylabel(r"RMSE of $\beta_{max}$"); ax.set_ylim(0, 0.085)
ax.set_title("A  RMSE with 95% CI")
ax.legend(frameon=False, fontsize=20, loc="upper left", handlelength=1.0)

# B: per-surface LOSO dumbbell
ax = fig.add_subplot(gs[1])
ps = pd.DataFrame({"GPR": out["GPR"]["per_surface"], "XGB": out["XGB"]["per_surface"]}).sort_values("XGB")
yy = np.arange(len(ps))
ax.hlines(yy, ps.GPR, ps.XGB, color=GRID, lw=5, zorder=1)
ax.scatter(ps.XGB, yy, s=170, color=ORANGE, zorder=3, edgecolor="white", lw=1.5, label="XGBoost")
ax.scatter(ps.GPR, yy, s=170, color=NAVY, zorder=4, edgecolor="white", lw=1.5, label="Gaussian process")
ax.set_yticks(yy, ps.index, fontsize=20); ax.tick_params(axis="y", length=0); ax.grid(axis="y", visible=False)
ax.set_xlabel(r"RMSE when the surface is held out"); ax.set_xlim(0.02, 0.125)
ax.set_title(f"B  Per surface: GPR lower on {out['GPR_wins_surfaces']}/12")

# C: cumulative share of impacts within an absolute error, unseen surfaces
ax = fig.add_subplot(gs[2])
e = np.linspace(0, 0.2, 400)
for a, c, n in [(ae_g, NAVY, "Gaussian process"), (ae_x, ORANGE, "XGBoost")]:
    ax.plot(e, [(a <= v).mean() for v in e], color=c, lw=3.5, label=n)
ax.axvline(0.05, color=MUTED, ls="--", lw=1.4)
ax.text(0.053, 0.08, f"within 0.05:\nGPR {out['LOSO_within_0.05']['GPR']:.0%}\nXGB {out['LOSO_within_0.05']['XGB']:.0%}", fontsize=19, color=INK2)
ax.set_xlim(0, 0.2); ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_xlabel(r"absolute error in $\beta_{max}$"); ax.set_ylabel("share of impacts")
ax.set_title("C  Every impact, unseen surface")

os.makedirs("poster/assets", exist_ok=True)
fig.savefig("poster/assets/fig_gpr_vs_xgb.svg", transparent=True, bbox_inches="tight", pad_inches=0.25)
fig.savefig("poster/assets/fig_gpr_vs_xgb.png", dpi=150, facecolor="white", bbox_inches="tight", pad_inches=0.25)
print(json.dumps({k: v for k, v in out.items() if k not in ("GPR", "XGB")}, indent=1))
for k in ("GPR", "XGB"): print(k, {m: round(out[k][m], 4) for m in ("ID", "LOSO", "REFH")}, {m: [round(x, 4) for x in out[k][m + "_CI"]] for m in ("ID", "LOSO", "REFH")})
