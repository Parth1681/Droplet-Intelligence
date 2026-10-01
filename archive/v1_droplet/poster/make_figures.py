"""Poster figures, drawn at print scale: a 250 mm-wide figure is placed 1:1 on the A0 sheet, so font sizes are real pt."""
import json, glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image
import sys
sys.path.insert(0, "archive/v1_droplet")
from src.data import load

for f in glob.glob("/root/.fonts/Inter-*.ttf"): fm.fontManager.addfont(f)
INK, INK2, MUTED, GRID = "#14213d", "#4a5268", "#8a90a0", "#e3e6ec"
NAVY, BLUE, ORANGE, AQUA = "#2f4574", "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]       # ordinal blue, glycerol 0 -> 91 wt.%
plt.rcParams.update({"font.family": "Inter", "font.size": 27, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.2, "axes.axisbelow": True,
                     "svg.fonttype": "path", "axes.titleweight": "semibold", "axes.titlesize": 24,
                     "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.linewidth": 1.4,
                     "xtick.major.width": 1.4, "ytick.major.width": 1.4})
W = 250 / 25.4          # 250 mm in inches
OUT = "archive/v1_droplet/poster/assets"
R = "archive/v1_droplet/results"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.svg", transparent=True, bbox_inches="tight", pad_inches=0.25); plt.close(fig)


# ---------- SEM panels ----------
SEM = "archive/v1_droplet/data/raw/06 - SEM images of the test surfaces/"
def sem_crop(name, box, size):
    im = Image.open(SEM + name + ".tif").crop(box); im = im.resize(size, Image.LANCZOS); im.save(f"{OUT}/{name}.jpg", quality=92)
sem_crop("D200_43", (300, 200, 1500, 1400), (1200, 1200))
sem_crop("REF-H_43", (300, 200, 1500, 1400), (1200, 1200))
sem_crop("D200_350", (200, 100, 2400, 1880), (2200, 1780))
sem_crop("S800_43", (300, 200, 1500, 1400), (1200, 1200))

# ---------- zoo results ----------
rows = []
for f in glob.glob(f"{R}/zoo/*.json"): rows += json.load(open(f))
Z = pd.DataFrame(rows)
short = {"Gaussian process (Matern 3/2, ARD)": "Gaussian process", "Random forest": "Random forest",
         "Extra trees": "Extra trees", "FT-Transformer (d=32, 2 layers)": "FT-Transformer",
         "Deep ensemble (5 x MLP)": "Deep ensemble (5 MLP)", "CatBoost": "CatBoost",
         "XGBoost (reference, same features)": "XGBoost", "LightGBM": "LightGBM",
         "Ridge, quadratic in log-features": "Quadratic ridge", "MLP (64-64, strong L2)": "MLP",
         "kNN (k=10, distance-weighted)": "kNN", "SVR (RBF)": "SVR (RBF)", "Kernel ridge (RBF)": "Kernel ridge"}
Z["name"] = Z.Model.map(short); Z = Z.sort_values("LOSO RMSE").reset_index(drop=True)
Z.to_csv(f"{R}/zoo_table.csv", index=False)

# Fig: LOSO RMSE dot plot (single measure, one axis) -- the model-selection criterion
fig, ax = plt.subplots(figsize=(W, 9.6))
y = np.arange(len(Z))[::-1]
best = Z.name == "Gaussian process"
ax.hlines(y, 0, Z["LOSO RMSE"], color=GRID, lw=3)
ax.scatter(Z["LOSO RMSE"], y, s=220, color=np.where(best, NAVY, "#9aa3b5"), zorder=3, edgecolor="white", lw=2)
for yi, v, b in zip(y, Z["LOSO RMSE"], best):
    ax.text(v + 0.006, yi, f"{v:.3f}", va="center", fontsize=25, color=INK if b else INK2, weight="bold" if b else "normal")
ax.set_yticks(y, Z.name); ax.set_xlim(0, 0.30); ax.grid(axis="y", visible=False)
ax.get_yticklabels()[0].set_fontweight("bold"); ax.get_yticklabels()[0].set_color(INK)
ax.set_xlabel(r"RMSE of $\beta_{max}$ on the unseen surface (12 folds)")
ax.tick_params(axis="y", length=0)
save(fig, "fig_loso")

# Fig: encoding sensitivity (range per model) -- before vs after the phi descriptor
before = [("XGBoost, raw inputs", 0.0588, 0.0721), ("XGBoost, Re/We/Oh", 0.0624, 0.0847), ("XGBoost, hybrid", 0.0611, 0.0826)]
after = []
for n in ["Gaussian process", "XGBoost", "CatBoost"]:
    lo, hi = Z.loc[Z.name == n, "REF-H RMSE over w +/-10um"].iloc[0]; after.append((n, lo, hi))
fig, ax = plt.subplots(figsize=(W, 6.0))
labs = []
for i, (n, lo, hi) in enumerate(before + after):
    yy = len(before + after) - 1 - i + (0 if i < 3 else -0.6)
    c = ORANGE if i < 3 else NAVY
    ax.plot([lo, hi], [yy, yy], color=c, lw=10, solid_capstyle="round", alpha=0.9)
    ax.text(hi + 0.0015, yy, f"spread {hi - lo:.3f}", va="center", fontsize=23, color=INK2)
    labs.append((yy, n))
ax.set_yticks([l[0] for l in labs], [l[1] for l in labs]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0.055, 0.105); ax.set_ylim(-1.2, 5.6); ax.grid(axis="y", visible=False)
ax.text(0.0552, 5.45, "Before: smooth plate encoded as arbitrary spacing/depth", color=ORANGE, fontsize=23, weight="semibold")
ax.text(0.0552, 1.85, "After: φ = 1 by definition; track width w ± 10 µm", color=NAVY, fontsize=23, weight="semibold")
ax.set_xlabel("REF-H RMSE across encoding choices")
save(fig, "fig_encoding")

# Fig: REF-H parity for the GPR (the blind test), coloured by glycerol level (ordinal ramp)
t, r = load()
p = np.load(f"{R}/zoo/gaussian_process_matern_3_2_ard__ref.npy")
fig, ax = plt.subplots(figsize=(W * 0.58, W * 0.58))
lims = [1.35, 3.0]
ax.plot(lims, lims, color=MUTED, lw=1.6, ls="--", zorder=1)
for g, c in zip([0, 20, 60, 78, 91], RAMP):
    m = r.gly == g
    ax.scatter(r.beta[m], p[m], s=90, color=c, edgecolor="white", lw=1.2, zorder=3, label=f"{g}")
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
ax.set_xlabel(r"measured $\beta_{max}$"); ax.set_ylabel(r"predicted $\beta_{max}$")
leg = ax.legend(title="glycerol wt.%", fontsize=21, title_fontsize=21, frameon=False, loc="lower right",
                handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25, markerscale=1.2)
save(fig, "fig_parity")

# Fig: CNN descriptor recovery (true vs predicted phi), LOSO + REF-H blind
C = json.load(open(f"{R}/cnn_descriptors.json"))
fig, ax = plt.subplots(figsize=(W * 0.58, W * 0.58))
ax.plot([0, 1.05], [0, 1.05], color=MUTED, lw=1.6, ls="--")
for k, v in C.items():
    tr, pr = v["true"][0], v["pred"][0]
    if k == "REF-H":
        ax.scatter(tr, pr, s=260, marker="D", color=ORANGE, edgecolor="white", lw=1.5, zorder=4)
        ax.annotate("REF-H\n(never seen)", (tr, pr), (0.62, 0.42), fontsize=22, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.4))
    else:
        c = NAVY if k[0] == "D" else BLUE
        ax.scatter(tr, pr, s=150, color=c, marker="o" if k[0] == "D" else "s", edgecolor="white", lw=1.2, zorder=3)
ax.scatter([], [], s=150, color=NAVY, label="deep (held out)"); ax.scatter([], [], s=150, color=BLUE, marker="s", label="shallow (held out)")
ax.legend(fontsize=22, frameon=False, loc="lower right", handletextpad=0.1)
ax.set_xlim(-0.03, 1.05); ax.set_ylim(-0.03, 1.05); ax.set_aspect("equal")
ax.set_xlabel("φ from geometry"); ax.set_ylabel("φ read by CNN")
save(fig, "fig_cnn")

# Fig: OOD score vs error per held-out surface
O = pd.read_csv(f"{R}/ood_scores.csv").set_index("surface")
E = json.load(open(f"{R}/cnn_pipeline_eval.json"))
fig, ax = plt.subplots(figsize=(W * 0.58, W * 0.58))
for s in O.index:
    if s == "REF-H": continue
    x, yv = O.loc[s, "frac_flagged(>95)"] * 100, E[s]["rmse_measured_desc"]
    hot = x > 50
    ax.scatter(x, yv, s=150, color=ORANGE if hot else "#9aa3b5", edgecolor="white", lw=1.2, zorder=3)
    if hot:
        ax.annotate(s, (x, yv), (x - 22 if x > 60 else x + 4, yv + 0.0015), fontsize=23, color=INK)
xr, yr = O.loc["REF-H", "frac_flagged(>95)"] * 100, E["REF-H"]["rmse_measured_desc"]
ax.scatter(xr, yr, s=260, marker="D", color=NAVY, edgecolor="white", lw=1.5, zorder=4)
ax.annotate("REF-H", (xr, yr), (xr + 5, yr + 0.002), fontsize=23, color=INK)
ax.set_xlim(-6, 106); ax.set_ylim(0.02, 0.07)
ax.set_xlabel("% impacts flagged OOD"); ax.set_ylabel("GPR RMSE")
save(fig, "fig_ood")
print("figures written")
