"""Poster figures, drawn 1:1 for an A0 column (250 mm) from saved result files."""
from pathlib import Path as _P; _ROOT = _P(__file__).resolve().parents[2]  # repo root (droplet-intelligence/)
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ['fonts/ibm-plex-sans-latin-400-normal.ttf', 'fonts/ibm-plex-sans-latin-500-normal.ttf', 'fonts/ibm-plex-sans-latin-600-normal.ttf']:
    fm.fontManager.addfont(f)
V = str(_ROOT/'Droplet_Intelligence_v2')
N = json.load(open(str(_ROOT/'paper/numbers.json')))
INK, INK2, MUTED, GRID = '#13294b', '#4a5268', '#8a90a0', '#e3e6ec'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': ['IBM Plex Sans', 'DejaVu Sans'], 'font.size': 22, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 1.2, 'axes.axisbelow': True, 'axes.linewidth': 1.4, 'svg.fonttype': 'path',
    'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'legend.frameon': False, 'mathtext.fontset': 'dejavusans'})
MM = 1 / 25.4
def save(fig, name):
    fig.savefig(name + '.svg', transparent=True, bbox_inches='tight', pad_inches=0.15)
    fig.savefig('check_' + name + '.png', dpi=60, facecolor='white', bbox_inches='tight'); plt.close(fig)

# a) 13 learners, unseen-surface RMSE
z = pd.DataFrame(N['zoo']).sort_values('loso', ascending=False)
fig, ax = plt.subplots(figsize=(185 * MM, 136 * MM)); y = np.arange(len(z))
for yi, (_, r) in zip(y, z.iterrows()):
    gp = r['name'] == 'Gaussian process'
    ax.hlines(yi, 0, r.loso, color=BLUE if gp else GRID, lw=5 if gp else 3.5, zorder=1)
    ax.plot(r.loso, yi, 'o', ms=15 if gp else 11, color=BLUE if gp else '#8793ab', mec='white', mew=2, zorder=3)
    ax.text(r.loso + 0.004, yi, f'{r.loso:.3f}', va='center', fontsize=19, color=INK if gp else INK2, weight=600 if gp else 400)
ax.set_yticks(y, z['name']); ax.tick_params(axis='y', length=0)
for t in ax.get_yticklabels():
    if t.get_text() == 'Gaussian process': t.set_color(INK); t.set_fontweight(600)
ax.grid(axis='y', visible=False); ax.set_xlim(0, 0.29); ax.set_xlabel('RMSE on the held-out surface')
save(fig, 'fig_models')

# b) what did not help: paired surface-bootstrap differences vs baseline GP
rows = N['forest']
fig, ax = plt.subplots(figsize=(150 * MM, 120 * MM)); y = np.arange(len(rows))[::-1]
for yi, (n, d, ci) in zip(y, rows):
    sig = ci[0] > 0 or ci[1] < 0
    ax.plot([ci[0] * 1e3, ci[1] * 1e3], [yi, yi], color=INK2, lw=3, solid_capstyle='round')
    ax.plot(d * 1e3, yi, 'o', ms=14, color=ORANGE if sig else BLUE, mec='white', mew=2, zorder=3)
ax.axvline(0, color=MUTED, lw=1.6, ls='--')
ax.set_yticks(y, [r[0] for r in rows]); ax.tick_params(axis='y', length=0); ax.grid(axis='y', visible=False)
ax.set_xlim(-6, 29); ax.set_xlabel('Δ RMSE vs baseline GP (×10⁻³)')
ax.text(-5.6, len(rows) - 0.25, '← better', fontsize=18, color=INK2); ax.text(28.6, len(rows) - 0.25, 'worse →', fontsize=18, color=INK2, ha='right')
save(fig, 'fig_forest')

# c) interval coverage by fluid
S = N['by_fluid']; x = np.arange(len(S)); w = 0.36
fig, ax = plt.subplots(figsize=(232 * MM, 118 * MM))
for j, (lab, c, k) in enumerate([('Unseen textured surface', BLUE, 'loso_cov'), ('Smooth plate REF-H', ORANGE, 'refh_cov')]):
    v = [s[k] for s in S]; b = ax.bar(x + (j - .5) * w, v, w * .9, color=c, label=lab, zorder=2)
    for xi, vi in zip(x + (j - .5) * w, v): ax.text(xi, vi - 0.03, f'{vi:.0%}', ha='center', va='top', fontsize=16.5, color='white', weight=600)
ax.axhline(.9, color=INK, lw=1.8, ls='--', zorder=3); ax.text(4.6, 1.035, '- - 90% target', fontsize=17, color=INK, va='center', ha='right')
ax.set_xticks(x, [f"{s['gly']}%" for s in S]); ax.set_xlabel('Glycerol (wt%)   ·   viscosity 0.9 → 151 mPa·s')
ax.set_ylim(0, 1.12); ax.set_yticks([0, .25, .5, .75, 1]); ax.yaxis.set_major_formatter(lambda v, p: f'{v:.0%}')
ax.set_ylabel('inside 90% interval'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=18, handlelength=1.1)
save(fig, 'fig_fluid')

# d) REF-H parity, coloured by glycerol
d = pd.read_csv(f'{V}/data/experiments.csv'); r = d[d.split == 'REF-H'].reset_index(drop=True)
rf = pd.read_csv(f'{V}/results/refh_predictions.csv'); rf = rf[rf.model == 'baseline'].reset_index(drop=True); rf['gly'] = r.glycerol.values
cm = plt.get_cmap('Blues'); glys = [0, 20, 60, 78, 91]
fig, ax = plt.subplots(figsize=(150 * MM, 146 * MM))
ax.plot([1.35, 3.0], [1.35, 3.0], color=MUTED, lw=1.6, ls='--')
for i, g in enumerate(glys):
    m = rf.gly == g; ax.scatter(rf.beta[m], rf.prediction[m], s=70, color=cm(0.35 + 0.6 * i / 4), edgecolor='white', linewidth=0.8, label=f'{g}', zorder=3)
ax.set_xlim(1.35, 3.0); ax.set_ylim(1.35, 3.0); ax.set_aspect('equal')
ax.set_xlabel(r'measured $\beta_{\rm max}$'); ax.set_ylabel(r'predicted $\beta_{\rm max}$')
ax.legend(title='glycerol wt%', title_fontsize=15.5, fontsize=15.5, loc='lower right', handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
save(fig, 'fig_parity')

# e) v2.1 CNN: phi read from SEM images of a never-seen surface vs geometry phi
fd = json.load(open(f'{V}/results/sem/summary.json'))['fold_descriptors']
geo = {s['surface']: s['phi'] for s in N['surfaces']}
refh_cnn = json.load(open(f'{V}/models/sem_gp.json'))['surface_descriptors']['REF-H']['phi']
fig, ax = plt.subplots(figsize=(122 * MM, 118 * MM))
ax.plot([0, 1.05], [0, 1.05], color=MUTED, lw=1.6, ls='--')
for s, v in fd.items():
    deep = s.startswith('D')
    ax.plot(geo[s], v[0], 'o' if deep else 's', ms=13, color=INK if deep else BLUE, mec='white', mew=1.6, zorder=3)
ax.plot(1.0, refh_cnn, 'D', ms=15, color=ORANGE, mec='white', mew=1.6, zorder=4)
ax.annotate('REF-H read\nas φ ≈ %.2f' % refh_cnn, (1.0, refh_cnn), (0.52, 0.2), fontsize=17, color=ORANGE,
            arrowprops=dict(arrowstyle='-', color=ORANGE, lw=1.4))
ax.plot([], [], 'o', ms=12, color=INK, label='deep (held out)'); ax.plot([], [], 's', ms=12, color=BLUE, label='shallow (held out)')
ax.legend(loc='upper left', fontsize=16, handletextpad=0.1, borderaxespad=0.1)
ax.set_xlim(-0.03, 1.07); ax.set_ylim(-0.03, 1.07); ax.set_aspect('equal')
ax.set_xlabel('φ from geometry'); ax.set_ylabel('φ read by CNN')
save(fig, 'fig_cnn')
print('ok', refh_cnn)

# f) per held-out surface
ab = json.load(open(str(_ROOT/'paper/ablation.json'))); lg = json.load(open(f'{V}/results/legacy_gpr_vs_xgb.json'))
bm = {m['model']: m for m in json.load(open(f'{V}/results/benchmark.json'))['metrics']}
order = ['S50', 'S100', 'S200', 'S400', 'S600', 'S800', 'D50', 'D100', 'D200', 'D400', 'D600', 'D800']
ser = [('GP', bm['baseline']['per_surface'], BLUE, 'o'), (r'GP without $\varphi$, $V_{\rm tex}$', ab['blind3']['per_surface'], AQUA, 's'), ('XGBoost', lg['XGB']['per_surface'], ORANGE, 'D')]
fig, ax = plt.subplots(figsize=(236 * MM, 100 * MM)); x = np.arange(12)
for i, (n, dd, c, mk) in enumerate(ser):
    ax.plot(x + (i - 1) * 0.2, [dd[s] for s in order], mk, ms=11, color=c, mec='white', mew=1.4, label=n, zorder=3)
ax.axvline(5.5, color=MUTED, lw=1.4)
ax.text(2.5, 0.123, 'shallow · 6 µm', ha='center', fontsize=17, color=INK2); ax.text(8.5, 0.123, 'deep · 25 µm', ha='center', fontsize=17, color=INK2)
ax.set_xticks(x, [s[1:] for s in order], fontsize=17); ax.set_xlabel('channel pitch (µm) of the held-out surface'); ax.grid(axis='x', visible=False)
ax.set_ylim(0, 0.13); ax.set_yticks([0, .04, .08, .12]); ax.set_ylabel('RMSE')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=17, handletextpad=0.2, columnspacing=1.2)
save(fig, 'fig_surface')
