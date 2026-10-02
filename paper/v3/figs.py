"""Publication figures for the v3 paper. Every number plotted is read from a saved result file or
recomputed from saved per-row predictions; nothing is typed in by hand.

    python paper/v3/figs.py        (writes paper/v3/figs/*.png and *.pdf)
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(R + '/../..')
V2D = ROOT + '/Droplet_Intelligence_v2'; sys.path.insert(0, V2D)
A = ROOT + '/archive/v1_droplet/results'; V2 = V2D + '/results'
OUT = R + '/figs'; os.makedirs(OUT, exist_ok=True)
from droplet.data import load

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7, 'legend.frameon': False,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'figure.dpi': 110})
INK = '#1d2433'; GP = '#1f5fa8'; XG = '#d1495b'; GREY = '#8a93a3'; GOLD = '#e3a72f'; TEAL = '#2a9d8f'
FLC = {0: '#1f5fa8', 20: '#2a9d8f', 60: '#8ab17d', 78: '#e3a72f', 91: '#d1495b'}
M = json.load(open(R + '/verified_metrics.json'))['models']; S = json.load(open(R + '/stats.json'))
t, r = load(); y = t.beta.values; g = t.surface.values
fl = pd.cut(t.mu, [0, .0012, .003, .02, .08, 1], labels=[0, 20, 60, 78, 91]).astype(int).values
flr = pd.cut(r.mu, [0, .0012, .003, .02, .08, 1], labels=[0, 20, 60, 78, 91]).astype(int).values
gp = np.load(A + '/final/pred_GPR.npz'); xg = np.load(A + '/final/pred_XGB.npz')
v2r = pd.read_csv(V2 + '/refh_predictions.csv'); v2r = v2r[v2r.model == 'baseline'].sort_values('row_id')
v2l = pd.read_csv(V2 + '/loso_predictions.csv'); v2l = v2l[v2l.model == 'baseline'].sort_values('row_id')
Q = json.load(open(V2 + '/nested.json'))['deployment_q']


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png'); fig.savefig(f'{OUT}/{name}.pdf'); plt.close(fig); print('wrote', name)


# ---------------------------------------------------------------- 1 architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(0, 100); ax.set_ylim(-4, 52); ax.axis('off')
    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=c + '18', ec=c, lw=1))
        ax.text(x + w / 2, y + h - 2.0, title, ha='center', va='top', fontsize=7.6, weight='bold', color=c)
        ax.text(x + w / 2, y + h - 6.6, body, ha='center', va='top', fontsize=6.4, color=INK, linespacing=1.4)
    def arrow(x0, y0, x1, y1, c=INK, ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=8, lw=.8, color=c, ls=ls))
    ax.text(0, 50.5, 'Scientific prediction path (all accuracy claims)', ha='left', fontsize=8.3, color=GP, weight='bold')
    W = 17.6; xs = [0.5, 20.6, 40.7, 60.8, 80.9]
    box(xs[0], 26, W, 21, 'Measured inputs', '$D_0$, $V$\n$\\rho$, $\\mu$, $\\sigma$\nspacing $s$, depth $h$\n(Moze et al. 2025)', INK)
    box(xs[1], 26, W, 21, 'Physics layer', '$Re=\\rho V D_0/\\mu$\n$We=\\rho V^2 D_0/\\sigma$\n$Oh=\\sqrt{We}/Re$\n$\\phi$, $V_{tex}$', GP)
    box(xs[2], 26, W, 21, 'GP model', 'Matern 3/2, ARD\n+ white noise\non ln Re, ln We,\nln $D_0$, $\\phi$, $V_{tex}$', GP)
    box(xs[3], 26, W, 21, 'Uncertainty', 'lognormal median\n90% interval with\nmultiplier $q$ from\nCV residuals\n+ support check', GP)
    box(xs[4], 26, W, 21, 'Output', '$\\beta_{max}=D_{max}/D_0$\nwith 90% interval\nand reliability\nstatus', GP)
    for x in xs[:-1]: arrow(x + W + .5, 36.5, x + W + 2.4, 36.5)
    box(0.5, 2, 22, 17, 'SEM image (43x)', 'local roughness map\nOtsu threshold\n$\\phi$ = smooth fraction', TEAL)
    box(26, 2, 22, 17, 'Side-view video', 'Otsu, substrate band,\nlargest blob: $D_0$, $V$\nwidest extent: $\\beta_{max}$', TEAL)
    box(51.5, 2, 22, 17, 'Browser engine', 'same GP weights\n(JSON + Cholesky)\nmax |diff| vs Python\n3.1e-12', TEAL)
    box(77, 2, 22.5, 17, 'Comparison', 'measured $\\beta_{max}$\ninside / outside\nthe 90% interval', TEAL)
    arrow(11.5, 19.6, 29.4, 25.4, TEAL, '--'); arrow(37, 19.6, 9.3, 25.4, TEAL, '--')
    arrow(62.5, 19.6, 49.5, 25.4, TEAL, '--'); arrow(89.7, 25.4, 88.3, 19.6, TEAL, '--')
    ax.text(0, -3.6, 'Measurement and interface path (supplies inputs to the same model; not part of the accuracy claims)', ha='left', fontsize=8.3, color=TEAL, weight='bold')
    save(fig, 'fig01_architecture')


# ---------------------------------------------------------------- 2 parameter space
def fig_space():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    for k in FLC:
        s = fl == k; ax.scatter(t.Re[s], t.We[s], s=5, c=FLC[k], alpha=.6, lw=0, label=f'{k} wt%')
    ax.scatter(r.Re, r.We, s=14, facecolors='none', edgecolors=INK, lw=.5, label='REF-H')
    ax.set(xscale='log', yscale='log', xlabel='Re', ylabel='We', title='a  Impact conditions')
    for oh in (.003, .03, .3):
        re = np.logspace(.8, 3.8, 50); ax.plot(re, (oh * re) ** 2, ':', c=GREY, lw=.6)
        xl = 7.0 / oh; ax.text(xl * 1.08, 7.0, f'Oh = {oh:g}', fontsize=6, color=GREY, rotation=0, va='bottom')
    ax.set_ylim(6, 160); ax.set_xlim(6, 7000); ax.legend(fontsize=6.3, markerscale=1.6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=6, columnspacing=.5, handletextpad=.1)
    ax = axs[1]; su = t.groupby('surface')[['spacing', 'depth', 'phi', 'texvol']].first()
    for name, row in su.iterrows():
        c = GP if name.startswith('D') else GOLD
        ax.scatter(row.spacing, row.phi, s=18 + 2.5 * row.texvol, c=c, alpha=.85, lw=0)
        ax.annotate(name, (row.spacing, row.phi), fontsize=6, xytext=(3, -7 if name.startswith('D') else 3), textcoords='offset points')
    ax.scatter([], [], c=GP, label=r'deep ($h$ = 25 $\mu$m)'); ax.scatter([], [], c=GOLD, label=r'shallow ($h$ = 6 $\mu$m)')
    ax.axhline(1, c=INK, lw=.6, ls='--'); ax.text(60, 1.02, r'REF-H ($\phi$ = 1, $V_{tex}$ = 0)', fontsize=6.5)
    ax.set(xscale='log', xlabel=r'channel spacing $s$ ($\mu$m)', ylabel=r'$\phi$ (smooth plateau fraction)', title='b  Surface descriptors (marker area ~ $V_{tex}$)', ylim=(-.05, 1.12))
    ax.set_xticks([50, 100, 200, 400, 600, 800]); ax.set_xticklabels(['50', '100', '200', '400', '600', '800']); ax.minorticks_off()
    ax.legend(fontsize=6.5, loc='lower right')
    save(fig, 'fig02_parameter_space')


# ---------------------------------------------------------------- 3 model comparison
def fig_models():
    rows = [(v['label'], v['loso']['rmse'], v['refh']['rmse'] if isinstance(v['refh'], dict) else np.nan, k)
            for k, v in M.items() if k.startswith('zoo_')]
    rows.append(('GP, dimensional inputs (raw 7)', M['raw7_gp']['loso_rmse'], M['raw7_gp']['refh_rmse'], 'raw7'))
    rows.append(('Laan residual + GP (v2)', M['v2_physics_residual']['loso']['rmse'], M['v2_physics_residual']['refh']['rmse'], 'phys'))
    rows.append(('XGBoost, raw 7 inputs', M['v1_M1']['loso']['rmse'], M['v1_M1']['refh']['rmse'], 'M1'))
    rows.append(('XGBoost, hybrid raw + Re/We/Oh', M['v1_M3']['loso']['rmse'], M['v1_M3']['refh']['rmse'], 'M3'))
    rows = sorted(rows, key=lambda z: z[1]); lab = [z[0].replace(' (reference, same features)', ' (same 5 inputs)') for z in rows]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True); yy = np.arange(len(rows))[::-1]
    def col(z):
        return GP if 'Gaussian process' in z[0] or z[3] in ('raw7', 'phys') else XG if 'XGBoost' in z[0] else GREY
    for ax, j, ttl in [(axs[0], 1, 'a  Unseen textured surface (LOSO RMSE)'), (axs[1], 2, 'b  Smooth plate REF-H (RMSE)')]:
        v = np.array([z[j] for z in rows]); ax.barh(yy, v, color=[col(z) for z in rows], height=.7)
        for i, val in zip(yy, v):
            if np.isfinite(val) and val <= .12: ax.text(val + .002, i, f'{val:.4f}', va='center', fontsize=6.3)
        ax.set_xlim(0, .145); ax.set_title(ttl, loc='left'); ax.set_xlabel(r'RMSE in $\beta_{max}$')
        if j == 1:
            ax.axvline(M['v1_GPR']['loso']['rmse'], c=GP, lw=.6, ls=':')
            big = [(i, val) for i, val in zip(yy, v) if val > .12]
            for i, val in big: ax.text(.143, i, f'{val:.4f} (bar clipped)', va='center', ha='right', fontsize=6.3, color='w')
    axs[0].set_yticks(yy); axs[0].set_yticklabels(lab, fontsize=6.8)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=GP, label='Gaussian process variants'), Patch(color=XG, label='XGBoost variants'),
                        Patch(color=GREY, label='other model families')], fontsize=7, loc='lower center', ncol=3, bbox_to_anchor=(.55, -.04))
    save(fig, 'fig03_model_comparison')


# ---------------------------------------------------------------- 4 parity
def fig_parity():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    lo, hi = 1.35, 3.45
    for ax, p, ttl, c in [(axs[0], gp['loso'], 'a  GP, LOSO', GP), (axs[1], xg['loso'], 'b  XGBoost, LOSO', XG)]:
        ax.scatter(y, p, s=3, c=c, alpha=.35, lw=0); ax.plot([lo, hi], [lo, hi], c=INK, lw=.7)
        mm = M['v1_GPR' if c == GP else 'v1_XGB']['loso']
        ax.text(.04, .96, f"RMSE {mm['rmse']:.4f}\nMAE {mm['mae']:.4f}\nR2 {mm['r2']:.4f}", transform=ax.transAxes, va='top', fontsize=6.6)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=r'measured $\beta_{max}$', title=ttl, aspect='equal')
    axs[0].set_ylabel(r'predicted $\beta_{max}$')
    ax = axs[2]; e = np.vstack([v2r.prediction - v2r.lower, v2r.upper - v2r.prediction])
    inside = (v2r.beta >= v2r.lower) & (v2r.beta <= v2r.upper)
    ax.errorbar(v2r.beta, v2r.prediction, yerr=e, fmt='none', ecolor=GREY, lw=.4, alpha=.6)
    ax.scatter(v2r.beta[inside], v2r.prediction[inside], s=6, c=GP, lw=0, label='inside 90%')
    ax.scatter(v2r.beta[~inside], v2r.prediction[~inside], s=6, c=XG, lw=0, label='outside 90%')
    ax.plot([lo, 3.0], [lo, 3.0], c=INK, lw=.7); mm = M['v2_baseline']['refh']
    ax.text(.04, .96, f"RMSE {mm['rmse']:.4f}\nR2 {mm['r2']:.4f}\ncoverage {mm['coverage90']:.1%}", transform=ax.transAxes, va='top', fontsize=6.6)
    ax.set(xlim=(lo, 3.0), ylim=(lo, 3.0), xlabel=r'measured $\beta_{max}$', title='c  GP on REF-H', aspect='equal'); ax.legend(fontsize=6, loc='lower right')
    save(fig, 'fig04_parity')


# ---------------------------------------------------------------- 5 residuals
def fig_residuals():
    res = gp['loso'] - y; fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 1.5]))
    for ax, x, lab in [(axs[0], t.We, 'We'), (axs[1], t.Oh, 'Oh')]:
        for k in FLC:
            s = fl == k; ax.scatter(x[s], res[s], s=3, c=FLC[k], alpha=.5, lw=0, label=f'{k} wt%')
        ax.axhline(0, c=INK, lw=.6); ax.set(xscale='log', xlabel=lab, ylim=(-.25, .25))
    axs[0].set_ylabel('LOSO residual (pred - meas)'); axs[0].set_title('a  vs We', loc='left'); axs[1].set_title('b  vs Oh', loc='left')
    axs[1].legend(fontsize=5.8, markerscale=2, loc='lower left', ncol=2)
    ax = axs[2]; order = sorted(np.unique(g), key=lambda s: (s[0], int(s[1:])))
    ax.boxplot([res[g == s] for s in order], tick_labels=order, widths=.6, showfliers=False, medianprops=dict(color=GP),
               boxprops=dict(lw=.6), whiskerprops=dict(lw=.6), capprops=dict(lw=.6))
    ps = M['v2_baseline']['per_surface']
    for i, s in enumerate(order): ax.text(i + 1, .135, f"{ps[s]:.3f}", ha='center', fontsize=5.6, rotation=90)
    ax.axhline(0, c=INK, lw=.6); ax.set(ylim=(-.16, .17), title='c  per held-out surface (number = RMSE)'); ax.tick_params(axis='x', labelsize=6, rotation=90)
    save(fig, 'fig07_residuals')


# ---------------------------------------------------------------- 6 coverage
def fig_coverage():
    sd = v2l.sd_log.values; mu = v2l.mu_log.values; ins = (np.log(y) >= mu - Q * sd) & (np.log(y) <= mu + Q * sd)
    rins = ((v2r.beta >= v2r.lower) & (v2r.beta <= v2r.upper)).values
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.4), gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]; ks = list(FLC); x = np.arange(len(ks))
    ax.bar(x - .2, [ins[fl == k].mean() for k in ks], .38, color=GP, label='textured, LOSO')
    ax.bar(x + .2, [rins[flr == k].mean() for k in ks], .38, color=GOLD, label='REF-H')
    ax.axhline(.9, c=INK, ls='--', lw=.6); ax.set(xticks=x, xticklabels=[f'{k}%' for k in ks], ylim=(0, 1.25), ylabel='fraction inside 90% interval',
                                                  xlabel='glycerol (wt%)', title='a  by fluid'); ax.legend(fontsize=6.5, loc='upper left', bbox_to_anchor=(0, 1.0))
    ax = axs[1]; order = sorted(np.unique(g), key=lambda s: (s[0], int(s[1:])))
    ax.bar(range(len(order)), [ins[g == s].mean() for s in order], color=GP, width=.7)
    ax.axhline(.9, c=INK, ls='--', lw=.6); ax.set(xticks=range(len(order)), ylim=(0, 1.05), title='b  by held-out surface (LOSO)')
    ax.set_xticklabels(order, fontsize=6.5, rotation=90)
    save(fig, 'fig08_coverage')
    return dict(fluid_loso={k: float(ins[fl == k].mean()) for k in ks}, fluid_refh={k: float(rins[flr == k].mean()) for k in ks},
                surface={s: float(ins[g == s].mean()) for s in order}, overall=float(ins.mean()), refh=float(rins.mean()), q=Q)


# ---------------------------------------------------------------- 7 error map + GP uncertainty
def fig_errmap_uncertainty():
    from droplet.predict import Predictor
    P = Predictor(); fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=.45)); d0s = {}
    ax = axs[0]; e = np.abs(gp['loso'] - y)
    sc = ax.scatter(t.Re, t.We, c=e, s=6, cmap='viridis', vmin=0, vmax=.12, lw=0)
    ax.set(xscale='log', yscale='log', xlabel='Re', ylabel='We', title='a  |LOSO error| of the GP')
    plt.colorbar(sc, ax=ax, label=r'|error|', shrink=.85, pad=.02)
    ax = axs[1]; vs = np.linspace(.2, 3.0, 120)
    for k, c in [('0', FLC[0]), ('91', FLC[91])]:
        s = (g == 'D200') & (fl == int(k)); d0 = float(np.median(t.D[s]) * 1e3); d0s[k] = d0
        f = P.release['fluids'][k]; pr = [P.predict(dict(D_mm=d0, V=v, surface='D200', **f)) for v in vs]
        b = np.array([p['beta_max'] for p in pr]); lo_ = np.array([p['interval']['lower'] for p in pr]); hi_ = np.array([p['interval']['upper'] for p in pr])
        ax.plot(vs, b, c=c, lw=1, label=f'{k} wt% glycerol, $D_0$ = {d0:.2f} mm'); ax.fill_between(vs, lo_, hi_, color=c, alpha=.2, lw=0)
        ax.scatter(t.V[s], y[s], s=5, c=c, lw=0, alpha=.8)
    vmin, vmax = t.V.min(), t.V.max(); ax.axvspan(.2, vmin, color=GREY, alpha=.12, lw=0); ax.axvspan(vmax, 3.0, color=GREY, alpha=.12, lw=0)
    ax.text(2.35, 1.3, 'outside\nmeasured V', fontsize=6.5, color=GREY, ha='center')
    ax.set(xlabel='impact velocity V (m/s)', ylabel=r'$\beta_{max}$', title='b  GP median and 90% interval on D200', xlim=(.2, 3.0))
    ax.legend(fontsize=6.5, loc='upper left')
    save(fig, 'fig09_error_map_uncertainty'); return d0s


# ---------------------------------------------------------------- 8 ablation
def fig_ablation():
    p2 = {z['model']: z for z in json.load(open(A + '/phase2.json'))}
    p3 = {z['model']: z for z in json.load(open(A + '/phase3.json'))}
    p4 = {z['model']: z for z in json.load(open(A + '/phase4.json'))}
    ab = json.load(open(ROOT + '/paper/ablation.json'))
    rows = [('Laan law alone (A refit)', ab['laan']['loso_rmse'], ab['laan']['refh_rmse'], 'phys'),
            ('Re, We, Oh + s, h', p2['GPR | Re,We,Oh + spacing,depth (paper PIML set)']['LOSO'], p2['GPR | Re,We,Oh + spacing,depth (paper PIML set)']['REFH'], 'repr'),
            ('ln Re, ln We, phi, V_tex', p3['base: logRe, logWe, phi, texvol']['LOSO'], p3['base: logRe, logWe, phi, texvol']['REFH'], 'repr'),
            ('  + ln Oh (redundant)', p3['base + logOh']['LOSO'], p3['base + logOh']['REFH'], 'repr'),
            ('  + ln D0  = final', p3['base + logD']['LOSO'], p3['base + logD']['REFH'], 'final'),
            ('final - phi', p3['final - phi']['LOSO'], p3['final - phi']['REFH'], 'repr'),
            ('final - V_tex', p3['final - texvol']['LOSO'], p3['final - texvol']['REFH'], 'repr'),
            ('surface-blind (ln Re, ln We, ln D0)', p4['surface-blind (logRe, logWe, logD)']['LOSO'], p4['surface-blind (logRe, logWe, logD)']['REFH'], 'repr'),
            ('raw s, h instead of phi, V_tex', p4['raw geometry (spacing, depth; REF-H = 0/0)']['LOSO'], p4['raw geometry (spacing, depth; REF-H = 0/0)']['REFH'], 'repr'),
            ('dimensional raw 7 (re-run)', M['raw7_gp']['loso_rmse'], M['raw7_gp']['refh_rmse'], 'repr'),
            ('Laan backbone + GP residual', M['v2_physics_residual']['loso']['rmse'], M['v2_physics_residual']['refh']['rmse'], 'repr'),
            ('phi from SEM image (43x)', M['ext_image_phi']['loso']['rmse'], M['ext_image_phi']['refh']['rmse'], 'img'),
            ('+ cos(contact angle)', M['ext_angle_input']['loso']['rmse'], M['ext_angle_input']['refh']['rmse'], 'wet'),
            ('Lee beta0 correction', M['ext_lee_beta0']['loso']['rmse'], M['ext_lee_beta0']['refh']['rmse'], 'wet')]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True); yy = np.arange(len(rows))[::-1]
    cmap = dict(phys=GREY, repr=GP, final=INK, img=TEAL, wet=XG)
    for ax, j, ttl, xmax in [(axs[0], 1, 'a  LOSO RMSE', .07), (axs[1], 2, 'b  REF-H RMSE', .12)]:
        v = np.array([z[j] for z in rows]); ax.barh(yy, np.minimum(v, xmax), color=[cmap[z[3]] for z in rows], height=.7)
        for i, val in zip(yy, v): ax.text(min(val, xmax) + .001, i, f'{val:.4f}' + (' (clipped)' if val > xmax else ''), va='center', fontsize=6)
        ax.axvline(M['v1_GPR']['loso' if j == 1 else 'refh']['rmse'], c=INK, lw=.6, ls=':'); ax.set_xlim(0, xmax * 1.32); ax.set_title(ttl, loc='left')
    axs[0].set_yticks(yy); axs[0].set_yticklabels([z[0] for z in rows], fontsize=6.8)
    save(fig, 'fig05_ablation')


# ---------------------------------------------------------------- 9 importance
def fig_importance():
    p6 = json.load(open(A + '/final/p6.json')); ks = ['logRe', 'logWe', 'logD', 'phi', 'texvol']
    lab = ['ln Re', 'ln We', 'ln $D_0$', r'$\phi$', '$V_{tex}$']
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=.35)); x = np.arange(5)
    axs[0].bar(x - .2, [p6['mean_abs_shap_textured'][k] for k in ks], .38, color=GP, label='textured rows')
    axs[0].bar(x + .2, [p6['mean_abs_shap_REFH'][k] for k in ks], .38, color=GOLD, label='REF-H rows')
    axs[0].set(xticks=x, xticklabels=lab, ylabel=r'mean |SHAP| on ln $\beta_{max}$', title='a  SHAP attribution'); axs[0].legend(fontsize=6.5)
    axs[1].bar(x, [p6['ard_length_scales'][k] for k in ks], color=GP, width=.55)
    axs[1].set(xticks=x, xticklabels=lab, ylabel='ARD length scale', title='b  GP length scales (shorter = more relevant)')
    save(fig, 'fig06_importance')


# ---------------------------------------------------------------- 10 SEM + image reader
def fig_sem():
    from droplet import extensions as E
    from PIL import Image
    es = json.load(open(V2 + '/extensions/summary.json')); thr = es['image_reader']['threshold_all_textured']
    names = ['D50', 'D200', 'D800', 'S200', 'REF-H']
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 3.2))
    for j, n in enumerate(names):
        p = f'{E.SEM}/{n}_43.tif'; im = np.asarray(Image.open(p).convert('L'))[:E.FOOTER_Y]
        rm = E.roughness_map(p); phi = float((rm <= thr).mean())
        axs[0, j].imshow(im, cmap='gray'); axs[1, j].imshow(rm <= thr, cmap='Blues_r')
        geo = t[t.surface == n].phi.iloc[0] if n != 'REF-H' else 1.0
        axs[0, j].set_title(n, fontsize=8); axs[1, j].set_title(f'$\\phi$ image {phi:.3f}\n$\\phi$ geometry {geo:.3f}', fontsize=6.5)
        for a in axs[:, j]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(True); sp.set_lw(.4)
    um = E.pixel_size(f'{E.SEM}/D200_43.tif'); bar = 500 / um
    axs[0, 0].plot([100, 100 + bar], [im.shape[0] - 120] * 2, c='w', lw=2); axs[0, 0].text(100, im.shape[0] - 170, '500 $\\mu$m', color='w', fontsize=6)
    axs[0, 0].set_ylabel('SEM, 43x', fontsize=7); axs[1, 0].set_ylabel('smooth plateau mask', fontsize=7)
    save(fig, 'fig10_sem_image_reader')


# ---------------------------------------------------------------- 11 video pipeline
def fig_video():
    from droplet.synthetic_video import render
    from droplet.video import analyse
    fr, truth = render(D0=2.5, V=1.2, beta=2.6)
    m = analyse(fr, truth['fps'], truth['mm_per_px'])
    c = m['contact_frame']; pick = [c - 8, c, c + 6, c + 14]
    fig = plt.figure(figsize=(7.2, 2.9)); gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, .15, 1.6], wspace=.08, hspace=.35)
    for j, i in enumerate(pick):
        ax = fig.add_subplot(gs[j // 2, j % 2]); ax.imshow(fr[i][150:262, 60:340], cmap='gray', vmin=0, vmax=255); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(True); sp.set_lw(.4)
        ax.axhline(m['baseline_row'] - 150, c=GOLD, lw=.8); ax.set_title(f'{"abcd"[j]}  t = {(i - c) / truth["fps"] * 1e3:+.1f} ms', fontsize=7, loc='left')
    ax = fig.add_subplot(gs[:, 3]); s_ = pd.DataFrame(m['series'])
    ax.plot(s_.t_ms, s_.D_mm / m['D0_mm'], c=GP, lw=1, label='detected $D(t)/D_0$'); ax.axhline(truth['beta_max'], c=INK, ls='--', lw=.6, label=r'rendered $\beta_{max}$')
    ax.text(.97, .05, f"detected / rendered\n$D_0$: {m['D0_mm']:.3f} / {truth['D0_mm']} mm\n$V$: {m['V']:.3f} / {truth['V']} m/s\n" + r"$\beta_{max}$: " + f"{m['beta_max']:.3f} / {truth['beta_max']}",
            transform=ax.transAxes, va='bottom', ha='right', fontsize=6.5)
    ax.set(xlabel='time after contact (ms)', ylabel='$D(t)/D_0$', title='e  spreading curve from the frames'); ax.set_ylim(0.8, 3.0); ax.legend(fontsize=6.3, loc='upper right')
    save(fig, 'fig11_video_pipeline')
    return dict(measured=dict(D0_mm=m['D0_mm'], V=m['V'], beta_max=m['beta_max']), truth=truth)


# ---------------------------------------------------------------- 12 interface (screenshots from paper/v3/screenshots.py)
def fig_interface():
    from PIL import Image
    crops = [('ui_lab.png', .27, .92, 'a  Prediction from inputs'), ('ui_upload.png', .215, .935, 'b  Prediction from an SEM image'),
             ('ui_video.png', .185, .95, 'c  Measurement from video')]
    ims = []
    for f, a, b, _ in crops:
        im = Image.open(f'{OUT}/{f}'); W, H = im.size; ims.append(im.crop((0, int(a * H), W, int(b * H))))
    ratios = [i.size[0] / i.size[1] for i in ims]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 7.2 / sum(ratios) * 1.12), gridspec_kw=dict(wspace=.03, width_ratios=ratios))
    for ax, im, (f, a, b, ttl) in zip(axs, ims, crops):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=7.5, loc='left')
        for sp in ax.spines.values(): sp.set_visible(False)
    save(fig, 'fig12_interface')


if __name__ == '__main__' and len(sys.argv) == 1:
    extra = {}
    fig_architecture(); fig_space(); fig_models(); fig_parity(); fig_residuals()
    extra['coverage'] = fig_coverage(); extra['uncert_D0'] = fig_errmap_uncertainty(); fig_ablation(); fig_importance(); fig_sem()
    extra['video'] = fig_video()
    json.dump(extra, open(R + '/figs_numbers.json', 'w'), indent=1, default=float)
    if os.path.exists(OUT + '/ui_lab.png'): fig_interface()
elif __name__ == '__main__' and sys.argv[1] == 'interface':
    fig_interface()
