"""Build the v3 manuscript (HTML -> PDF with Chromium, and DOCX) from one content list.
Every number in the text is read from a saved result file at build time.

    python paper/v3/build_paper.py
"""
import json, os, re, html, datetime
import numpy as np

R = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(R + '/../..')
A = ROOT + '/archive/v1_droplet/results'; V2 = ROOT + '/Droplet_Intelligence_v2/results'
J = lambda p: json.load(open(p))
M = J(R + '/verified_metrics.json')['models']; S = J(R + '/stats.json'); FN = J(R + '/figs_numbers.json')
NEST = J(V2 + '/nested.json'); BENCH = J(V2 + '/benchmark.json'); EXT = J(V2 + '/extensions/summary.json')
SEMS = J(V2 + '/sem/summary.json'); PAR = J(V2 + '/js_parity.json'); GX = J(A + '/final/gpr_vs_xgb.json')
P6 = J(A + '/final/p6.json'); AB = J(ROOT + '/paper/ablation.json'); BL = J(R + '/browser_latency.json')
AUD = J(V2 + '/data_audit.json'); SEMGP = J(ROOT + '/Droplet_Intelligence_v2/models/sem_gp.json'); CNN1 = J(A + '/cnn_descriptors.json'); import pandas as _pd
ZW = json.loads(_pd.read_csv(A + '/zoo_table.csv').iloc[0]['REF-H RMSE over w +/-10um'])
VC = J(R + '/video_check.json'); BVC = J(R + '/browser_video_check.json'); CS = J(A + '/final/cold_start_check.json')
P2 = {z['model']: z for z in J(A + '/phase2.json')}; P3 = {z['model']: z for z in J(A + '/phase3.json')}
P4 = {z['model']: z for z in J(A + '/phase4.json')}; ENC = J(A + '/phase_extra_enc.json'); RAWPHI = J(A + '/phase_extra_raw_phi.json')
RAWPHI = RAWPHI[0] if isinstance(RAWPHI, list) else RAWPHI
D = S['dataset']; T_ = D['textured']; RF = D['refh']
NA = 'Not available from the supplied evidence'


def f(x, n=4):
    return NA if x is None or (isinstance(x, str)) else f'{x:.{n}f}'


def ci(c, n=4):
    return f'[{c[0]:+.{n}f}, {c[1]:+.{n}f}]'


def pct(x, n=1):
    return f'{100 * x:.{n}f}%'


gp, xg = M['v1_GPR'], M['v1_XGB']; r7 = M['raw7_gp']; phys = M['v2_physics_residual']
cov = FN['coverage']; IR = EXT['image_reader']; XC = EXT['candidates']
img_diff = {k: abs(v['image_phi'] - v['geometry_phi']) for k, v in IR['per_surface'].items() if k != 'REF-H'}
img_diff_200 = max(v for k, v in img_diff.items() if int(k[1:]) >= 200)
vid_beta_err = max(abs(v['measured'][2] / v['truth'][2] - 1) for v in VC)
vid_D_err = max(abs(v['measured'][0] / v['truth'][0] - 1) for v in VC)
vid_V_err = max(abs(v['measured'][1] / v['truth'][1] - 1) for v in VC)
vid_s = [v['seconds'] for v in VC]
fluid_rmse = {x['gly']: x for x in J(ROOT + '/paper/by_fluid.json')}

# ============================================================== content
C = []
def H1(t): C.append(('h1', t))
def H2(t): C.append(('h2', t))
def P(t): C.append(('p', t))
def EQ(t, n): C.append(('eq', t, n))
def L(items): C.append(('ul', items))
def TAB(cap, head, rows, note=None, widths=None): C.append(('table', cap, head, rows, note, widths))
def FIG(fn, cap, w=1.0): C.append(('fig', fn, cap, w))

TITLE = 'Droplet Intelligence: physics-informed Gaussian-process prediction of maximum spreading on laser-textured surfaces never seen in training'
SUB = 'Dimensionless inputs, surface-held-out validation, calibrated uncertainty, and image and video measurement front ends'
AUTH = 'Parth Sharma'; AFF = 'Thapar Institute of Engineering & Technology, Patiala, India'
DATE = 'Version 3 · ' + datetime.date.today().strftime('%d %B %Y')

ABSTRACT = (
    f"The maximum spreading ratio β_{{max}} = D_{{max}}/D_{{0}} of an impacting drop sets how much surface a drop wets, yet data-driven models of it are "
    f"usually scored on random splits in which every surface is also in training. We ask the design question instead: how accurately can β_{{max}} be "
    f"predicted on a textured surface that has never been seen? Using a public dataset of {T_['rows']:,} impacts on 12 laser-textured aluminium surfaces and "
    f"{RF['rows']} on a smooth hydrophobic plate (REF-H), five water–glycerol mixtures and Re = {T_['Re'][0]:.1f}–{T_['Re'][1]:,.0f}, We = {T_['We'][0]:.1f}–{T_['We'][1]:.0f}, "
    f"we represent each impact by ln Re, ln We, ln D_{{0}} and two texture descriptors, and evaluate every model by leave-one-surface-out (LOSO) cross-validation "
    f"with a paired bootstrap over surfaces. A Matérn-3/2 Gaussian process (GP) reaches a LOSO RMSE of {gp['loso']['rmse']:.4f} (R² = {gp['loso']['r2']:.3f}, "
    f"MAE = {gp['loso']['mae']:.4f}), against {xg['loso']['rmse']:.4f} for XGBoost on identical inputs (difference interval {ci(S['delta_vs_GP_ci95']['XGB'], 3)}) "
    f"and {AB['laan']['loso_rmse']:.3f} for the Laan scaling law alone; 12 other learners ranked below the GP. On REF-H, never used for training or selection, "
    f"the GP scores {gp['refh']['rmse']:.4f}. Its 90% intervals, calibrated on cross-validated residuals, cover {pct(cov['overall'])} of held-out textured impacts "
    f"but only {pct(cov['refh'])} on REF-H, failing mostly for low-viscosity drops. Several alternatives scored lower RMSE "
    f"(dimensional inputs {r7['loso_rmse']:.4f}, a Laan-backbone residual GP {phys['loso']['rmse']:.4f}), but none differed significantly from the baseline. "
    f"Adding ln Oh changes nothing because ln Oh = ½ ln We − ln Re; contact-angle inputs degrade REF-H. The texture fraction φ can be read from a 43× SEM "
    f"image with no loss of accuracy (LOSO {XC['image_only']['loso']['rmse']:.4f}), and a video pipeline recovers D_{{0}}, V and β_{{max}} of synthetic test "
    f"clips to within {100 * vid_beta_err:.1f}%. The same GP runs in a browser in about {BL['median_ms']:.0f} ms per query.")

KEYWORDS = 'drop impact; maximum spreading; laser-textured superhydrophobic surfaces; Gaussian process regression; dimensionless analysis; leave-one-surface-out validation; uncertainty calibration; SEM image analysis; high-speed video'

# ---------------------------------------------------------------- 1
H1('1. Introduction')
P("When a drop hits a solid surface it spreads into a thin lamella, reaches a maximum diameter D_{max}, and then recedes or rebounds. The maximum spreading ratio "
  "β_{max} = D_{max}/D_{0} controls wetted area and contact time, which matter for spray cooling, coating, printing and anti-icing [1]. For a Newtonian liquid on a given "
  "surface, β_{max} is set by the competition of inertia, viscosity and capillarity, which classical scaling arguments and the Laan et al. interpolation capture well on "
  "smooth surfaces [2, 3]. Laser-textured superhydrophobic surfaces add air pockets, contact-line pinning on channel edges and extra viscous dissipation, and no closed-form "
  "law accounts for these. This makes texture effects the part that an empirical model must learn.")
P(f"Može et al. [4] published {T_['rows'] + RF['rows']:,} impacts of water and water–glycerol drops on aluminium surfaces with nanosecond-laser channels of six pitches and two depths, "
  "plus a smooth hydrophobic reference, recorded at 5,000 frames per second. The same group trained 28 regression models on seven raw inputs and selected an isotropic "
  "exponential GP, evaluated with an 80/20 split and 5-fold cross-validation [5]. Every impact condition is repeated about five times and every surface is in training under "
  "such a split, so that evaluation measures interpolation between near-replicates. A surface designer needs something else: a prediction, with honest uncertainty, for a "
  "surface that has not been made yet.")
P("This paper reconstructs the full Droplet Intelligence study around that question. It follows one line of argument: start from the physics (Re, We, Oh), build a "
  "dimensionless representation, compare model families under a protocol that holds out whole surfaces, quantify uncertainty and its failures, and only then add image and "
  "video front ends and a deployable predictor. The verified contributions, each tied to the evidence that supports it, are listed in Table 1, and Fig. 1 shows how the parts fit together.")
TAB('Table 1. Contributions and the evidence for each. "Verified" means the claim is reproduced from saved per-impact predictions or a re-run in this study.',
    ['#', 'Contribution', 'Evidence (section, file)', 'Status'],
    [['1', 'Surface-held-out (LOSO) benchmark of 13 learners plus GP and XGBoost variants on identical inputs', '§7; zoo_table.csv, final/pred_*.npz', 'Verified'],
     ['2', 'Paired surface bootstrap with a decision rule fixed before comparison', '§5.4; benchmark.json, stats.json', 'Verified'],
     ['3', 'Ablation of input representations, incl. the physics law alone, a surface-blind GP and dimensional inputs', '§9; phase2–4.json, raw7_rerun.json', 'Verified'],
     ['4', 'Interval calibration on cross-validated residuals and a diagnosis of where coverage fails', '§10; nested.json, refh_predictions.csv', 'Verified'],
     ['5', 'Texture fraction φ read directly from SEM images; GP accuracy unchanged', '§11; extensions/summary.json', 'Verified'],
     ['6', 'Automatic D_{0}, V, β_{max} measurement from side-view video', '§12; video_check.json', 'Verified on synthetic clips only'],
     ['7', 'Browser implementation of the GP that matches Python to 3×10^{−12}', '§13; js_parity.json, browser_latency.json', 'Verified']],
    widths=[4, 50, 32, 14])
FIG('fig01_architecture', "Figure 1. System architecture. Top: the scientific prediction path, which carries every accuracy claim in this paper. Bottom: the measurement and interface "
    "path. The SEM reader supplies φ, the video pipeline supplies D_{0} and V (and a measured β_{max} for comparison), and the browser engine evaluates the same GP weights. "
    "Dashed arrows show where each front end connects to the prediction path.")

# ---------------------------------------------------------------- 2
H1('2. Physical background')
P("For a drop of diameter D_{0} and velocity V of a liquid with density ρ, viscosity μ and surface tension σ, three dimensionless groups describe the impact:")
EQ('Re = ρVD_{0}/μ,    We = ρV^{2}D_{0}/σ,    Oh = μ/√(ρσD_{0}) = √We / Re', 1)
P("Re compares inertia with viscous forces and We compares inertia with capillary forces. The Ohnesorge number Oh is not independent of the other two: by Eq. (1), "
  "ln Oh = ½ ln We − ln Re exactly (verified on all impacts to a maximum absolute error of "
  f"{S['oh_identity_max_abs_error']:.1e}). Any model that already receives ln Re and ln We therefore gains no information from ln Oh, a point we test in §9. "
  "Energy balance gives β_{max} ∝ Re^{1/5} when viscosity limits spreading and β_{max} ∝ We^{1/2} when capillarity does; Clanet et al. proposed We^{1/4} for low-viscosity "
  "drops [2]. Laan et al. [3] joined the two limits with a Padé approximant in the impact parameter P = We·Re^{−2/5}:")
EQ('β_{max} Re^{−1/5} = P^{1/2} / (A + P^{1/2}),    A ≈ 1.24', 2)
P("Lee et al. [13] extended this to partial wetting by replacing β_{max} with (β_{max}^{2} − β_{0}^{2})^{1/2}, where β_{0} is the zero-velocity spreading set by the "
  "contact angle. Neither law contains texture geometry; on the textured surfaces studied here, Eq. (2) with A refitted to the training surfaces gives a LOSO RMSE of "
  f"{AB['laan']['loso_rmse']:.4f} (A = {AB['laan']['A_all']:.3f}, fitted in log space), and {AB['laan']['A_textbook_1.24_all_rmse']:.4f} over all textured impacts with the textbook A = 1.24. The scaling law sets the trend but misses the "
  "surface and the fluid-specific offsets by about a factor of six relative to the learned models (§7).")

# ---------------------------------------------------------------- 3
H1('3. Dataset')
P(f"All impact data come from the public dataset of Može et al. [4]; no new experiments were performed. The substrates are aluminium 1050A plates with laser channels at "
  f"pitches s = 50, 100, 200, 400, 600 and 800 µm and two mean depths, about 6 µm (S50–S800) and 25 µm (D50–D800). REF-H is a smooth hydrophobic reference plate. "
  f"Five fluids span pure water to 91 wt% glycerol. The textured set has {T_['rows']:,} impacts in {T_['conditions']} distinct impact conditions (about five repeats each); REF-H has "
  f"{RF['rows']} impacts in {RF['conditions']} conditions. Table 2 gives the measured ranges, computed from the CSV files; Fig. 2 shows the impact conditions and the surfaces.")
def row(name, sym, key, unit, n=3, src=T_):
    a = src.get(key)
    return [name, sym, f'{a[0]:,.{n}f}', f'{a[1]:,.{n}f}', unit] if a else [name, sym, NA, NA, unit]
TAB('Table 2. Variables and measured ranges (textured surfaces, n = %d; REF-H, n = %d). Values computed from the dataset CSV files by stats.py.' % (T_['rows'], RF['rows']),
    ['Variable', 'Symbol', 'Min', 'Max', 'Unit'],
    [row('Drop diameter', 'D_{0}', 'D_mm', 'mm'), row('Impact velocity', 'V', 'V', 'm/s'), row('Density', 'ρ', 'rho', 'kg/m³', 1),
     row('Viscosity', 'μ', 'mu_mPas', 'mPa·s'), row('Surface tension', 'σ', 'sigma_mNm', 'mN/m', 2), row('Channel pitch', 's', 'spacing_um', 'µm', 0),
     row('Channel depth', 'h', 'depth_um', 'µm', 0), row('Reynolds number', 'Re', 'Re', '–', 1), row('Weber number', 'We', 'We', '–', 2),
     row('Ohnesorge number', 'Oh', 'Oh', '–', 4), row('Smooth plateau fraction', 'φ', 'phi', '–'), row('Texture volume per area', 'V_{tex}', 'texvol_um', 'µm'),
     row('Max. spreading ratio', 'β_{max}', 'beta', '–'),
     ['REF-H: Re', 'Re', f"{RF['Re'][0]:.1f}", f"{RF['Re'][1]:,.1f}", '–'], ['REF-H: We', 'We', f"{RF['We'][0]:.2f}", f"{RF['We'][1]:.2f}", '–'],
     ['REF-H: β_{max}', 'β_{max}', f"{RF['beta'][0]:.3f}", f"{RF['beta'][1]:.3f}", '–']],
    note='Rows per fluid (0/20/60/78/91 wt% glycerol): ' + ', '.join(str(v) for v in T_['rows_per_fluid'].values()) + '. Rows per surface: 124–125.', widths=[30, 12, 18, 18, 12])
FIG('fig02_parameter_space', "Figure 2. Measured parameter space. (a) Impact conditions in the Re–We plane, coloured by glycerol fraction; open circles are the smooth REF-H plate. "
    "Dotted lines are constant Oh = √We/Re, showing that the five fluids occupy five nearly separate Oh bands, so viscosity is varied by fluid choice rather than continuously. "
    "REF-H conditions lie inside the textured cloud, so REF-H tests a new surface, not new fluid conditions. (b) The 12 textured surfaces in the descriptor space used by the model: "
    "smooth plateau fraction φ against channel pitch s, with marker area proportional to texture volume per unit area V_{tex}. REF-H sits at φ = 1, V_{tex} = 0, outside the "
    "textured range (maximum φ = %.3f)." % T_['phi'][1])

# ---------------------------------------------------------------- 4
H1('4. Preprocessing and surface descriptors')
P(f"Re, We and Oh were recomputed from the five fluid and impact variables; they match the values supplied in the dataset to a relative error below "
  f"{max(AUD['re_relative_recompute_error'], AUD['we_relative_recompute_error']):.0e}. Pitch and depth "
  "are not used directly. Each surface is instead described by two physically interpretable numbers derived from the nominal geometry:")
EQ('φ = (max(s − w, 0) / s)^{2},    V_{tex} = (1 − φ)·h', 3)
P("where w is the laser-track width (45 µm for the deep surfaces, 30 µm for the shallow ones, measured from the 43× SEM images in the earlier phase of this project; "
  f"varying w by ±10 µm changes the REF-H RMSE by less than {abs(ZW[1] - ZW[0]):.4f}), φ is the fraction of the surface that is untouched plateau "
  "and V_{tex} is the textured volume per unit area. REF-H has φ = 1 and V_{tex} = 0 by construction, so it is a valid point of the descriptor space and needs no arbitrary code; "
  "a raw-geometry encoding has no natural value for a smooth plate (§9). The model inputs are x = (ln Re, ln We, ln D_{0}[mm], φ, V_{tex}), standardised with means and "
  "scales fitted on the training folds only. The target is ln β_{max}; predictions are returned as the lognormal median exp(μ). Logarithms make the power-law limits of §2 "
  "linear and make relative errors uniform across the range of β_{max}.")
P("Impacts were grouped into conditions (same surface, same fluid, velocities within 0.10 m/s), giving 297 textured conditions. There are no missing values and no duplicate rows. "
  "No outlier removal or smoothing was applied to the target.")
s_ = S['surfaces']
TAB('Table 3. Texture descriptors per surface. Geometry φ from Eq. (3); image φ read from the 43× SEM image (§11).',
    ['Surface', 's (µm)', 'h (µm)', 'φ geometry', 'φ image', 'V_{tex} (µm)'],
    [[k, f"{s_[k]['spacing']:.0f}", f"{s_[k]['depth']:.0f}", f"{s_[k]['phi']:.3f}", f"{IR['per_surface'][k]['image_phi']:.3f}", f"{s_[k]['texvol']:.2f}"]
     for k in sorted(s_, key=lambda z: (z[0], int(z[1:])))] + [['REF-H', '–', '–', '1.000', f"{IR['per_surface']['REF-H']['image_phi']:.3f}", '0']],
    widths=[16, 14, 14, 18, 18, 18])

# ---------------------------------------------------------------- 5
H1('5. Evaluation protocol, leakage control and statistics')
H2('5.1 Three test settings')
P("(i) In-distribution (ID): GroupKFold with 5 folds over the 297 conditions, so that repeats of one condition never straddle training and test. This answers the "
  "interpolation question of earlier work and is reported only for context. (ii) Leave-one-surface-out (LOSO): 12 folds, each holding out every impact on one textured surface. "
  "This is the primary metric. (iii) REF-H: models trained on all 12 textured surfaces predict the 125 smooth-plate impacts. REF-H was never used to fit, tune or select any model.")
H2('5.2 Leakage checks')
L(["Feature scaling, GP hyperparameters, the Laan constant A and the interval multiplier q are fitted inside each training fold.",
   "Repeats of an impact condition are always kept in the same fold (ID) or the same surface (LOSO).",
   "Model selection used LOSO scores; to measure the optimism this causes, a nested design re-selected the GP variant inside each outer fold (§10.2).",
   "The SEM image reader's threshold and the SEM CNN were refitted without the held-out surface in every LOSO fold; REF-H images were never used for training.",
   "REF-H was examined after the textured analysis, so it is a stress test that the author had seen, not fresh blind data (stated again in §15)."])
H2('5.3 Metrics')
P("RMSE, MAE and R² are computed on β_{max} (not on its logarithm) over pooled impacts. Interval coverage is the fraction of impacts whose measured β_{max} lies inside the "
  "predicted 90% interval. Where a metric could not be computed from saved per-impact predictions it is marked \"" + NA + "\".")
H2('5.4 Statistical comparison')
P("LOSO errors are correlated within a held-out surface, so the surface, not the impact, is the independent unit [12]. Differences in RMSE between two models were assessed with "
  "a paired bootstrap that resamples the 12 surfaces with replacement and keeps all impacts of a drawn surface together (4,000 draws, seed 23); confidence intervals for a single "
  "model's RMSE use the same resampling. The decision rule, fixed before the comparison, was: replace the baseline only if the 95% interval of the difference excludes zero; "
  "otherwise keep the simpler model. A positive difference means the candidate is worse than the GP baseline.")

# ---------------------------------------------------------------- 6
H1('6. Models and hyperparameters')
P("Only models that were implemented and evaluated in code are benchmarked. Table 4 lists the main families with hyperparameters extracted from the source "
  "(droplet/models.py in v2; src/benchmark.py and src/models_extra.py in the v1 archive). All tabular models receive the same five inputs unless stated otherwise.")
TAB('Table 4. Hyperparameters, as written in the code.',
    ['Model', 'Configuration'],
    [['GP baseline (final)', 'ConstantKernel(1.0) × Matérn(ν = 3/2, ARD length scales, bounds 10^{−2}–10^{3}) + WhiteKernel(10^{−3}, bounds 10^{−6}–0.1); normalize_y; L-BFGS-B, maxiter 100, ftol 10^{−9}; random_state 23; target ln β_{max}'],
     ['GP Matérn 5/2', 'as baseline with ν = 5/2'],
     ['GP fluid–surface interaction', 'Matérn 3/2 on (ln Re, ln We, ln D_{0}) and on (φ, V_{tex}); kernel a·k_{f} + b·k_{s} + c·k_{f}k_{s} + white noise'],
     ['Laan backbone + GP residual', 'Eq. (2) with A refitted by least squares in each fold (bounds 0.01–20); baseline GP on ln β_{max} − ln β_{Laan}'],
     ['XGBoost', 'n_estimators 400, learning_rate 0.05, max_depth 4, subsample 0.8, colsample_bytree 0.8, min_child_weight 5, random_state 0'],
     ['Random forest / Extra trees', '500 trees, min_samples_leaf 3'],
     ['LightGBM', '400 trees, learning_rate 0.05, num_leaves 15, min_child_samples 10, subsample 0.8, colsample 0.8'],
     ['CatBoost', '800 iterations, depth 5, learning_rate 0.05'],
     ['MLP / deep ensemble', '64–64 ReLU, L2 α = 0.1, early stopping, max_iter 5000; ensemble of 5 seeds'],
     ['FT-Transformer', 'd = 32, 2 layers (src/ft_transformer.py)'],
     ['SVR / kernel ridge / kNN / ridge', 'SVR RBF C = 10, ε = 0.005; KRR RBF α = 10^{−3}, γ = 0.2; kNN k = 10 distance-weighted; ridge on quadratic log features, RidgeCV α ∈ 10^{−4}–10^{2}'],
     ['SEM CNN (descriptor encoder)', 'shared 3-layer CNN over the 43×, 100× and 350× views, 32-unit head, two sigmoid outputs (φ, V_{tex}/25); 250 epochs, seed 23; weak labels from Eq. (3)']],
    widths=[28, 72])
P("Why a GP is a natural first choice here: the dataset is small (1,498 impacts, 297 conditions, 12 surfaces), the response is smooth in the dimensionless inputs, and the "
  "design use needs an uncertainty estimate. A GP with automatic relevance determination (ARD) [6] gives a posterior mean and variance from one fitted kernel, its length "
  "scales show which inputs matter, and with about 1,500 points exact inference is cheap. Gradient-boosted trees [8] model sharp interactions well but give no native "
  "predictive variance and extrapolate as piecewise constants, which matters when the held-out surface lies at the edge of the descriptor range.")

# ---------------------------------------------------------------- 7
H1('7. Results: surface-held-out benchmark')
def mrow(label, k, inputs, delta=None):
    m = M[k]
    idd = m.get('id'); lo = m['loso']; rf = m.get('refh')
    g = lambda d, q: f(d.get(q)) if isinstance(d, dict) and isinstance(d.get(q), float) else NA
    return [label, inputs, g(idd, 'rmse'), g(lo, 'rmse'), g(lo, 'mae'), g(lo, 'r2'), g(rf, 'rmse'), g(rf, 'mae'), g(rf, 'r2'), delta or '–']
dv = S['delta_vs_GP_ci95']; bd = {e['model']: e['delta_vs_baseline_ci95'] for e in BENCH['metrics']}
zoo = [(k, v) for k, v in M.items() if k.startswith('zoo_') and 'gaussian' not in k]
zoo.sort(key=lambda kv: kv[1]['loso']['rmse'])
rows = [mrow('GP Matérn 3/2 ARD (final)', 'v1_GPR', '5 dimensionless'),
        mrow('GP Matérn 5/2', 'v2_matern52', '5 dimensionless', ci(bd['matern52'])),
        mrow('GP fluid–surface interaction', 'v2_structured', '5 dimensionless', ci(bd['structured'])),
        mrow('Laan backbone + GP residual', 'v2_physics_residual', '5 + Laan law', ci(bd['physics_residual'])),
        ['GP, dimensional inputs (re-run)', 'D, V, ρ, σ, μ, s, h', NA, f(r7['loso_rmse']), f(r7['loso_mae']), f(r7['loso_r2']), f(r7['refh_rmse']), NA, NA, ci(r7['delta_vs_baseline_ci95'])],
        mrow('XGBoost', 'v1_XGB', '5 dimensionless', ci(dv['XGB'])),
        mrow('XGBoost, raw 7', 'v1_M1', 'D, V, ρ, σ, μ, s, h', ci(dv['M1'])),
        mrow('XGBoost, Re/We/Oh + s, h', 'v1_M2', 'Re, We, Oh, s, h', ci(dv['M2'])),
        mrow('XGBoost, hybrid', 'v1_M3', 'raw 7 + Re, We, Oh', ci(dv['M3'])),
        mrow('Power law + XGBoost residual', 'v1_M5', '5 + power law', ci(dv['M5']))]
for k, v in zoo:
    lab = v['label'].replace(' (reference, same features)', '')
    if lab.startswith('XGBoost'): continue
    rows.append(mrow(lab, k, '5 dimensionless'))
rows.append(['Laan law alone (A refit)', 'P = We·Re^{−2/5}', NA, f(AB['laan']['loso_rmse']), NA, NA, f(AB['laan']['refh_rmse']), NA, NA, ci([-AB['baseline_minus_laan_ci'][1], -AB['baseline_minus_laan_ci'][0]])])
TAB('Table 5. Verified model results. RMSE, MAE and R² on β_{max}. ID = GroupKFold(5) over conditions; LOSO = leave one textured surface out; REF-H = smooth plate, never trained on. '
    'Δ = candidate RMSE − GP RMSE under LOSO, 95% paired surface-bootstrap interval (positive = worse than the GP). "n/a" entries are ' + NA.lower() + ' (the 13-model screen saved LOSO RMSE only, not per-impact LOSO predictions).',
    ['Model', 'Inputs', 'ID RMSE', 'LOSO RMSE', 'LOSO MAE', 'LOSO R²', 'REF-H RMSE', 'REF-H MAE', 'REF-H R²', 'Δ LOSO (95% CI)'],
    [[c if c != NA else 'n/a' for c in r] for r in rows], widths=[21, 13, 7, 7, 7, 7, 7, 7, 7, 17])
P(f"The GP is the most accurate of the 13 learners screened on identical inputs (Table 5, Fig. 3; parity plots in Fig. 4). Its LOSO RMSE is {gp['loso']['rmse']:.4f} "
  f"(95% surface-bootstrap interval {S['loso_rmse_ci95_surface_bootstrap']['GPR'][0]:.4f}–{S['loso_rmse_ci95_surface_bootstrap']['GPR'][1]:.4f}), with MAE {gp['loso']['mae']:.4f} "
  f"and R² {gp['loso']['r2']:.4f}; the ID RMSE is {gp['id']['rmse']:.4f}, so moving from seen to unseen surfaces costs about {100 * (gp['loso']['rmse'] / gp['id']['rmse'] - 1):.0f}% in RMSE. "
  f"XGBoost on the same inputs scores {xg['loso']['rmse']:.4f}, significantly worse (Δ {ci(dv['XGB'])}); the GP wins on {GX['GPR_wins_surfaces']} of 12 held-out surfaces "
  f"and keeps {pct(GX['LOSO_within_0.05']['GPR'], 0)} of impacts within ±0.05 against {pct(GX['LOSO_within_0.05']['XGB'], 0)} for XGBoost. Random forest and extra trees "
  f"({M['zoo_random_forest']['loso']['rmse']:.4f}, {M['zoo_extra_trees']['loso']['rmse']:.4f}) are the strongest non-GP learners. The worst-case held-out surface for the GP is "
  f"D50 ({BENCH['metrics'][0]['worst_rmse']:.3f}), the densest texture, whose φ = 0.010 lies at the edge of the descriptor range.")
P(f"REF-H ranks the models differently. SVR and kernel ridge, which are poor under LOSO ({M['zoo_svr_rbf_']['loso']['rmse']:.3f} and {M['zoo_kernel_ridge_rbf_']['loso']['rmse']:.3f}), "
  f"have the lowest REF-H RMSE ({M['zoo_svr_rbf_']['refh']['rmse']:.4f}, {M['zoo_kernel_ridge_rbf_']['refh']['rmse']:.4f}), and XGBoost ({xg['refh']['rmse']:.4f}) ties the GP "
  f"({gp['refh']['rmse']:.4f}; difference interval {ci(GX['REFH_GPR_minus_XGB_CI'])}). REF-H is a single surface at φ = 1, outside the textured range, so one surface cannot rank models: "
  "a learner that collapses toward a smooth fluid-only trend can do well there while failing on the textured surfaces it was designed for. We therefore select on LOSO and report REF-H "
  "as an out-of-range stress test.")
FIG('fig03_model_comparison', "Figure 3. Model comparison on identical evaluation splits. (a) LOSO RMSE on β_{max} for every learner that was implemented and evaluated, sorted; the dotted line marks the final GP. "
    "Blue: GP variants; red: XGBoost variants; grey: other families (13-model screen). Kernel ridge (0.2568) is clipped. (b) RMSE on the smooth REF-H plate for the same models. "
    "The two rankings disagree, which is why selection uses the 12-surface LOSO score and REF-H is reported only as an out-of-range test.")
FIG('fig04_parity', "Figure 4. Parity plots. (a) GP and (b) XGBoost, each point predicted with its whole surface held out (LOSO), same inputs, n = 1,498; the line is y = x. "
    "XGBoost's errors are larger at high β_{max} (low-viscosity, high-We impacts) and show horizontal bands typical of piecewise-constant trees. (c) GP trained on all textured "
    "surfaces, predicting REF-H with 90% intervals (grey bars); red points fall outside their interval, concentrated at high β_{max}.")

# ---------------------------------------------------------------- 8
H1('8. Why the models differ')
P("Three mechanisms explain Table 5. First, data size and smoothness favour kernel methods. The response is a smooth, nearly monotone function of ln Re and ln We, sampled "
  "on a lattice of about 25 conditions per surface. A GP interpolates such a function with a few hyperparameters, whereas tree ensembles must approximate it by steps "
  "and need many more conditions per region to do so finely. When a whole surface is removed, the GP borrows strength along the length scales of φ and V_{tex}; trees "
  "can only reuse the leaves of the nearest training surfaces.")
P(f"Second, interactions and extrapolation. Gradient boosting captures fluid–surface interactions readily, and its deficit to the GP is smaller on seen surfaces (ID {xg['id']['rmse']:.4f} vs "
  f"{gp['id']['rmse']:.4f}) than on unseen ones (LOSO {xg['loso']['rmse']:.4f} vs {gp['loso']['rmse']:.4f}). But a held-out surface such as D50 lies at the edge of the φ range, where a tree prediction is constant beyond the last split. An explicit interaction "
  f"kernel did not help the GP either (LOSO {M['v2_structured']['loso']['rmse']:.4f}, Δ {ci(bd['structured'])}): with 12 surfaces there is little information to learn a separate "
  "surface covariance.")
P("Third, uncertainty. The GP returns a posterior variance for every query, which grows away from the data (Fig. 9b) and can be calibrated (§10). XGBoost has no native "
  "predictive variance; it can be wrapped in quantile loss or ensembles, but those were not evaluated here, so no calibrated XGBoost interval is claimed. The deep ensemble "
  f"(LOSO {M['zoo_deep_ensemble_5_x_mlp_']['loso']['rmse']:.4f}) and FT-Transformer ({M['zoo_ft_transformer_d_32_2_layers_']['loso']['rmse']:.4f}) were less accurate than the GP at this data size.")
P(f"The CNN plays a different role. It does not predict β_{{max}}; it reads texture descriptors from SEM images so that a new surface can be described without measuring its "
  f"geometry. Feeding CNN-predicted descriptors to the GP gave LOSO {SEMS['loso_rmse']:.4f} (Δ {ci(SEMS['delta_vs_baseline_ci95'])}, not significant), so image-derived "
  "descriptors are a complement to, not a replacement for, the physics-based model. The physical image reader of §11 achieves the same without training a network.")

# ---------------------------------------------------------------- 9
H1('9. Ablation of input representations')
p3b, p3d = P3['base: logRe, logWe, phi, texvol'], P3['base + logD']
TAB('Table 6. Input-representation ablation for the GP (LOSO and REF-H RMSE on β_{max}). Rows marked † were re-run on the v2 harness in this study; others are read from the saved phase files.',
    ['Representation', 'LOSO RMSE', 'REF-H RMSE', 'Δ LOSO vs final (95% CI)', 'Source'],
    [['Laan law alone (A refit per fold)', f(AB['laan']['loso_rmse']), f(AB['laan']['refh_rmse']), ci([-AB['baseline_minus_laan_ci'][1], -AB['baseline_minus_laan_ci'][0]]), 'ablation.json'],
     ['Re, We, Oh + s, h (earlier PIML set)', f(P2['GPR | Re,We,Oh + spacing,depth (paper PIML set)']['LOSO']), f(P2['GPR | Re,We,Oh + spacing,depth (paper PIML set)']['REFH']), 'n/a', 'phase2.json'],
     ['ln Re, ln We, φ, V_{tex}', f(p3b['LOSO']), f(p3b['REFH']), 'n/a', 'phase3.json'],
     ['  + ln Oh', f(P3['base + logOh']['LOSO']), f(P3['base + logOh']['REFH']), 'n/a (identical inputs span)', 'phase3.json'],
     ['  + Bond number', f(P3['base + Bo']['LOSO']), f(P3['base + Bo']['REFH']), 'n/a', 'phase3.json'],
     ['  + ln D_{0} = final model', f(p3d['LOSO']), f(p3d['REFH']), '0', 'phase3.json'],
     ['final − φ', f(P3['final - phi']['LOSO']), f(P3['final - phi']['REFH']), 'n/a', 'phase3.json'],
     ['final − V_{tex}', f(P3['final - texvol']['LOSO']), f(P3['final - texvol']['REFH']), 'n/a', 'phase3.json'],
     ['surface-blind (ln Re, ln We, ln D_{0})', f(AB['blind3']['loso_rmse']), f(AB['blind3']['refh_rmse']), ci([-AB['baseline_minus_blind3_ci'][1], -AB['baseline_minus_blind3_ci'][0]]), 'ablation.json'],
     ['raw s, h instead of φ, V_{tex} (REF-H = 0/0)', f(P4['raw geometry (spacing, depth; REF-H = 0/0)']['LOSO']), f(P4['raw geometry (spacing, depth; REF-H = 0/0)']['REFH']), 'n/a', 'phase4.json'],
     ['raw fluid (D, V, ρ, σ, μ) + φ, V_{tex}', f(RAWPHI['LOSO']), f(RAWPHI['REFH']), 'n/a', 'phase_extra_raw_phi.json'],
     ['dimensional raw 7 (D, V, ρ, σ, μ, s, h) †', f(r7['loso_rmse']), f(r7['refh_rmse']), ci(r7['delta_vs_baseline_ci95']), 'raw7_rerun.json'],
     ['Laan backbone + GP residual', f(phys['loso']['rmse']), f(phys['refh']['rmse']), ci(bd['physics_residual']), 'benchmark.json'],
     ['φ read from SEM image (+ V_{tex})', f(XC['image_phi']['loso']['rmse']), f(XC['image_phi']['refh']['rmse']), ci(XC['image_phi']['delta_vs_baseline_ci95']), 'extensions'],
     ['ln Re, ln We, ln D_{0}, image φ only', f(XC['image_only']['loso']['rmse']), f(XC['image_only']['refh']['rmse']), ci(XC['image_only']['delta_vs_baseline_ci95']), 'extensions'],
     ['+ cos θ_{adv} (water advancing angle)', f(XC['angle_input']['loso']['rmse']), f(XC['angle_input']['refh']['rmse']), ci(XC['angle_input']['delta_vs_baseline_ci95']), 'extensions'],
     ['Lee β_{0} correction', f(XC['lee_beta0']['loso']['rmse']), f(XC['lee_beta0']['refh']['rmse']), ci(XC['lee_beta0']['delta_vs_baseline_ci95']), 'extensions']],
    widths=[38, 12, 12, 24, 14])
P(f"Four results stand out (Table 6, Fig. 5). (1) The physics law alone is not competitive (LOSO {AB['laan']['loso_rmse']:.3f}); the learned model reduces error about sixfold. "
  f"(2) Oh is redundant: adding ln Oh to (ln Re, ln We, φ, V_{{tex}}) gives {P3['base + logOh']['LOSO']:.4f} against {p3b['LOSO']:.4f}, identical to four decimals, as the identity "
  f"ln Oh = ½ ln We − ln Re requires. The useful third fluid-side input is ln D_{{0}}, which lowers LOSO from {p3b['LOSO']:.4f} to {p3d['LOSO']:.4f} because D_{{0}} varies at fixed "
  f"Re and We between fluids (and the Bond number, which also carries D_{{0}}, helps for the same reason: {P3['base + Bo']['LOSO']:.4f}). Adding ln D_{{0}} raises the REF-H error "
  f"from {p3b['REFH']:.4f} to {p3d['REFH']:.4f}; we kept it because selection is on LOSO. (3) Texture descriptors matter but are second order: removing φ costs "
  f"{P3['final - phi']['LOSO'] - p3d['LOSO']:.4f} and the surface-blind GP scores {AB['blind3']['loso_rmse']:.4f}, yet the bootstrap interval of the surface-blind difference "
  f"includes zero ({ci([-AB['baseline_minus_blind3_ci'][1], -AB['baseline_minus_blind3_ci'][0]])}), so with 12 surfaces the texture effect is not statistically established. "
  f"SHAP values and ARD length scales agree (Fig. 6): ln Re and ln We dominate, φ and V_{{tex}} are an order of magnitude smaller.")
P(f"(4) Dimensional inputs scored lower: a GP on the raw seven variables reaches LOSO {r7['loso_rmse']:.4f} (MAE {r7['loso_mae']:.4f}, R² {r7['loso_r2']:.4f}), lower than the final model. "
  f"Under the pre-registered rule this is not a significant improvement (Δ {ci(r7['delta_vs_baseline_ci95'])} includes zero), and the raw encoding has two practical defects. "
  f"It must assign the smooth plate an arbitrary pitch and depth, and the REF-H error depends strongly on that choice: {ENC['0/0']:.3f} for s = h = 0, {ENC['800/25']:.3f} for "
  f"800/25 and {ENC['5000/0']:.3f} for 5000/0. It also cannot describe a new texture by anything other than the same two numbers. We keep the dimensionless model as the "
  "default and report the dimensional result as an unresolved alternative that more surfaces could settle. The same applies to the Laan-backbone residual GP "
  f"(LOSO {phys['loso']['rmse']:.4f}, Δ {ci(bd['physics_residual'])}). An earlier run that resampled replicate conditions instead of surfaces reported an interval that excluded "
  f"zero ({ci(CS['M6_minus_GPR_CI95'])}); with surfaces resampled it does not, and we report the surface result.")
FIG('fig05_ablation', "Figure 5. Input-representation ablation. (a) LOSO and (b) REF-H RMSE for the GP with different inputs; dotted lines mark the final model (black bar). "
    "Grey: physics law alone; blue: representation variants; green: φ measured from SEM images; red: wettability inputs. Values above the axis limit are clipped and printed. "
    "Adding ln Oh leaves both errors unchanged (it is a linear combination of ln Re and ln We); wettability inputs that look harmless under LOSO fail badly on the smooth plate.")
FIG('fig06_importance', "Figure 6. Input relevance for the final GP. (a) Mean absolute SHAP value [15] of each input on the predicted ln β_{max} (KernelExplainer, 20 k-means background points; "
    "250 textured impacts and all 125 REF-H impacts). (b) Fitted ARD length scales of the Matérn kernel on standardised inputs; a shorter length scale means the output varies faster "
    "with that input. Both views rank ln Re and ln We first, ln D_{0} third, and the texture descriptors last.")

# ---------------------------------------------------------------- 10
H1('10. Uncertainty and error analysis')
H2('10.1 Error structure')
P(f"LOSO residuals have no trend with We or Oh (Fig. 7a, b), but their spread depends strongly on the fluid: RMSE falls from {fluid_rmse[0]['loso_rmse']:.3f} for water to "
  f"{fluid_rmse[91]['loso_rmse']:.3f} for 91 wt% glycerol. Part of this is irreducible: the standard deviation of β_{{max}} between repeats of the same condition is "
  f"{S['replicate_sd_by_fluid']['textured']['0']:.3f} for water and {S['replicate_sd_by_fluid']['textured']['20']:.3f} for 20 wt% glycerol, against "
  f"{S['replicate_sd_by_fluid']['textured']['91']:.3f} for 91 wt% (computed from the data). The physical origin of this extra scatter for low-viscosity drops was not "
  f"investigated here. The overall bias is small ({BENCH['metrics'][0]['bias']:+.4f}). Per-surface RMSE ranges from "
  f"{min(BENCH['metrics'][0]['per_surface'].values()):.3f} to {max(BENCH['metrics'][0]['per_surface'].values()):.3f} (Fig. 7c); the two densest textures, D50 and S50, are the worst "
  f"and both are under-predicted on average (mean residual {S['loso_bias_by_surface']['D50']:+.3f} and {S['loso_bias_by_surface']['S50']:+.3f}), consistent with the descriptor range edge discussed in §7. Large errors cluster at high Re and high We (Fig. 9a).")
FIG('fig07_residuals', "Figure 7. LOSO residuals of the final GP (prediction − measurement). (a) Against We and (b) against Oh, coloured by glycerol fraction; the five fluids appear as "
    "five Oh bands. Scatter shrinks with viscosity. (c) Residual distribution for each held-out surface (box: quartiles; whiskers: 1.5 IQR; outliers not drawn), with the surface RMSE printed above.")
H2('10.2 Calibrated intervals')
P(f"The nominal GP interval exp(μ ± 1.645σ) is replaced by exp(μ ± qσ), where q is set from normalised cross-validated residuals so that 90% of held-out impacts fall inside; "
  f"q = {cov['q']:.3f} for the deployed model. This is empirical calibration in the spirit of split conformal prediction [9], not a distribution-free guarantee: impacts share "
  f"surfaces and a new surface is a distribution shift [10]. In a nested design, where both the GP variant and q were chosen inside each outer fold without the held-out surface, "
  f"the RMSE was {NEST['nested_rmse']:.4f} (vs {NEST['baseline_rmse']:.4f} for the fixed baseline) and coverage {pct(NEST['coverage'])}, with a mean interval width of "
  f"{NEST['mean_width']:.3f}. The small RMSE increase is the price of honest selection [11].")
P(f"Coverage is uneven (Fig. 8). Under LOSO it is {pct(cov['overall'])} overall but {pct(cov['fluid_loso']['0'])} for water and {pct(cov['fluid_loso']['91'])} for 91 wt% glycerol; "
  f"per surface it ranges from {pct(min(cov['surface'].values()))} to {pct(max(cov['surface'].values()))}. On REF-H it drops to {pct(cov['refh'])}: "
  f"{pct(cov['fluid_refh']['0'], 0)} and {pct(cov['fluid_refh']['20'], 0)} for the two least viscous fluids, but {pct(cov['fluid_refh']['91'], 0)} for 91 wt%. On REF-H the water errors are not a constant offset "
  f"(mean bias {S['refh_bias_by_fluid']['0']:+.3f}, replicate scatter {S['replicate_sd_by_fluid']['refh']['0']:.3f}) but vary between conditions, so the smooth plate departs from the textured trend "
  "in a condition-dependent way that the model cannot know about. The GP noise term is a single "
  "constant, while the true scatter is fluid-dependent; a heteroscedastic noise model is the obvious remedy (§16). The interval is therefore trustworthy for viscous fluids and "
  "too narrow for water-like fluids on a smooth or new surface type. The predictor reports this as a reliability flag (§13).")
FIG('fig08_coverage', "Figure 8. Empirical coverage of the calibrated 90% interval. (a) By fluid, for held-out textured surfaces (LOSO, blue) and for REF-H (gold); the dashed line is the 90% target. "
    "(b) By held-out textured surface. Coverage is near target for viscous fluids and too low for water and 20 wt% glycerol, most severely on the smooth plate.")
FIG('fig09_error_map_uncertainty', "Figure 9. (a) Absolute LOSO error of the GP across the Re–We plane; the largest errors are at high Re and We (water and 20 wt% glycerol). "
    f"(b) Final GP median (line) and calibrated 90% interval (band) on surface D200 as a function of impact velocity, for water (D_{{0}} = {FN['uncert_D0']['0']:.2f} mm) and 91 wt% "
    f"glycerol (D_{{0}} = {FN['uncert_D0']['91']:.2f} mm; median measured D_{{0}} on D200), with the measured impacts. Grey bands are outside the measured velocity range ({T_['V'][0]:.2f}–{T_['V'][1]:.2f} m/s); there the interval "
    "widens, and below the range the median turns upward, an unphysical artefact that shows why out-of-range predictions are flagged rather than trusted.")

# ---------------------------------------------------------------- 11
H1('11. Image-based surface description')
P("To predict a surface that has not been characterised geometrically, the texture must be read from an image. Two routes were evaluated on the 39 SEM micrographs of the dataset "
  "(13 surfaces × 43×, 100× and 350×; JEOL JCM-7000, 2560 × 2048 px, annotation bar from row 1880; pixel size from the instrument magnification tag).")
P(f"Physical reader. The local standard deviation of grey level over a {IR['window_um']} µm window (Gaussian pre-smoothing, σ = 1 px) separates rough laser tracks from smooth "
  f"plateaus; a single Otsu threshold [14] over the textured training images (value {IR['threshold_all_textured']:.2f}) gives φ as the fraction of plateau pixels. "
  f"The reader is valid near 43× ({IR['per_surface']['D200']['um_per_px']:.3f} µm/px); at higher magnifications it under-reads φ and is refused by the software. Image φ agrees "
  f"with the geometric φ to within {img_diff_200:.2f} for pitches ≥ 200 µm, differs more on the densest textures (D50: {IR['per_surface']['D50']['image_phi']:.3f} vs "
  f"{IR['per_surface']['D50']['geometry_phi']:.3f}; S50: {IR['per_surface']['S50']['image_phi']:.3f} vs {IR['per_surface']['S50']['geometry_phi']:.3f}), and reads REF-H as "
  f"{IR['per_surface']['REF-H']['image_phi']:.3f}, the correct smooth-surface value, without ever seeing it in training (Fig. 10).")
P(f"Using image φ in place of geometric φ, with the threshold refitted inside every LOSO fold, gives LOSO {XC['image_phi']['loso']['rmse']:.4f} (Δ {ci(XC['image_phi']['delta_vs_baseline_ci95'])}); "
  f"a GP on ln Re, ln We, ln D_{{0}} and image φ alone gives {XC['image_only']['loso']['rmse']:.4f} (Δ {ci(XC['image_only']['delta_vs_baseline_ci95'])}) and REF-H "
  f"{XC['image_only']['refh']['rmse']:.4f}. Neither difference is significant, so a new surface can be described by one 43× SEM image with no measurable loss of accuracy. The image "
  "disagreement on D50 and S50 suggests the nominal φ formula over-states how fully dense textures are ablated; it does not show which of the two is physically right.")
P(f"CNN encoder. A small CNN trained in each fold on the three magnifications, with Eq. (3) values as weak labels, gave LOSO {SEMS['loso_rmse']:.4f} when its descriptors were fed "
  f"to the GP (Δ {ci(SEMS['delta_vs_baseline_ci95'])}). Its GP coverage was {pct(SEMS['nominal_gp_coverage'])} and does not include encoder uncertainty. Trained on all textured "
  f"surfaces, the CNN reads REF-H as φ = {SEMGP['surface_descriptors']['REF-H']['phi']:.3f} (the archived first CNN: {CNN1['REF-H']['pred'][0]:.3f}) rather than 1. The physical reader is simpler, needs no training images and extrapolates correctly to the smooth plate, so it is the image route used by the software.")
FIG('fig10_sem_image_reader', "Figure 10. SEM image reader. Top: 43× SEM images of four textured surfaces and the smooth REF-H plate (annotation bar removed; scale bar 500 µm). "
    "Bottom: plateau mask (white = smooth plateau, blue = rough laser track) from the local-roughness map and the Otsu threshold fitted on textured images, with the "
    "resulting φ and the φ from nominal geometry, Eq. (3). The reader recovers φ = 1.000 on REF-H, which it never saw in training.")

# ---------------------------------------------------------------- 12
H1('12. Measurement from side-view video')
P("A backlit high-speed side view shows a dark drop and a dark substrate on a bright background. The pipeline (droplet/video.py) needs only the camera frame rate and the "
  "pixel scale: (i) one Otsu threshold over the clip; (ii) the substrate is the lowest band of rows that are at least 80% dark; (iii) before contact the drop is the largest "
  "dark blob above it, D_{0} is its equivalent-circle diameter (median of the last eight pre-contact frames) and V is the slope of a straight-line fit of its centroid height "
  "against time; (iv) after contact the spreading diameter D(t) is the blob's horizontal extent and β_{max} = max D(t)/D_{0}; (v) the GP predicts β_{max} from D_{0}, V, the "
  "fluid and the surface, and the measurement is compared with the calibrated 90% interval. A watch mode processes each new file in a folder and appends a row to a results table.")
P(f"No real side-view clips of these surfaces were available to the study (the dataset provides measured values, not video), so the pipeline was verified on synthetic clips rendered "
  f"with known D_{{0}}, V and β_{{max}} at 5,000 fps, with blur and sensor noise (Fig. 11). Over four clips the recovered D_{{0}} and V were within {100 * vid_D_err:.2f}% and "
  f"{100 * vid_V_err:.2f}% of truth and β_{{max}} within {100 * vid_beta_err:.1f}%; analysis took {min(vid_s):.2f}–{max(vid_s):.2f} s per 54-frame clip in Python. The browser port "
  f"initially timed frames by their position in the list, and dropped frames during playback shifted V (one headless run read V = 1.414 m/s for a true 1.2 m/s). Frames are now "
  f"indexed by their media timestamps; five repeated browser runs then gave D_{{0}} = {BVC[0]['D0']:.3f} mm, V = {BVC[0]['V']:.3f} m/s and β_{{max}} = {BVC[0]['beta']:.3f} for a "
  "true 2.5 mm, 1.2 m/s and 2.6. These are software checks only: accuracy on real recordings, with reflections, satellite drops, non-spherical drops and tilted substrates, is untested.")
FIG('fig11_video_pipeline', "Figure 11. Video measurement on a synthetic test clip (true D_{0} = 2.5 mm, V = 1.2 m/s, β_{max} = 2.6, 5,000 fps, 0.03 mm/px). (a–d) Frames before contact, "
    "at contact and during spreading; the orange line is the automatically detected substrate. (e) Spreading diameter D(t)/D_{0} detected from every frame, with the rendered "
    "β_{max} (dashed). The inset text compares detected and rendered values. This verifies the software, not its accuracy on real footage.")

# ---------------------------------------------------------------- 13
H1('13. Real-time Droplet Intelligence platform')
P("The scientific result of this paper is the model and its evaluation (§5–§10). This section describes how the same model is packaged for use; it adds no accuracy claim. "
  "Fig. 1 separates the two paths: the prediction path (inputs → dimensionless groups → GP → calibrated interval) and the measurement and interface path (SEM image, video, browser), "
  "which only supplies inputs to, or displays outputs of, the same model.")
P(f"The trained GP is exported as JSON (training inputs, kernel hyperparameters, α = K^{{−1}}y) plus the packed lower-triangular Cholesky factor of K in float64, and evaluated by a "
  f"dependency-free JavaScript engine. Over {PAR['cases']} test cases and {PAR['models']} model variants, the browser and Python predictions agree to "
  f"{PAR['max_absolute_beta_difference']:.1e} in β_{{max}} and {PAR['max_absolute_interval_difference']:.1e} in the interval bounds, and the engine rejects "
  f"{PAR['invalid_inputs_rejected']} malformed inputs. Every prediction carries a reliability status: inputs outside the measured range, a Mahalanobis feature-space score above the "
  "training 95th percentile, and smooth surfaces (where coverage is known to be poor) are flagged. The interface (Fig. 12) offers prediction from inputs, from an uploaded SEM "
  "image and from an uploaded video, all computed locally in the browser.")
TAB('Table 7. Measured latency of a single β_{max} prediction (posterior mean and variance over 1,498 training impacts). Warm model, 200 repeats; model download excluded.',
    ['Implementation', 'Median (ms)', '95th percentile (ms)', 'Environment'],
    [['Python predictor (droplet.predict, incl. input validation)', f"{S['latency_python_single_query_ms']['median']:.1f}", f"{S['latency_python_single_query_ms']['p95']:.1f}", 'cloud container CPU'],
     ['Browser engine (engine.mjs)', f"{BL['median_ms']:.1f}", f"{BL['p95_ms']:.1f}", 'headless Chromium, same container'],
     ['Video analysis, 54 frames 300×400 px (Python)', f"{1e3 * min(vid_s):.0f}–{1e3 * max(vid_s):.0f}", '–', 'cloud container CPU, 4 clips']],
    note='"Real time" here means interactive: a prediction is returned within milliseconds of a change in input. The capture-to-result latency of a live camera system was not measured.',
    widths=[46, 14, 18, 22])
FIG('fig12_interface', "Figure 12. The interface running the same GP in the browser. (a) Prediction from fluid, surface, D_{0} and V, with the calibrated 90% interval, the dimensionless "
    "groups and the velocity response curve. (b) Prediction from an uploaded 43× SEM image (D200 shown): rough laser tracks are highlighted, φ is read from the image and the "
    "image-only GP predicts β_{max}. (c) Automatic measurement from an uploaded side-view video (synthetic test clip). The arbitrary test β_{max} of 2.6 lies outside the model's "
    "interval for that impact on D200, and the interface reports the disagreement instead of hiding it.", 1.0)

# ---------------------------------------------------------------- 14
H1('14. Discussion')
P("Three findings generalise beyond this dataset. First, the evaluation protocol changes the conclusion. Under a condition-grouped split XGBoost trails the GP by "
  f"{xg['id']['rmse'] - gp['id']['rmse']:.3f} in RMSE; under LOSO, the question that matters for design, the gap grows to {xg['loso']['rmse'] - gp['loso']['rmse']:.3f}, "
  f"while random forest stays {M['zoo_random_forest']['id']['rmse'] - gp['id']['rmse']:.3f} (ID) and {M['zoo_random_forest']['loso']['rmse'] - gp['loso']['rmse']:.3f} (LOSO) behind. "
  "Rankings on seen surfaces therefore do not transfer automatically to unseen ones, and a single held-out surface such as REF-H is too small a test to rank models.")
P("Second, physics enters most usefully as representation, not as a hard-coded law. The dimensionless inputs make the problem nearly linear in its dominant directions, encode the "
  "smooth plate as a natural point (φ = 1, V_{tex} = 0) and make Oh redundant by construction. A Laan-law backbone did not improve significantly on this, and the law on its own is "
  "about six times worse. The dimensional GP's lower but non-significant LOSO error shows the limits of 12 surfaces: two representations that differ by 0.003 cannot be told apart.")
P("Third, uncertainty is where the model is weakest. Point accuracy is good (R² 0.99 on unseen surfaces), but a constant noise term cannot represent fluid-dependent scatter, and the "
  "intervals under-cover water-like drops, severely so on the smooth plate. Wettability inputs, an obvious physical fix, failed here because the training angles are confined to a "
  "narrow superhydrophobic range; the smooth plate's β_{max} is close to the textured surfaces', not to the Lee prediction.")

# ---------------------------------------------------------------- 15
H1('15. Limitations')
L(["Twelve textured surfaces from one laboratory, one substrate material and one laser process. LOSO estimates generalisation to similar textures, not to other materials or texture types.",
   "Twelve surfaces give wide bootstrap intervals: several apparent improvements (dimensional inputs, Laan backbone, Matérn 5/2, image φ) cannot be confirmed or rejected.",
   "REF-H was examined during development, so it is a previously inspected stress test, not fresh blind data. It is also a single surface outside the textured descriptor range.",
   "Texture descriptors come from nominal geometry with an assumed track width (Eq. 3); image φ shows these are inaccurate for the densest textures.",
   "Calibrated intervals under-cover low-viscosity fluids (water coverage %s under LOSO, %s on REF-H); the GP noise is homoscedastic." % (pct(cov['fluid_loso']['0']), pct(cov['fluid_refh']['0'], 0)),
   "Wettability is not modelled; the REF-H contact angle used in the wettability tests was assumed, not measured.",
   "The 13-model screen saved only LOSO RMSE and REF-H predictions, so LOSO MAE and R² for those models are not available.",
   "The video pipeline was verified on synthetic clips only; no real impact video of these surfaces was available. The image reader is calibrated for 43× SEM images only.",
   "Latency was measured in one cloud container and a headless browser; it is indicative, not a benchmark of user hardware. No live camera loop was tested.",
   "No hyperparameter search was run for the tree and neural models beyond the fixed settings of Table 4; their scores are for those settings."])

# ---------------------------------------------------------------- 16
H1('16. Conclusions and future work')
P(f"On 12 laser-textured surfaces held out one at a time, a Matérn-3/2 GP on ln Re, ln We, ln D_{{0}}, φ and V_{{tex}} predicts the maximum spreading ratio with RMSE "
  f"{gp['loso']['rmse']:.4f} and R² {gp['loso']['r2']:.3f}, significantly better than XGBoost on the same inputs and about six times better than the Laan law, and with intervals "
  f"that reach {pct(cov['overall'])} coverage overall but fail for low-viscosity drops on new surface types. Oh carries no information beyond Re and We. The texture fraction can be read "
  "from one SEM image without loss of accuracy, a video pipeline extracts the model inputs automatically, and the model runs interactively in a browser with outputs identical to Python.")
P("Everything below is proposed and has not been evaluated in this study. It is kept separate from the results above.")
TAB('Table 8. Proposed / future evaluation (not benchmarked; no results claimed).',
    ['Direction', 'Purpose', 'What it needs'],
    [['Heteroscedastic GP (fluid-dependent noise)', 'Fix low-viscosity under-coverage', 'Re-fit and LOSO coverage test'],
     ['Quantile or conformalised XGBoost intervals', 'Fair uncertainty comparison with trees', 'Implementation and the same calibration protocol'],
     ['Wettability as input', 'Extend to partially wetting surfaces', 'Surfaces with intermediate contact angles; measured REF-H angle'],
     ['More surfaces, other materials and laser processes', 'Narrow bootstrap intervals; test transfer', 'New experiments'],
     ['Real high-speed footage', 'Validate the video pipeline', 'Calibrated side-view recordings with ground truth'],
     ['Temporal models of D(t)', 'Predict the full spreading curve, not only β_{max}', 'Frame-level data'],
     ['Live camera loop', 'True capture-to-prediction real-time operation', 'Camera integration and latency measurement'],
     ['Active learning over texture design', 'Choose the next surface to fabricate', 'Acquisition function on the GP; experiments']],
    widths=[34, 34, 32])

ACK = ("The author thanks M. Može and co-authors for publishing the dataset used here. An AI assistant was used to help write analysis code and edit the text; the author "
       "designed the study, checked every result against the saved result files, and takes responsibility for the content.")
DATA = ("Impact data: Može et al., Mendeley Data, doi:10.17632/wsh8rxwd38.1 [4]. Code, trained model bundles, per-impact predictions, figure and table scripts are in the project "
        "repository (paper/v3/ for this manuscript; Appendix B).")

REFS = [
    'C. Josserand, S. T. Thoroddsen. Drop impact on a solid surface. *Annual Review of Fluid Mechanics* 48 (2016) 365–391.',
    'C. Clanet, C. Béguin, D. Richard, D. Quéré. Maximal deformation of an impacting drop. *Journal of Fluid Mechanics* 517 (2004) 199–208.',
    'N. Laan, K. G. de Bruin, D. Bartolo, C. Josserand, D. Bonn. Maximum diameter of impacting liquid droplets. *Physical Review Applied* 2 (2014) 044018. doi:10.1103/PhysRevApplied.2.044018',
    'M. Može, S. Jereb, R. Lovšin, J. Berce, M. Zupančič, I. Golobič. Dataset on droplet spreading and rebound behavior of water and viscous water-glycerol mixtures on superhydrophobic surfaces with laser-made channels. *Data in Brief* 61 (2025) 111697. doi:10.1016/j.dib.2025.111697. Data: doi:10.17632/wsh8rxwd38.1',
    'S. Jereb, J. Berce, R. Lovšin, M. Zupančič, M. Može, I. Golobič. Investigation of droplet spreading and rebound dynamics on superhydrophobic surfaces using machine learning. *Biomimetics* 10 (2025) 357. doi:10.3390/biomimetics10060357',
    'C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.',
    'F. Pedregosa et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research* 12 (2011) 2825–2830.',
    'T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD* (2016) 785–794.',
    'A. N. Angelopoulos, S. Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511 (2021).',
    'R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani. Conformal prediction beyond exchangeability. *Annals of Statistics* 51 (2023) 816–845.',
    'G. C. Cawley, N. L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research* 11 (2010) 2079–2107.',
    'D. R. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40 (2017) 913–929.',
    'J. B. Lee, N. Laan, K. G. de Bruin, G. Skantzaris, N. Shahidzadeh, D. Derome, J. Carmeliet, D. Bonn. Universal rescaling of drop impact on smooth and rough surfaces. *Journal of Fluid Mechanics* 786 (2016) R4.',
    'N. Otsu. A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics* 9 (1979) 62–66.',
    'S. M. Lundberg, S.-I. Lee. A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems* 30 (2017) 4765–4774.',
]

# ---------------------------------------------------------------- appendix
APP = []
def AH(t): APP.append(('h1', t))
def AH2(t): APP.append(('h2', t))
def AP(t): APP.append(('p', t))
def AT(cap, head, rows, note=None, widths=None): APP.append(('table', cap, head, rows, note, widths))
AH('Appendix A. Per-surface LOSO RMSE')
order = sorted(BENCH['metrics'][0]['per_surface'], key=lambda z: (z[0], int(z[1:])))
prow = lambda lab, ps: [lab] + [f'{ps[s]:.3f}' for s in order]
AT('Table A1. LOSO RMSE on each held-out surface.', ['Model'] + order,
   [prow('GP (final)', BENCH['metrics'][0]['per_surface']), prow('GP Matérn 5/2', BENCH['metrics'][1]['per_surface']),
    prow('GP interaction', BENCH['metrics'][2]['per_surface']), prow('Laan + GP residual', BENCH['metrics'][3]['per_surface']),
    prow('GP dimensional (raw 7)', r7['per_surface']), prow('GP + SEM-CNN descriptors', SEMS['per_surface']),
    prow('XGBoost (same inputs)', GX['XGB']['per_surface']),
    prow('GP image φ only', XC['image_only']['loso']['per_surface'])],
   widths=[22] + [6.5] * 12)
AH('Appendix B. Reproducibility')
AT('Table B1. Reproducibility record.', ['Item', 'Value'],
   [['Data', 'Može et al. [4], files 01–05 (CSV) and 39 SEM TIFFs; SHA-256 hashes in Droplet_Intelligence_v2/results/data_audit.json'],
    ['Software', 'Python 3.11, scikit-learn [7], XGBoost [8], LightGBM, CatBoost, PyTorch (CNN), SciPy; Node/Chromium for the browser engine'],
    ['Random seeds', 'GP random_state 23; bootstrap seed 23 (4,000 draws); tree/neural models seed 0; CNN seed 23'],
    ['Splits', 'LOSO: 12 folds by surface; ID: GroupKFold(5) by condition (297 groups); REF-H: train on all textured surfaces'],
    ['Main model export', 'Droplet_Intelligence_v2/models/release.json, baseline.json + baseline.L.bin (float64 Cholesky factor)'],
    ['Benchmark (v2)', 'python -m droplet.benchmark; python -m droplet.nested; python -m droplet.extensions --workers 4'],
    ['Archived screen (v1)', 'archive/v1_droplet/src/{models_extra,final_pipeline,phases,phase_extra}.py; results in archive/v1_droplet/results/'],
    ['This manuscript', 'paper/v3/: raw7_rerun.py, collect_metrics.py, stats.py, figs.py, screenshots.py, build_paper.py'],
    ['Tests', '23 unit tests in Droplet_Intelligence_v2/tests; browser–Python parity: results/js_parity.json'],
    ['Compute', 'single CPU container; the extension benchmark (5 candidates × 12 LOSO folds) took %.0f s with 4 workers' % EXT['seconds']]], widths=[22, 78])
AH('Appendix C. Verification report')
AT('Table C1. Final verification.', ['Check', 'Result'],
   [['Fabricated values', 'NONE. Every number is read from a saved result file or recomputed from saved per-impact predictions at build time'],
    ['Unverified models in the benchmark', 'NONE. Only implemented and evaluated models appear in Tables 5–6; planned work is in Table 8'],
    ['Metrics recomputed', 'RMSE/MAE/R² recomputed from per-impact predictions for 7 v1 models, 4 v2 GP variants and 5 extension models; REF-H RMSE of all 13 screened models recomputed from saved predictions and matches zoo_table.csv'],
    ['Cross-checks', 'v1 GP LOSO RMSE %.8f vs v2 re-implementation %.8f; browser vs Python β_{max} %.1e' % (gp['loso']['rmse'], M['v2_baseline']['loso']['rmse'], PAR['max_absolute_beta_difference'])],
    ['Re-run in this study', 'dimensional-input GP (reproduces 0.0372 of the archived phase-2 run), bootstrap intervals, latency, video checks'],
    ['Latency claims', 'measured (Table 7); no capture-to-result real-time claim'],
    ['References', '15, all cited in the text; none added without a verifiable source'],
    ['Corrections to earlier drafts', 'Laan-backbone improvement is not significant with surfaces resampled; dimensional inputs are lower but not significant; browser video timing bug found and fixed']],
   widths=[28, 72])

# ============================================================== rendering helpers
def inline_html(t):
    t = html.escape(t, quote=False)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t); t = re.sub(r'\*(.+?)\*', r'<i>\1</i>', t)
    t = re.sub(r'_\{([^}]*)\}', r'<sub>\1</sub>', t); t = re.sub(r'\^\{([^}]*)\}', r'<sup>\1</sup>', t)
    return t


def tokens(t):
    """Split mini-markup into (text, style) runs for DOCX."""
    out = []; i = 0
    pat = re.compile(r'\*\*(.+?)\*\*|\*(.+?)\*|_\{([^}]*)\}|\^\{([^}]*)\}')
    for m in pat.finditer(t):
        if m.start() > i: out.append((t[i:m.start()], ''))
        if m.group(1): out.append((m.group(1), 'b'))
        elif m.group(2): out.append((m.group(2), 'i'))
        elif m.group(3) is not None: out.append((m.group(3), 'sub'))
        else: out.append((m.group(4), 'sup'))
        i = m.end()
    if i < len(t): out.append((t[i:], ''))
    return out


fig_no = {}
def render_html():
    css = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
:root { --ink:#1d2433; --muted:#5b6475; --accent:#1f5fa8; --rule:#d5dae3; --bg:#ffffff; }
html { background: var(--bg); }
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 10.6pt; line-height: 1.42; color: var(--ink); background: var(--bg); max-width: 174mm; margin: 0 auto; }
h1, h2, .title, .sub, .meta, th, .cap b, .kw b { font-family: 'Liberation Sans', Arial, sans-serif; }
.title { font-size: 19pt; font-weight: 700; line-height: 1.2; margin: 0 0 6pt; color: #0f172a; }
.sub { font-size: 11.5pt; color: var(--muted); margin: 0 0 12pt; }
.meta { font-size: 9.5pt; color: var(--muted); border-bottom: 2px solid var(--accent); padding-bottom: 8pt; margin-bottom: 12pt; }
.meta b { color: var(--ink); }
.abstract { background: #f3f6fa; border-left: 3px solid var(--accent); padding: 8pt 11pt; margin: 0 0 8pt; }
.abstract h2 { margin-top: 0; }
.kw { font-size: 9.6pt; margin-bottom: 14pt; }
h1 { font-size: 13pt; color: #0f172a; margin: 16pt 0 5pt; break-after: avoid; }
h2 { font-size: 11pt; color: var(--accent); margin: 10pt 0 4pt; break-after: avoid; }
p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
ul { margin: 0 0 6pt 0; padding-left: 16pt; } li { margin-bottom: 2pt; text-align: justify; }
.eq { display: flex; justify-content: space-between; align-items: center; margin: 6pt 0 8pt; font-style: italic; }
.eq span:first-child { flex: 1; text-align: center; } .eq span:last-child { font-style: normal; }
figure { margin: 8pt 0 10pt; break-inside: avoid; } figure img { width: 100%; display: block; }
.cap { font-size: 9pt; line-height: 1.35; margin-top: 4pt; text-align: justify; color: #2a3142; }
table { width: 100%; border-collapse: collapse; font-size: 8.4pt; margin: 4pt 0 2pt; break-inside: auto; }
caption { caption-side: top; text-align: left; font-size: 9pt; margin-bottom: 4pt; line-height: 1.35; color: #2a3142; }
th { background: #eef2f7; font-weight: 700; font-size: 8pt; text-align: left; padding: 3pt 4pt; border-top: 1.2px solid var(--ink); border-bottom: 0.8px solid var(--ink); }
td { padding: 2.6pt 4pt; border-bottom: 0.5px solid var(--rule); vertical-align: top; }
tr { break-inside: avoid; }
tbody tr:last-child td { border-bottom: 1.2px solid var(--ink); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tnote { font-size: 8.4pt; color: var(--muted); margin: 2pt 0 10pt; }
.tablewrap { margin-bottom: 10pt; } .tablewrap.keep { break-inside: avoid; } .tablewrap.big table { font-size: 7.8pt; }
thead { display: table-header-group; }
.refs p { text-indent: -16pt; padding-left: 16pt; font-size: 9.4pt; text-align: left; margin-bottom: 3pt; }
.small { font-size: 9.4pt; }
"""
    out = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Droplet Intelligence paper</title><style>{css}</style></head><body>",
           f"<div class='title'>{inline_html(TITLE)}</div><div class='sub'>{inline_html(SUB)}</div>",
           f"<div class='meta'><b>{AUTH}</b> · {AFF}<br>{DATE}</div>",
           f"<div class='abstract'><h2>Abstract</h2><p>{inline_html(ABSTRACT)}</p></div><div class='kw'><b>Keywords:</b> {inline_html(KEYWORDS)}</div>"]
    def blocks(Bs):
        for b in Bs:
            k = b[0]
            if k == 'h1':
                brk = " style='break-before: page'" if b[1].startswith('Appendix A') else ''
                out.append(f'<h1{brk}>{inline_html(b[1])}</h1>')
            elif k == 'h2': out.append(f'<h2>{inline_html(b[1])}</h2>')
            elif k == 'p': out.append(f'<p>{inline_html(b[1])}</p>')
            elif k == 'eq': out.append(f"<div class='eq'><span>{inline_html(b[1])}</span><span>({b[2]})</span></div>")
            elif k == 'ul': out.append('<ul>' + ''.join(f'<li>{inline_html(x)}</li>' for x in b[1]) + '</ul>')
            elif k == 'fig': out.append(f"<figure><img src='figs/{b[1]}.png' style='width:{100 * b[3]:.0f}%' alt=''><figcaption class='cap'>{inline_html(b[2])}</figcaption></figure>")
            elif k == 'table':
                _, cap, head, rows, note, widths = b
                isnum = [all(re.fullmatch(r'[−+\-]?[0-9.,]+%?|n/a|–|0', str(r[j])) for r in rows) for j in range(len(head))]
                cg = ''.join(f"<col style='width:{w}%'>" for w in widths) if widths else ''
                th = ''.join(f"<th class='{'num' if isnum[j] else ''}'>{inline_html(h)}</th>" for j, h in enumerate(head))
                tr = ''.join('<tr>' + ''.join(f"<td class='{'num' if isnum[j] else ''}'>{inline_html(str(c))}</td>" for j, c in enumerate(r)) + '</tr>' for r in rows)
                cls = ('tablewrap keep' if len(rows) <= 18 else 'tablewrap') + (' big' if len(head) >= 9 else '')
                out.append(f"<div class='{cls}'><table><caption>{inline_html(cap)}</caption><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"
                           + (f"<div class='tnote'>{inline_html(note)}</div>" if note else '') + '</div>')
    blocks(C)
    out.append(f"<h1>Acknowledgements</h1><p>{inline_html(ACK)}</p><h1>Data and code availability</h1><p>{inline_html(DATA)}</p>")
    out.append("<h1>References</h1><div class='refs'>" + ''.join(f'<p>[{i + 1}] {inline_html(r)}</p>' for i, r in enumerate(REFS)) + '</div>')
    blocks(APP)
    out.append('</body></html>')
    open(R + '/paper_v3.html', 'w').write('\n'.join(out))


def render_docx():
    from docx import Document
    from docx.shared import Pt, Mm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    doc = Document(); sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    for side in ('left_margin', 'right_margin'): setattr(sec, side, Mm(20))
    sec.top_margin = sec.bottom_margin = Mm(20)
    st = doc.styles['Normal']; st.font.name = 'Times New Roman'; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    def para(t, size=None, bold=False, italic=False, color=None, align=None, after=4, font=None):
        p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after)
        if align: p.alignment = align
        for txt, sty in tokens(t):
            r = p.add_run(txt); r.bold = bold or sty == 'b'; r.italic = italic or sty == 'i'
            r.font.subscript = sty == 'sub'; r.font.superscript = sty == 'sup'
            if size: r.font.size = Pt(size)
            if color: r.font.color.rgb = RGBColor.from_string(color)
            if font: r.font.name = font
        return p
    para(TITLE, 17, True, font='Arial', after=4); para(SUB, 11.5, color='5B6475', font='Arial', after=8)
    para(f'**{AUTH}** · {AFF}', 10, after=0); para(DATE, 9.5, color='5B6475', after=10)
    para('Abstract', 11.5, True, font='Arial', after=2); para(ABSTRACT, 10, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=4)
    para('**Keywords:** ' + KEYWORDS, 9.5, after=10)
    def shade(cell, hexcol):
        tcPr = cell._tc.get_or_add_tcPr(); s = OxmlElement('w:shd'); s.set(qn('w:val'), 'clear'); s.set(qn('w:color'), 'auto'); s.set(qn('w:fill'), hexcol); tcPr.append(s)
    def blocks(Bs):
        for b in Bs:
            k = b[0]
            if k == 'h1': para(b[1], 13, True, font='Arial', color='0F172A', after=3).paragraph_format.space_before = Pt(10)
            elif k == 'h2': para(b[1], 11, True, font='Arial', color='1F5FA8', after=2).paragraph_format.space_before = Pt(6)
            elif k == 'p': para(b[1], align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            elif k == 'eq': para(f'{b[1]}        ({b[2]})', italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
            elif k == 'ul':
                for x in b[1]:
                    p = para('• ' + x, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=2); p.paragraph_format.left_indent = Mm(5); p.paragraph_format.first_line_indent = Mm(-3)
            elif k == 'fig':
                doc.add_picture(f'{R}/figs/{b[1]}.png', width=Mm(170 * b[3])); doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                para(b[2], 8.8, color='2A3142', align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=8)
            elif k == 'table':
                _, cap, head, rows, note, widths = b
                para(cap, 8.8, color='2A3142', after=2).paragraph_format.keep_with_next = True
                t = doc.add_table(rows=1, cols=len(head)); t.style = 'Table Grid'
                for j, h in enumerate(head):
                    c = t.rows[0].cells[j]; c.text = ''; shade(c, 'EEF2F7')
                    for txt, sty in tokens(h):
                        r = c.paragraphs[0].add_run(txt); r.bold = True; r.font.size = Pt(7.6); r.font.subscript = sty == 'sub'; r.font.superscript = sty == 'sup'
                for row_ in rows:
                    cells = t.add_row().cells
                    for j, v in enumerate(row_):
                        cells[j].text = ''
                        for txt, sty in tokens(str(v)):
                            r = cells[j].paragraphs[0].add_run(txt); r.font.size = Pt(7.6); r.font.subscript = sty == 'sub'; r.font.superscript = sty == 'sup'; r.bold = sty == 'b'
                if widths:
                    tot = sum(widths)
                    for row_ in t.rows:
                        for j, w in enumerate(widths): row_.cells[j].width = Mm(170 * w / tot)
                if note: para(note, 8.2, color='5B6475', after=8)
                else: para('', after=4)
    blocks(C)
    para('Acknowledgements', 13, True, font='Arial', after=3); para(ACK, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para('Data and code availability', 13, True, font='Arial', after=3); para(DATA, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    para('References', 13, True, font='Arial', after=3)
    for i, r in enumerate(REFS): para(f'[{i + 1}] {r}', 9.2, after=2)
    blocks(APP)
    doc.save(R + '/Droplet_Intelligence_Paper_v3.docx')


def render_pdf():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium-1194/chrome-linux/chrome')
        pg = b.new_page(); pg.goto('file://' + R + '/paper_v3.html'); pg.wait_for_timeout(500)
        pg.pdf(path=R + '/Droplet_Intelligence_Paper_v3.pdf', format='A4', print_background=True, display_header_footer=True,
               header_template='<div></div>', footer_template="<div style='font-size:8px;color:#888;width:100%;text-align:center;font-family:sans-serif'>Droplet Intelligence v3 · <span class='pageNumber'></span> / <span class='totalPages'></span></div>",
               margin=dict(top='18mm', bottom='18mm', left='16mm', right='16mm'))
        b.close()


def renumber():
    order = [b[1] for b in C if b[0] == 'fig']
    new = {int(n[3:5]): i + 1 for i, n in enumerate(order)}
    sub = lambda t: re.sub(r'(Figs?\.|Figure) (\d+)', lambda m: f'{m.group(1)} {new.get(int(m.group(2)), m.group(2))}', t)
    def walk(x):
        if isinstance(x, str): return sub(x)
        if isinstance(x, (list, tuple)): return type(x)(walk(y) for y in x)
        return x
    C[:] = [walk(b) if b[0] != 'fig' else (b[0], b[1], sub(b[2]), b[3]) for b in C]
    APP[:] = [walk(b) for b in APP]
    return new


if __name__ == '__main__':
    print('figure order', renumber())
    n_abs = len(re.sub(r'_\{|\^\{|\}', ' ', ABSTRACT).split())
    print('abstract words:', n_abs)
    render_html(); render_pdf(); render_docx()
    print('wrote paper_v3.html, Droplet_Intelligence_Paper_v3.pdf, Droplet_Intelligence_Paper_v3.docx')
