const fs = require('fs');
const D = require('docx');
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, AlignmentType,
  HeadingLevel, BorderStyle, ShadingType, LevelFormat, Footer, PageNumber, TabStopType } = D;
const N = JSON.parse(fs.readFileSync('numbers.json'));
const X = JSON.parse(fs.readFileSync('../Droplet_Intelligence_v2/results/extensions/summary.json'));   // post-submission extensions
const XC = X.candidates, XR = X.image_reader.per_surface, XS = X.refh_angle_sweep;
const R = Object.fromEntries(N.rows.map(r => [r.key, r]));
const F = Object.fromEntries(N.forest.map(r => [r[0], r]));
const BF = Object.fromEntries(N.by_fluid.map(r => [r.gly, r]));
const f = (x, d = 3) => Number(x).toFixed(d);
const pc = (x, d = 1) => (100 * x).toFixed(d) + '%';
const e3 = x => (x * 1000 >= 0 ? '+' : '−') + Math.abs(x * 1000).toFixed(1);
const ci3 = c => `[${e3(c[0])}, ${e3(c[1])}]`;

const FONT = 'Times New Roman', HFONT = 'Arial';
// mini markup: **bold** *italic* _{sub} ^{sup}
function runs(s, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  for (const part of s.split(re)) {
    if (!part) continue;
    if (part.startsWith('**')) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith('*')) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else if (part.startsWith('_{')) out.push(new TextRun({ ...base, text: part.slice(2, -1), subScript: true }));
    else if (part.startsWith('^{')) out.push(new TextRun({ ...base, text: part.slice(2, -1), superScript: true }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (s, o = {}) => new Paragraph({ children: runs(s, o.run || {}), alignment: o.align ?? AlignmentType.JUSTIFIED,
  spacing: { after: o.after ?? 120, line: 276, before: o.before ?? 0 }, indent: o.indent, keepNext: o.keepNext });
const H1 = s => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(s)], spacing: { before: 280, after: 120 }, keepNext: true });
const H2 = s => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(s)], spacing: { before: 200, after: 80 }, keepNext: true });
const EQ = (s, n) => new Paragraph({ children: [...runs(s), new TextRun({ text: `\t(${n})` })], alignment: AlignmentType.CENTER,
  tabStops: [{ type: TabStopType.RIGHT, position: 9020 }], spacing: { before: 60, after: 120 } });
const B = s => new Paragraph({ numbering: { reference: 'bul', level: 0 }, children: runs(s), spacing: { after: 60, line: 276 }, alignment: AlignmentType.LEFT });
const NUM = (s, ref = 'num') => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(s), spacing: { after: 60, line: 276 }, alignment: AlignmentType.LEFT });
const CAP = s => new Paragraph({ children: runs(s, { size: 19 }), spacing: { before: 60, after: 200, line: 252 }, alignment: AlignmentType.JUSTIFIED });
function FIG(file, wIn, s) {
  const png = fs.readFileSync(file);
  const w = png.readUInt32BE(16), h = png.readUInt32BE(20);
  const W = wIn * 96, Hh = W * h / w;
  return [new Paragraph({ children: [new ImageRun({ type: 'png', data: png, transformation: { width: W, height: Hh },
    altText: { title: file, description: s.replace(/\*|_\{|\^\{|\}/g, ''), name: file } })], alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 } }), CAP(s)];
}
const TW = 9026; // A4 text width with 1" margins
const bd = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const nob = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
function TABLE(head, rows, widths, opt = {}) {
  const tot = widths.reduce((a, b) => a + b, 0);
  const mk = (cells, isHead, i) => new TableRow({ tableHeader: isHead, cantSplit: true, children: cells.map((c, j) => new TableCell({
    width: { size: widths[j], type: WidthType.DXA },
    shading: isHead ? { fill: 'E8EEF6', type: ShadingType.CLEAR, color: 'auto' } : (opt.hl && opt.hl(i) ? { fill: 'F4F6F9', type: ShadingType.CLEAR, color: 'auto' } : undefined),
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    borders: { top: isHead ? bd : nob, bottom: bd, left: nob, right: nob },
    children: [new Paragraph({ keepNext: i < rows.length - 1, keepLines: true, children: runs(String(c), { size: 18, bold: isHead }), alignment: (j === 0 || opt.left) ? AlignmentType.LEFT : AlignmentType.RIGHT, spacing: { after: 0 } })] })) });
  return new Table({ width: { size: tot, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
    rows: [mk(head, true, -1), ...rows.map((r, i) => mk(r, false, i))] });
}
const TCAP = s => new Paragraph({ children: runs(s, { size: 19 }), spacing: { before: 200, after: 80, line: 252 }, keepNext: true });
const SP = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------------- numbers used in prose ----------------
const b = R.baseline, x = R.xgb, L = R.laan, bl = R.blind, ph = R.physics_residual, sm = R.sem, m52 = R.matern52, st = R.structured;
const xgbCI = F['XGBoost, same five inputs'][2], blCI = F['GP without surface descriptors'][2], phCI = F['Laan backbone + residual GP'][2];
const semCI = F['SEM CNN descriptors + GP'][2], nestCI = F['Nested model selection'][2], m52CI = F['Matérn 5/2 GP'][2], stCI = F['Fluid–surface interaction GP'][2];
const d = N.data, nz = N.nested, lg = N.legacy;
const covs = Object.values(N.per_surface_cov), covMin = Math.min(...covs), covMax = Math.max(...covs);
const covMinS = Object.keys(N.per_surface_cov).find(k => N.per_surface_cov[k] === covMin);
const zoo = N.zoo, zr = Object.fromEntries(zoo.map(z => [z.name, z]));

const C = [];
// ---------------- title block ----------------
C.push(new Paragraph({ children: [new TextRun({ text: 'Predicting droplet spreading on laser-textured surfaces the model has never seen', font: HFONT, size: 34, bold: true })], spacing: { after: 80 }, alignment: AlignmentType.LEFT }));
C.push(new Paragraph({ children: [new TextRun({ text: 'A surface-held-out evaluation of Gaussian-process, tree, neural and physics-informed models of β', font: HFONT, size: 24 }), new TextRun({ text: 'max', font: HFONT, size: 24, subScript: true })], spacing: { after: 200 } }));
C.push(new Paragraph({ children: [new TextRun({ text: 'Parth Sharma', bold: true, size: 22 })], spacing: { after: 20 } }));
C.push(new Paragraph({ children: [new TextRun({ text: 'Thapar Institute of Engineering & Technology, Patiala, India', size: 20, italics: true })], spacing: { after: 20 } }));
C.push(new Paragraph({ children: [new TextRun({ text: 'Working paper · 24 September 2026, extended 1 October 2026 · prepared for Interfacial Phenomena in Droplets 2026, IISc Bengaluru, 5–7 October 2026', size: 18, color: '555555' })],
  spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '2A78D6', space: 8 } } }));

// ---------------- abstract ----------------
C.push(new Paragraph({ children: [new TextRun({ text: 'Abstract', font: HFONT, bold: true, size: 22 })], spacing: { after: 80 } }));
C.push(P(`Data-driven models of the maximum spreading ratio β_{max} of impacting drops are usually scored on random splits in which every surface appears in training. They then measure interpolation between replicate impacts, not the question a surface designer asks: how far will a drop spread on a surface that has not been made yet? We re-analyse a public dataset of ${d.n_t.toLocaleString('en-US')} water–glycerol drop impacts on 12 laser-channelled superhydrophobic aluminium surfaces (Može et al., 2025), holding out one entire surface at a time. A further ${d.n_r} impacts on a smooth hydrophobic plate (REF-H) serve as a stress test. A Gaussian process (GP) on ln Re, ln We, ln D_{0} and two geometric texture descriptors predicts β_{max} on unseen surfaces with a root-mean-square error (RMSE) of ${f(b.loso)} (mean absolute percentage error ${f(b.mape, 1)}%). XGBoost on the same inputs gives ${f(x.loso)} (paired surface-bootstrap 95% interval of the difference ${f(xgbCI[0])} to ${f(xgbCI[1])}), and the Laan scaling law alone gives ${f(L.loso)}. A Laan-law backbone, a smoother kernel, a fluid–surface interaction kernel, nested model selection and a convolutional network reading SEM images each changed the error by less than 0.002, with intervals that include zero. Removing the texture descriptors raised the error to ${f(bl.loso)}, but with only 12 surfaces that interval also includes zero. Nominal 90% prediction intervals covered ${pc(b.loso_cov)} of unseen-surface impacts overall, yet ${pc(BF[0].loso_cov, 0)} for water and ${pc(BF[91].loso_cov, 0)} for 91 wt% glycerol. On REF-H the error rose to ${f(b.refh)} and coverage fell to ${pc(b.refh_cov, 0)}, with the shortfall concentrated in the low-viscosity fluids. Reading the flat-area fraction directly from one 43× SEM image, without channel spacing or depth, gives the same unseen-surface error (${f(XC.image_only.loso.rmse)}) and reads the smooth plate correctly as φ = 1; adding contact angles through the Lee et al. correction or as an input does not work on this data. The evidence favours a simple GP. Further gains are more likely from wettability inputs, viscosity-dependent noise and more surfaces than from larger models.`));
C.push(P('**Keywords:** drop impact; maximum spreading; superhydrophobic surfaces; laser texturing; Gaussian process; leave-one-surface-out validation; uncertainty calibration', { after: 200 }));

// ---------------- 1 Introduction ----------------
C.push(H1('1. Introduction'));
C.push(P('When a drop hits a solid surface it spreads into a thin lamella, reaches a maximum diameter D_{max}, and then retracts or rebounds. The maximum spreading ratio β_{max} = D_{max}/D_{0} controls how much surface a drop wets and for how long, which matters for spray cooling, coating, printing and anti-icing surfaces [1]. For a Newtonian liquid, β_{max} depends on the Reynolds number Re = ρVD_{0}/μ, the Weber number We = ρV^{2}D_{0}/σ and the surface. Energy arguments give β_{max} ∝ Re^{1/5} when viscosity limits spreading and β_{max} ∝ We^{1/2} when capillarity does; Clanet et al. proposed We^{1/4} for low-viscosity drops [2]. Laan et al. [3] joined the two limits with a Padé approximant in the impact parameter P = We·Re^{−2/5}:'));
C.push(EQ('β_{max} Re^{−1/5} = P^{1/2} / (A + P^{1/2}),   A ≈ 1.24', 1));
C.push(P('Equation (1) has no term for surface texture. Laser-textured superhydrophobic surfaces change the contact line, trap air and alter viscous losses, so texture effects are exactly what an empirical model has to learn.'));
C.push(P('Može et al. [4] published 1,623 impacts of water and water–glycerol drops on aluminium surfaces with nanosecond-laser channels of six pitches and two depths, plus a smooth hydrophobic reference, recorded at 5,000 frames per second. The same group trained 28 regression models on seven raw inputs and selected an isotropic exponential GP, evaluated with an 80/20 split and 5-fold cross-validation [5]. Every impact condition in the dataset is repeated about five times and every surface is present in training under such a split, so that evaluation answers an interpolation question.'));
C.push(P('This paper asks the harder question: how well can β_{max} be predicted for a textured surface that was never seen in training? Its contributions are:'));
C.push(NUM('A leave-one-surface-out (LOSO) protocol with a paired bootstrap that resamples whole surfaces, and a decision rule fixed before the comparison.', 'c'));
C.push(NUM('A benchmark of 13 learners and four GP variants on identical inputs, including a Laan-law physics backbone and a CNN that reads SEM images.', 'c'));
C.push(NUM('Two ablations that were missing from earlier work: the physics law on its own, and the GP without texture descriptors.', 'c'));
C.push(NUM('A diagnosis of prediction-interval coverage by fluid and on the smooth plate, showing where and why the model stops being trustworthy.', 'c'));
C.push(NUM('A tested implementation: a Python API and a browser predictor that agree to 3×10^{−12}.', 'c'));
C.push(NUM('Extensions toward new surfaces: a physical SEM reader that needs no geometry, a test of contact-angle inputs, and automated measurement of D_{0}, V and β_{max} from impact video (Section 4.8–4.10).', 'c'));

// ---------------- 2 Data ----------------
C.push(H1('2. Data'));
C.push(H2('2.1 Source and scope'));
C.push(P(`All impact data come from the public dataset of Može et al. [4] (Mendeley Data, doi:10.17632/wsh8rxwd38.1); no new experiments were performed for this study. The substrates are aluminium 1050A plates. Channels were made with a nanosecond fibre laser at pitches s = 50, 100, 200, 400, 600 and 800 µm and two mean depths: about 6 µm (surfaces S50–S800) and 25 µm (D50–D800). REF-H is a smooth hydrophobic aluminium reference. Five fluids were used, from water to 91 wt% glycerol (Table 1). Across all ${(d.n_t + d.n_r).toLocaleString('en-US')} impacts, D_{0} spans ${f(d.D[0], 2)}–${f(d.D[1], 2)} mm, V spans ${f(d.V[0], 2)}–${f(d.V[1], 2)} m/s, Re spans ${f(d.Re[0], 1)}–${Math.round(d.Re[1]).toLocaleString('en-US')}, We spans ${f(d.We[0], 1)}–${f(d.We[1], 0)} and the Ohnesorge number spans ${f(d.Oh[0], 4)}–${f(d.Oh[1], 2)}. Measured β_{max} ranges from ${f(d.beta[0], 2)} to ${f(d.beta[1], 2)} on the textured surfaces and from ${f(d.beta_r[0], 2)} to ${f(d.beta_r[1], 2)} on REF-H.`));
C.push(P(`The textured set contains ${d.n_t.toLocaleString('en-US')} impacts in ${d.cond_t} replicate conditions (surface × fluid × velocity); REF-H contains ${d.n_r} impacts in ${d.cond_r} conditions. An audit found no missing cells and no duplicate rows, and Re and We recomputed from the raw properties agree with the supplied columns to a relative error below 5×10^{−7}. Contact angles are supplied for water on all 12 textured surfaces (advancing angles 162–167°) and for glycerol mixtures on only six, so they were not used as model inputs.`));
C.push(TCAP('**Table 1.** Fluids. Mean properties over all impacts of each fluid, as recorded per impact in the dataset; viscosity range in brackets.'));
C.push(TABLE(['Glycerol (wt%)', 'ρ (kg m^{−3})', 'μ (mPa s)', 'σ (mN m^{−1})', 'Textured impacts', 'REF-H impacts'],
  N.fluids.map(q => [q.glycerol, f(q.rho, 1), q.muhi - q.mulo > 1e-7 ? `${f(q.mu * 1000, 2)} [${f(q.mulo * 1000, 2)}–${f(q.muhi * 1000, 2)}]` : f(q.mu * 1000, 2), f(q.sigma * 1000, 1), q.n_t, q.n_r]), [1300, 1400, 2100, 1400, 1413, 1413]));
C.push(SP());
C.push(H2('2.2 Texture descriptors'));
C.push(P('Channel pitch and depth are not defined for a smooth plate, so each surface is described by two geometric quantities that every surface has. With a track width w,'));
C.push(EQ('φ = [max(s − w, 0) / s]^{2},    V_{tex} = (1 − φ)·h', 2));
C.push(P(`where s is the pitch and h the mean channel depth. φ approximates the fraction of un-ablated plateau and V_{tex} is a texture volume per unit area, in µm. The track widths were estimated from SEM images in an earlier phase of this project as 45 µm for deep and 30 µm for shallow channels, and are held fixed within each depth class. The descriptors are therefore geometric proxies, not per-surface measurements, and all results are conditional on them. REF-H has φ = 1 and V_{tex} = 0. The largest textured φ is ${f(Math.max(...N.surfaces.filter(s => s.surface !== 'REF-H').map(s => s.phi)), 3)}, so REF-H lies outside the textured range of φ (Table 2).`));
C.push(TCAP('**Table 2.** Surfaces and descriptors. S = shallow (≈6 µm), D = deep (≈25 µm) channels.'));
const surfOrder = ['S50', 'S100', 'S200', 'S400', 'S600', 'S800', 'D50', 'D100', 'D200', 'D400', 'D600', 'D800', 'REF-H'];
const SS = Object.fromEntries(N.surfaces.map(s => [s.surface, s]));
C.push(TABLE(['Surface', 'Pitch s (µm)', 'Depth h (µm)', 'φ', 'V_{tex} (µm)', 'Surface', 'Pitch s (µm)', 'Depth h (µm)', 'φ', 'V_{tex} (µm)'],
  [0, 1, 2, 3, 4, 5].map(i => { const a = SS[surfOrder[i]], c = SS[surfOrder[i + 6]]; return [a.surface, a.spacing, a.depth, f(a.phi, 3), f(a.texvol, 2), c.surface, c.spacing, c.depth, f(c.phi, 3), f(c.texvol, 2)]; })
    .concat([['REF-H', '—', '0', '1.000', '0.00', '', '', '', '', '']]), [900, 900, 900, 800, 1000, 900, 900, 900, 800, 1000]));
C.push(SP());

// ---------------- 3 Methods ----------------
C.push(H1('3. Methods'));
C.push(H2('3.1 Evaluation protocols'));
C.push(P('**Leave-one-surface-out (LOSO)** is the primary protocol. Each of the 12 textured surfaces is held out in turn; every step of fitting (feature scaling, target normalisation, hyperparameter optimisation, backbone fitting) uses only the other 11. Models are selected on LOSO error.'));
C.push(P('**Grouped 5-fold cross-validation** by replicate condition (in-distribution, ID) keeps the five replicates of a condition together. It is reported for the model screen as a measure of interpolation.'));
C.push(P('**REF-H** predictions come from models fitted on all textured impacts. REF-H was inspected during earlier development, so it is reported as a stress test outside the textured family, not as a blind test, and it was never used to choose a model.'));
C.push(P('Metrics are the RMSE of β_{max} (dimensionless; not a percentage), the macro RMSE (the mean of the 12 per-surface RMSEs), the worst-surface RMSE and the mean absolute percentage error (MAPE).'));
C.push(H2('3.2 Comparing models'));
C.push(P('LOSO errors are correlated within a held-out surface, so the independent unit is the surface, not the impact or the condition [12]. Differences in RMSE between a candidate and the baseline were assessed with a paired bootstrap that resamples the 12 surfaces with replacement, keeping all impacts of a drawn surface together (4,000 draws, seed 23). The rule, fixed before the comparison, was: replace the baseline only if the 95% interval of the difference excludes zero; otherwise keep the simpler model. During development, resampling replicate conditions instead of surfaces understated the uncertainty and made the Laan backbone appear significantly better; with surfaces resampled it is not (Section 4.2). The earlier XGBoost comparison used the same scheme with 2,000 draws.'));
C.push(H2('3.3 Models'));
C.push(P('All GP models predict y = ln β_{max} from the standardised input vector x = [ln Re, ln We, ln D_{0}, φ, V_{tex}], with D_{0} in mm. The point prediction is the lognormal median exp(μ). The **baseline GP** uses the kernel'));
C.push(EQ('k(x, x′) = c · Matérn_{ν=3/2}(x, x′; ℓ_{1…5}) + σ_{n}^{2} δ(x, x′)', 3));
C.push(P('with one automatic-relevance length scale ℓ per input and a white-noise term. Hyperparameters are set by maximising the marginal likelihood with L-BFGS-B (at most 100 iterations), started from the same fixed values in every fold so that no fold inherits information from another [6, 7]. Four alternatives were fitted under the identical protocol:'));
C.push(B('**Matérn 5/2 GP:** the same model with a smoother kernel (ν = 5/2).'));
C.push(B('**Fluid–surface interaction GP:** k = a·k_{f} + b·k_{s} + c·k_{f}k_{s} + noise, where k_{f} acts on (ln Re, ln We, ln D_{0}) and k_{s} on (φ, V_{tex}), with positive amplitudes. Its analytic gradients were checked against finite differences.'));
C.push(B(`**Laan backbone + residual GP:** ln β_{max} = ln[Re^{1/5} P^{1/2}/(A + P^{1/2})] + g(x), where g is the baseline GP fitted to the residual and A is fitted by least squares in each fold. On all textured impacts A = ${f(N.physA, 2)}, against 1.24 in [3].`));
C.push(B(`**SEM CNN descriptors + GP (experimental):** a 9,070-parameter convolutional network reads three SEM views of a surface (43×, 100×, 350×; footer cropped, resized to 128×128) and predicts φ and V_{tex}/25. It is trained per fold on the 33 images of the 11 training surfaces. Its targets are the geometric proxies of Section 2.2, so these are weak labels. A GP is then fitted on ln Re, ln We, ln D_{0} and the CNN descriptors.`));
C.push(P(`Two ablations were added for this paper: the **Laan law alone** (Eq. 1 with A fitted on the 11 training surfaces in each fold) and the **GP without surface descriptors** (baseline GP on ln Re, ln We and ln D_{0} only). An earlier **screen of 13 learners** used the same five inputs and the same protocols: GP, random forest, extra trees, CatBoost, XGBoost [8], LightGBM, a multilayer perceptron, a deep ensemble of five MLPs, an FT-Transformer, quadratic ridge, k-nearest neighbours, RBF support-vector regression and kernel ridge regression.`));
C.push(H2('3.4 Prediction intervals'));
C.push(P('The nominal 90% interval is exp(μ ± 1.645 σ), where σ is the GP predictive standard deviation of ln β_{max}. To test calibration without using the held-out surface, a nested experiment was run: for each of the 12 outer surfaces, the remaining 11 were split into three inner surface folds, and all four GP candidates were fitted in every inner fold (144 inner fits). Normalised residuals |y − μ|/σ from the inner held-out surfaces set the interval multiplier q as a finite-sample order statistic, and the same inner scores chose the candidate for that outer fold (first candidate within 0.001 macro RMSE of the best). This is empirical cross-validated calibration in the spirit of split conformal prediction [9], not a distribution-free guarantee: impacts share surfaces and a new surface is a distribution shift [10]. Selection on the same scores used for reporting would be optimistic [11]; the nested design avoids that.'));
C.push(H2('3.5 Implementation and reliability checks'));
C.push(P(`The models were implemented in Python with NumPy, SciPy and scikit-learn [7] and exported as float64 Cholesky factors, so that inference needs no pickled objects. A browser predictor runs the same GP in a JavaScript Web Worker. Each prediction returns reliability flags: inputs outside the measured range of D_{0}, V, Re or We; a feature-space anomaly score at or above the training 95th percentile; and an explicit warning for smooth surfaces, whose historical interval coverage is poor. The optimiser reported bound or iteration-limit warnings in ${N.warnings_outer} outer and ${N.warnings_inner} inner fits; these are retained in the result files.`));

// ---------------- 4 Results ----------------
C.push(H1('4. Results'));
C.push(H2('4.1 Model screen'));
C.push(P(`The GP had the lowest error on unseen surfaces among the 13 learners (Table 3), at ${f(zr['Gaussian process'].loso)} against ${f(zr['Random forest'].loso)} for the best tree ensemble (random forest) and ${f(zr['FT-Transformer'].loso)} for the best neural model (FT-Transformer). It was also best in-distribution (${f(lg.gpr_id)} against ${f(lg.xgb_id)} for XGBoost; 95% interval of the difference ${f(lg.id_ci[0])} to ${f(lg.id_ci[1])}). Every learner's worst held-out surface was D50, the densest texture. Kernel ridge and RBF support-vector regression had the lowest REF-H errors (${f(zr['Kernel ridge'].refh)} and ${f(zr['SVR (RBF)'].refh)}) but among the worst LOSO errors (${f(zr['Kernel ridge'].loso)} and ${f(zr['SVR (RBF)'].loso)}). A single smooth plate is therefore a poor basis for choosing a model of textured surfaces.`));
C.push(TCAP('**Table 3.** Screen of 13 learners on the same five inputs. RMSE of β_{max}. Sorted by LOSO RMSE.'));
C.push(TABLE(['Model', 'ID RMSE', 'LOSO RMSE', 'Worst held-out surface', 'REF-H RMSE'],
  zoo.map(z => [z.name, f(z.id), f(z.loso), z.worst, f(z.refh)]), [3100, 1300, 1400, 1926, 1300], { hl: i => i === 0 }));
C.push(SP());
C.push(H2('4.2 GP variants, physics backbone and ablations'));
C.push(P(`Table 4 and Figure 1 compare the GP variants and ablations with the baseline. The Laan law on its own is far from adequate on these surfaces: with A fitted it gives an unseen-surface RMSE of ${f(L.loso)} (MAPE ${f(L.mape, 1)}%), six times the baseline GP. Using it as a backbone under the GP gave the lowest point estimate (${f(ph.loso, 4)} against ${f(b.loso, 4)}), but the difference, ${e3(ph.loso - b.loso)}×10^{−3} with 95% interval ${ci3(phCI)}×10^{−3}, includes zero. The Matérn 5/2 kernel (${e3(m52.loso - b.loso)}, ${ci3(m52CI)}), the interaction kernel (${e3(st.loso - b.loso)}, ${ci3(stCI)}) and nested selection (${e3(nz.rmse - b.loso)}, ${ci3(nestCI)}) are all indistinguishable from the baseline. Under the pre-set rule the baseline GP is retained.`));
C.push(P(`The SEM CNN branch lowered the error to ${f(sm.loso, 4)} and was better on ${N.sem_wins} of 12 surfaces, but its interval ${ci3(semCI)}×10^{−3} includes zero and its REF-H error was higher (${f(sm.refh)} against ${f(b.refh)}). It is kept as an experimental option. XGBoost on the same inputs was clearly worse on unseen surfaces (${e3(x.loso - b.loso)}×10^{−3}, interval ${ci3(xgbCI)}×10^{−3}); the GP had the lower error on ${lg.wins} of 12 surfaces and placed ${pc(lg.within05.GPR, 0)} of impacts within ±0.05 of the measurement, against ${pc(lg.within05.XGB, 0)}.`));
C.push(P(`Removing the two texture descriptors raised the unseen-surface error from ${f(b.loso)} to ${f(bl.loso)} (+${f((bl.loso / b.loso - 1) * 100, 0)}%) and the REF-H error from ${f(b.refh)} to ${f(bl.refh)}. The paired interval, ${ci3(blCI)}×10^{−3}, still includes zero. With 12 surfaces the descriptors help on average but the gain is not established; Figure 2 shows why.`));
C.push(TCAP('**Table 4.** GP variants, ablations and baselines under LOSO and on REF-H. Coverage is the share of impacts inside the 90% interval (GP models only).'));
C.push(TABLE(['Model', 'LOSO RMSE', 'Macro RMSE', 'Worst surface', 'REF-H RMSE', 'REF-H coverage'],
  ['baseline', 'matern52', 'structured', 'physics_residual', 'sem', 'blind', 'xgb', 'laan'].map(k => { const r = R[k];
    return [r.label, f(r.loso, 4), f(r.macro, 4), `${f(r.worst, 3)} (${r.worst_s})`, f(r.refh, 4), r.refh_cov != null ? pc(r.refh_cov) : '—']; }),
  [3226, 1100, 1100, 1300, 1100, 1200], { hl: i => i === 0 }));
C.push(...FIG('fig1_forest.png', 6.2, `**Figure 1.** Change in unseen-surface RMSE relative to the baseline GP, with paired surface-bootstrap 95% intervals (4,000 draws; XGBoost 2,000). Orange marks the only plotted interval that excludes zero. The Laan law alone (+${f((L.loso - b.loso) * 1000, 0)}×10^{−3}) is off the scale; its interval [+203, +217]×10^{−3} also excludes zero.`));
C.push(H2('4.3 Surface by surface'));
C.push(P(`Figure 2 shows that D50 is the hardest surface for every learner in the screen and every GP variant (the Laan law alone is uniformly poor): its φ = 0.010 is the smallest in the set, so holding it out forces extrapolation in φ. There the descriptors matter most (baseline ${f(N.rows[0].worst)} against ${f(bl.worst)} without descriptors and ${f(x.worst)} for XGBoost). The descriptors also help on D100, D400 and D600, but they hurt most on S50, D200 and S600, where the surface-blind GP was more accurate. The descriptors are proxies with a fixed track width per depth class, which plausibly limits how well they separate surfaces of the same depth.`));
C.push(...FIG('fig2_per_surface.png', 6.2, '**Figure 2.** RMSE for each held-out surface: baseline GP, GP without surface descriptors, and XGBoost on the same five inputs.'));
C.push(H2('4.4 Where the error is: viscosity'));
C.push(P(`Predictions on unseen surfaces lie close to the measurements across the full range (Figure 3a). The error, however, is not uniform across fluids (Figure 4). Unseen-surface RMSE falls from ${f(BF[0].loso_rmse)} for water to ${f(BF[91].loso_rmse)} for 91 wt% glycerol, as mean viscosity rises about ${Math.round(N.mu_ratio)}-fold and β_{max} falls. The GP assumes one noise level in ln β_{max} for all fluids, so its intervals are too narrow for water (coverage ${pc(BF[0].loso_cov, 0)}) and 20 wt% glycerol (${pc(BF[20].loso_cov, 0)}), and too wide for 78 and 91 wt% (${pc(BF[78].loso_cov, 0)} and ${pc(BF[91].loso_cov, 0)}). Overall coverage of ${pc(b.loso_cov)} hides this imbalance.`));
C.push(...FIG('fig3_parity.png', 6.2, '**Figure 3.** Predicted against measured β_{max} for the baseline GP. (a) Each textured surface predicted by a model that never saw it. (b) REF-H predicted by the model fitted on all textured surfaces. Colour: glycerol content.'));
C.push(H2('4.5 Calibration and the smooth plate'));
C.push(P(`Per held-out surface, nominal 90% coverage ranged from ${pc(covMin, 0)} (${covMinS}) to ${pc(covMax, 0)}. In the nested experiment, where the interval multiplier was learned only from inner folds, coverage was ${pc(nz.cov)} with per-surface values of ${pc(nz.fold_cov[0])}–${pc(nz.fold_cov[1])} and a mean width of ${f(nz.width)} in β_{max}. The nested selector chose the interaction kernel in 11 of 12 outer folds, yet its error (${f(nz.rmse, 4)}) was no better than the fixed baseline. The deployment multiplier learned from LOSO residuals is q = ${f(nz.q, 3)}, essentially the Gaussian value 1.645: on average across textured surfaces, the GP's own variance is already calibrated.`));
C.push(P(`On REF-H the baseline GP's RMSE was ${f(b.refh)} (MAPE 1.9%) and its 90% intervals held for only ${pc(b.refh_cov)} of impacts. The shortfall is almost entirely in the low-viscosity fluids: coverage was ${pc(BF[0].refh_cov, 0)}, ${pc(BF[20].refh_cov, 0)} and ${pc(BF[60].refh_cov, 0)} for 0, 20 and 60 wt% glycerol, against ${pc(BF[78].refh_cov, 0)} for both 78 and 91 wt% (Figure 4b). The general-purpose anomaly score does not single out REF-H, because its Re, We and D_{0} are ordinary; REF-H is flagged only by the explicit smooth-surface rule.`));
C.push(...FIG('fig4_by_fluid.png', 6.2, '**Figure 4.** Baseline GP by fluid. (a) RMSE on unseen textured surfaces (LOSO) and on REF-H. (b) Share of impacts inside the nominal 90% interval.'));
C.push(H2('4.6 What the GP relies on'));
C.push(P(`In the deployed baseline, fitted on all ${d.n_t.toLocaleString('en-US')} textured impacts, the length scales on standardised inputs are ${f(N.ls.logWe, 1)} for ln We, ${f(N.ls.logRe, 1)} for ln Re, ${f(N.ls.logD, 1)} for ln D_{0}, ${f(N.ls.texvol, 1)} for V_{tex} and ${f(N.ls.phi, 1)} for φ. Shorter length scales mean faster variation of the prediction with that input, so the fluid-dynamic groups carry most of the signal and the texture descriptors act as slow corrections. This agrees with an earlier SHAP analysis on the same model, in which mean |SHAP| was about 0.12–0.13 for ln Re and ln We and about 0.003–0.005 for φ and V_{tex}.`));
C.push(H2('4.7 Implementation check'));
C.push(P('All 18 unit tests pass. They cover data integrity, physical identities, split isolation for impacts and images, kernel gradients, interval ordering, input validation and recomputation of the reported metrics. Across 30 test cases and all five exported GP models, the browser predictor matches Python to within 2.7×10^{−12} in β_{max} and 3.1×10^{−12} in interval bounds, and it rejects six classes of invalid input. The CNN command-line tool, run on the original D200 SEM images, reproduces the named-surface prediction to 10^{−8}.'));

C.push(H2('4.8 Reading the surface from one SEM image'));
C.push(P(`A surface that has not been characterised will rarely come with a channel spacing and depth. Laser tracks are rough and the plateaus between them are smooth, so the flat-area fraction can be measured directly: the local standard deviation of SEM intensity over a ${X.image_reader.window_um} µm window, with one threshold set by Otsu's method on the training surfaces only (recalibrated in every LOSO fold). Pixel size is read from the microscope metadata. On 43× images the reader gives φ = ${f(XR['REF-H'].image_phi, 3)} for the smooth plate (the CNN of Section 4.2 gave 0.71) and agrees with the geometric φ within ${f(Math.max(...Object.entries(XR).filter(([k]) => k !== 'REF-H' && +k.slice(1) >= 200).map(([, v]) => Math.abs(v.image_phi - v.geometry_phi))), 2)} at pitches of 200 µm and above. At 50 µm pitch it finds more flat area than the geometric formula assumes (D50: ${f(XR.D50.image_phi, 2)} vs ${f(XR.D50.geometry_phi, 2)}). A GP on ln Re, ln We, ln D_{0} and the image φ alone, with no spacing or depth, gives LOSO RMSE ${f(XC.image_only.loso.rmse)} against ${f(XC.baseline.loso.rmse)} for the baseline (difference interval ${f(XC.image_only.delta_vs_baseline_ci95[0])} to ${f(XC.image_only.delta_vs_baseline_ci95[1])}) and REF-H RMSE ${f(XC.image_only.refh.rmse)}. It is not better, so the baseline stays the default, but it is an equally accurate route for surfaces known only from an image. The reader is calibrated at 43×: at 100× and 350× it under-reads φ on textured surfaces, so the tool refuses other pixel sizes.`));
C.push(TABLE(['Candidate', 'LOSO RMSE', '95% CI vs baseline', 'REF-H RMSE', 'REF-H coverage'],
  [['Baseline (geometry φ, V_{tex})', 'baseline'], ['Image φ + V_{tex}', 'image_phi'], ['Image φ only', 'image_only'], ['Lee β_{0} correction', 'lee_beta0'], ['Contact angle as input', 'angle_input']].map(([n, k]) => [n, f(XC[k].loso.rmse), k === 'baseline' ? '—' : `${f(XC[k].delta_vs_baseline_ci95[0])} to ${f(XC[k].delta_vs_baseline_ci95[1])}`, f(XC[k].refh.rmse), pc(XC[k].refh.coverage, 0)]),
  [3000, 1300, 2000, 1400, 1300]));
C.push(H2('4.9 Wettability inputs'));
C.push(P(`Two ways of adding the water advancing angle (file 03 of [4]) were tested; REF-H has no measured angle, so ${X.angles_used.match(/assumed ([0-9.]+)/)[1]}° from a similar smooth coating was assumed and swept from 100° to 130°. The Lee et al. [13] correction, fitting the GP to (β_{max}^{2} − β_{0}^{2})^{1/2} with β_{0} from the spherical cap, leaves LOSO unchanged (${f(XC.lee_beta0.loso.rmse)}) but over-predicts REF-H by ${f(XC.lee_beta0.refh.bias, 2)} (RMSE ${f(XC.lee_beta0.refh.rmse)}), and still by ${f(XS['lee_beta0@130'].bias, 2)} at 130°. Using cos θ as a sixth input worsens LOSO to ${f(XC.angle_input.loso.rmse)} (interval excludes zero) and extrapolates badly on REF-H. The training angles span only about 160–167°, so this data cannot teach a wettability effect; the measured smooth-plate spreading is close to the textured surfaces, not to the Lee prediction. Surfaces with intermediate angles are needed.`));
C.push(H2('4.10 Automated measurement from impact video'));
C.push(P('To use the model without manual image analysis, a pipeline measures each backlit side-view recording automatically: one Otsu threshold separates the dark drop and substrate from the bright background; the substrate is the dark band spanning the frame; D_{0} follows from the drop area and V from a straight-line fit of its centroid before contact; the spreading diameter is the largest horizontal extent after contact. The GP then predicts β_{max} from the measured D_{0} and V, and the measurement is checked against the 90% interval. A watch mode processes every new video in a folder. On synthetic videos with known answers the pipeline recovers D_{0} and V to within 0.4% and β_{max} to within about 1%; a browser version gives identical results. Validation on real recordings with known values is the next step; the fluorescence videos of a related liquid-film study were not suitable (no visible drop edge, no scale).'));

// ---------------- 5 Discussion ----------------
C.push(H1('5. Discussion'));
C.push(P('**Why a GP.** The GP interpolates smoothly in the descriptors, whereas tree ensembles produce piecewise-constant responses. On a held-out surface at the edge of the descriptor range (D50), a tree model can only return values seen on its training surfaces, whereas the GP extrapolates more gently. With only 12 distinct surfaces, the data cannot support the extra flexibility of the interaction kernel, the physics backbone or the image branch; each helps on some surfaces and hurts on others.'));
C.push(P(`**Physics.** Equation (1) was derived for smooth, partially wetting surfaces. On these superhydrophobic textures it explains the overall trend but leaves errors of about 11%. As a backbone it adds a physically sensible mean function, which would matter more with fewer data or further extrapolation in Re and We. Within the measured range, the GP learns the same trend from the data. The model is physics-informed through its inputs (dimensionless groups and geometric descriptors) and its logarithmic target rather than through a constraint.`));
C.push(P('**Viscosity.** The concentration of error and under-coverage in the low-viscosity fluids is consistent with a simple picture. When viscous dissipation is weak, spreading is limited by capillarity and by what happens at the contact line and over the texture, so the surface matters more and replicate scatter is larger. Viscous drops lose their energy in the bulk of the lamella, and the surface matters less. This interpretation is a hypothesis; it is testable by fitting a noise model that depends on the Ohnesorge number, which should restore coverage for water without widening the intervals for viscous fluids.'));
C.push(P('**The smooth plate.** REF-H is hydrophobic, not superhydrophobic, and φ = 1 lies outside the textured range. No input describes wettability, so the model cannot know that a drop on REF-H may meet a different contact-line condition. The fact that REF-H errors appear in the same low-viscosity fluids where texture matters most supports wettability as the missing input. Section 4.9 shows that the advancing angles in the dataset are too uniform to learn this from, so new surfaces with intermediate angles are needed. Measured advancing and receding angles for every fluid–surface pair are the most direct fix.'));
C.push(P('**Limitations.** (i) Only 12 independent textured surfaces exist, so surface-level intervals are wide and several apparent gains cannot be confirmed. (ii) The texture descriptors use fixed track widths per depth class rather than per-surface measurements. (iii) REF-H was inspected during development and is not a blind test; no prospective test has been run. (iv) All surfaces share one substrate, laser and pattern family. (v) Some optimiser runs stopped at bounds or iteration limits. (vi) Ranking fixed candidates on the same LOSO scores used to report them carries some selection optimism, which the nested experiment bounds.'));

// ---------------- 6 Conclusions ----------------
C.push(H1('6. Conclusions'));
C.push(P(`On a public dataset of drop impacts on laser-channelled superhydrophobic aluminium, a five-input Gaussian process predicts β_{max} on a surface it has never seen with RMSE ${f(b.loso)} (MAPE ${f(b.mape, 1)}%). That is 29% below XGBoost on identical inputs and six times more accurate than the Laan scaling law alone. Physics backbones, richer kernels, nested selection and an SEM-image CNN did not produce statistically supported improvements. The texture descriptors reduce error on average, but 12 surfaces are too few to confirm it. Prediction intervals are calibrated on average but not per fluid, and they fail on a smooth hydrophobic plate. Both failures point to the same next steps: a viscosity-dependent noise model, measured wettability as an input, and more surfaces, with a prospective test set held back before any further tuning.`));

C.push(P(`For surfaces described only by an SEM image, reading φ from the image gives the same accuracy (${f(XC.image_only.loso.rmse)}), and automatic video measurement makes the model usable on new experiments without manual analysis.`));
C.push(H1('Data and code availability'));
C.push(P('Impact data: Može et al., Mendeley Data, doi:10.17632/wsh8rxwd38.1 [4]. The analysis code, trained model bundles, all fold-level predictions, the browser predictor and the tests are in the Droplet Intelligence v2.1 package accompanying this paper. The two ablations added here (Laan law alone; GP without descriptors) and the figures are reproduced by ablation.py and figs.py. The extensions of Sections 4.8–4.10 are in droplet/extensions.py, droplet/predict_image.py and droplet/video.py, with results in results/extensions/.'));
C.push(H1('Acknowledgements'));
C.push(P('The author thanks Može, Jereb, Lovšin, Berce, Zupančič and Golobič for making their dataset public. An AI assistant was used to help write analysis code and edit the text; the author designed the study, checked every result against the saved result files, and takes responsibility for the content.'));

// ---------------- References ----------------
C.push(H1('References'));
const refs = [
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
];
refs.forEach(r => C.push(NUM(r, 'ref')));

const doc = new Document({
  creator: 'Parth Sharma', title: 'Predicting droplet spreading on unseen laser-textured surfaces',
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: HFONT, size: 25, bold: true, color: '14213D' }, paragraph: { outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: HFONT, size: 21, bold: true, color: '2F4574' }, paragraph: { outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] },
    { reference: 'c', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '(%1)', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 520, hanging: 400 } } } }] },
    { reference: 'ref', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '[%1]', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 520, hanging: 520 } }, run: { size: 19 } } }] },
    { reference: 'num', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 400, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: '777777' })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync('Droplet_Spreading_Paper_v2.docx', buf); console.log('written', buf.length); });
