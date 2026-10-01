**Predicting droplet spreading on laser-textured surfaces the model has never seen**

A surface-held-out evaluation of Gaussian-process, tree, neural and physics-informed models of β<sub>max</sub>

**Parth Sharma**

*Thapar Institute of Engineering & Technology, Patiala, India*

Working paper · 24 September 2026 · prepared for Interfacial Phenomena in Droplets 2026, IISc Bengaluru, 5–7 October 2026

**Abstract**

Data-driven models of the maximum spreading ratio β<sub>max</sub> of impacting drops are usually scored on random splits in which every surface appears in training. They then measure interpolation between replicate impacts, not the question a surface designer asks: how far will a drop spread on a surface that has not been made yet? We re-analyse a public dataset of 1,498 water–glycerol drop impacts on 12 laser-channelled superhydrophobic aluminium surfaces (Može et al., 2025), holding out one entire surface at a time. A further 125 impacts on a smooth hydrophobic plate (REF-H) serve as a stress test. A Gaussian process (GP) on ln Re, ln We, ln D<sub>0</sub> and two geometric texture descriptors predicts β<sub>max</sub> on unseen surfaces with a root-mean-square error (RMSE) of 0.040 (mean absolute percentage error 1.1%). XGBoost on the same inputs gives 0.056 (paired surface-bootstrap 95% interval of the difference 0.006 to 0.028), and the Laan scaling law alone gives 0.250. A Laan-law backbone, a smoother kernel, a fluid–surface interaction kernel, nested model selection and a convolutional network reading SEM images each changed the error by less than 0.002, with intervals that include zero. Removing the texture descriptors raised the error to 0.046, but with only 12 surfaces that interval also includes zero. Nominal 90% prediction intervals covered 90.1% of unseen-surface impacts overall, yet 76% for water and 98% for 91 wt% glycerol. On REF-H the error rose to 0.059 and coverage fell to 62%, with the shortfall concentrated in the low-viscosity fluids. The evidence favours a simple GP. Further gains are more likely from wettability inputs, viscosity-dependent noise and more surfaces than from larger models.

**Keywords:** drop impact; maximum spreading; superhydrophobic surfaces; laser texturing; Gaussian process; leave-one-surface-out validation; uncertainty calibration

1\. Introduction

When a drop hits a solid surface it spreads into a thin lamella, reaches a maximum diameter D<sub>max</sub>, and then retracts or rebounds. The maximum spreading ratio β<sub>max</sub> = D<sub>max</sub>/D<sub>0</sub> controls how much surface a drop wets and for how long, which matters for spray cooling, coating, printing and anti-icing surfaces \[1\]. For a Newtonian liquid, β<sub>max</sub> depends on the Reynolds number Re = ρVD<sub>0</sub>/μ, the Weber number We = ρV<sup>2</sup>D<sub>0</sub>/σ and the surface. Energy arguments give β<sub>max</sub> ∝ Re<sup>1/5</sup> when viscosity limits spreading and β<sub>max</sub> ∝ We<sup>1/2</sup> when capillarity does; Clanet et al. proposed We<sup>1/4</sup> for low-viscosity drops \[2\]. Laan et al. \[3\] joined the two limits with a Padé approximant in the impact parameter P = We·Re<sup>−2/5</sup>:

β<sub>max</sub> Re<sup>−1/5</sup> = P<sup>1/2</sup> / (A + P<sup>1/2</sup>), A ≈ 1.24 (1)

Equation (1) has no term for surface texture. Laser-textured superhydrophobic surfaces change the contact line, trap air and alter viscous losses, so texture effects are exactly what an empirical model has to learn.

Može et al. \[4\] published 1,623 impacts of water and water–glycerol drops on aluminium surfaces with nanosecond-laser channels of six pitches and two depths, plus a smooth hydrophobic reference, recorded at 5,000 frames per second. The same group trained 28 regression models on seven raw inputs and selected an isotropic exponential GP, evaluated with an 80/20 split and 5-fold cross-validation \[5\]. Every impact condition in the dataset is repeated about five times and every surface is present in training under such a split, so that evaluation answers an interpolation question.

This paper asks the harder question: how well can β<sub>max</sub> be predicted for a textured surface that was never seen in training? Its contributions are:

1)  A leave-one-surface-out (LOSO) protocol with a paired bootstrap that resamples whole surfaces, and a decision rule fixed before the comparison.

2)  A benchmark of 13 learners and four GP variants on identical inputs, including a Laan-law physics backbone and a CNN that reads SEM images.

3)  Two ablations that were missing from earlier work: the physics law on its own, and the GP without texture descriptors.

4)  A diagnosis of prediction-interval coverage by fluid and on the smooth plate, showing where and why the model stops being trustworthy.

5)  A tested implementation: a Python API and a browser predictor that agree to 3×10<sup>−12</sup>.

2\. Data

2.1 Source and scope

All impact data come from the public dataset of Može et al. \[4\] (Mendeley Data, doi:10.17632/wsh8rxwd38.1); no new experiments were performed for this study. The substrates are aluminium 1050A plates. Channels were made with a nanosecond fibre laser at pitches s = 50, 100, 200, 400, 600 and 800 µm and two mean depths: about 6 µm (surfaces S50–S800) and 25 µm (D50–D800). REF-H is a smooth hydrophobic aluminium reference. Five fluids were used, from water to 91 wt% glycerol (Table 1). Across all 1,623 impacts, D<sub>0</sub> spans 2.15–2.61 mm, V spans 0.48–1.71 m/s, Re spans 8.9–4,435, We spans 8.0–119 and the Ohnesorge number spans 0.0022–0.37. Measured β<sub>max</sub> ranges from 1.44 to 3.34 on the textured surfaces and from 1.44 to 2.83 on REF-H.

The textured set contains 1,498 impacts in 297 replicate conditions (surface × fluid × velocity); REF-H contains 125 impacts in 25 conditions. An audit found no missing cells and no duplicate rows, and Re and We recomputed from the raw properties agree with the supplied columns to a relative error below 5×10<sup>−7</sup>. Contact angles are supplied for water on all 12 textured surfaces (advancing angles 162–167°) and for glycerol mixtures on only six, so they were not used as model inputs.

**Table 1.** Fluids. Mean properties over all impacts of each fluid, as recorded per impact in the dataset; viscosity range in brackets.

| **Glycerol (wt%)** | **ρ (kg m<sup>−3</sup>)** | **μ (mPa s)**            | **σ (mN m<sup>−1</sup>)** | **Textured impacts** | **REF-H impacts** |
|--------------------|---------------------------|--------------------------|---------------------------|----------------------|-------------------|
| 0                  | 997.2                     | 0.94                     | 72.5                      | 300                  | 25                |
| 20                 | 1045.2                    | 1.60 \[1.55–1.64\]       | 71.1                      | 298                  | 25                |
| 60                 | 1152.2                    | 9.10 \[8.53–9.63\]       | 67.6                      | 300                  | 25                |
| 78                 | 1199.3                    | 36.60 \[35.34–37.69\]    | 66.9                      | 300                  | 25                |
| 91                 | 1230.5                    | 151.20 \[146.32–159.81\] | 65.8                      | 300                  | 25                |

2.2 Texture descriptors

Channel pitch and depth are not defined for a smooth plate, so each surface is described by two geometric quantities that every surface has. With a track width w,

φ = \[max(s − w, 0) / s\]<sup>2</sup>, V<sub>tex</sub> = (1 − φ)·h (2)

where s is the pitch and h the mean channel depth. φ approximates the fraction of un-ablated plateau and V<sub>tex</sub> is a texture volume per unit area, in µm. The track widths were estimated from SEM images in an earlier phase of this project as 45 µm for deep and 30 µm for shallow channels, and are held fixed within each depth class. The descriptors are therefore geometric proxies, not per-surface measurements, and all results are conditional on them. REF-H has φ = 1 and V<sub>tex</sub> = 0. The largest textured φ is 0.926, so REF-H lies outside the textured range of φ (Table 2).

**Table 2.** Surfaces and descriptors. S = shallow (≈6 µm), D = deep (≈25 µm) channels.

| **Surface** | **Pitch s (µm)** | **Depth h (µm)** | **φ** | **V<sub>tex</sub> (µm)** | **Surface** | **Pitch s (µm)** | **Depth h (µm)** | **φ** | **V<sub>tex</sub> (µm)** |
|-------------|------------------|------------------|-------|--------------------------|-------------|------------------|------------------|-------|--------------------------|
| S50         | 50               | 6                | 0.160 | 5.04                     | D50         | 50               | 25               | 0.010 | 24.75                    |
| S100        | 100              | 6                | 0.490 | 3.06                     | D100        | 100              | 25               | 0.302 | 17.44                    |
| S200        | 200              | 6                | 0.722 | 1.67                     | D200        | 200              | 25               | 0.601 | 9.98                     |
| S400        | 400              | 6                | 0.856 | 0.87                     | D400        | 400              | 25               | 0.788 | 5.31                     |
| S600        | 600              | 6                | 0.902 | 0.59                     | D600        | 600              | 25               | 0.856 | 3.61                     |
| S800        | 800              | 6                | 0.926 | 0.44                     | D800        | 800              | 25               | 0.891 | 2.73                     |
| REF-H       | —                | 0                | 1.000 | 0.00                     |             |                  |                  |       |                          |

3\. Methods

3.1 Evaluation protocols

**Leave-one-surface-out (LOSO)** is the primary protocol. Each of the 12 textured surfaces is held out in turn; every step of fitting (feature scaling, target normalisation, hyperparameter optimisation, backbone fitting) uses only the other 11. Models are selected on LOSO error.

**Grouped 5-fold cross-validation** by replicate condition (in-distribution, ID) keeps the five replicates of a condition together. It is reported for the model screen as a measure of interpolation.

**REF-H** predictions come from models fitted on all textured impacts. REF-H was inspected during earlier development, so it is reported as a stress test outside the textured family, not as a blind test, and it was never used to choose a model.

Metrics are the RMSE of β<sub>max</sub> (dimensionless; not a percentage), the macro RMSE (the mean of the 12 per-surface RMSEs), the worst-surface RMSE and the mean absolute percentage error (MAPE).

3.2 Comparing models

LOSO errors are correlated within a held-out surface, so the independent unit is the surface, not the impact or the condition \[12\]. Differences in RMSE between a candidate and the baseline were assessed with a paired bootstrap that resamples the 12 surfaces with replacement, keeping all impacts of a drawn surface together (4,000 draws, seed 23). The rule, fixed before the comparison, was: replace the baseline only if the 95% interval of the difference excludes zero; otherwise keep the simpler model. During development, resampling replicate conditions instead of surfaces understated the uncertainty and made the Laan backbone appear significantly better; with surfaces resampled it is not (Section 4.2). The earlier XGBoost comparison used the same scheme with 2,000 draws.

3.3 Models

All GP models predict y = ln β<sub>max</sub> from the standardised input vector x = \[ln Re, ln We, ln D<sub>0</sub>, φ, V<sub>tex</sub>\], with D<sub>0</sub> in mm. The point prediction is the lognormal median exp(μ). The **baseline GP** uses the kernel

k(x, x′) = c · Matérn<sub>ν=3/2</sub>(x, x′; ℓ<sub>1…5</sub>) + σ<sub>n</sub><sup>2</sup> δ(x, x′) (3)

with one automatic-relevance length scale ℓ per input and a white-noise term. Hyperparameters are set by maximising the marginal likelihood with L-BFGS-B (at most 100 iterations), started from the same fixed values in every fold so that no fold inherits information from another \[6, 7\]. Four alternatives were fitted under the identical protocol:

- **Matérn 5/2 GP:** the same model with a smoother kernel (ν = 5/2).

- **Fluid–surface interaction GP:** k = a·k<sub>f</sub> + b·k<sub>s</sub> + c·k<sub>f</sub>k<sub>s</sub> + noise, where k<sub>f</sub> acts on (ln Re, ln We, ln D<sub>0</sub>) and k<sub>s</sub> on (φ, V<sub>tex</sub>), with positive amplitudes. Its analytic gradients were checked against finite differences.

- **Laan backbone + residual GP:** ln β<sub>max</sub> = ln\[Re<sup>1/5</sup> P<sup>1/2</sup>/(A + P<sup>1/2</sup>)\] + g(x), where g is the baseline GP fitted to the residual and A is fitted by least squares in each fold. On all textured impacts A = 1.14, against 1.24 in \[3\].

- **SEM CNN descriptors + GP (experimental):** a 9,070-parameter convolutional network reads three SEM views of a surface (43×, 100×, 350×; footer cropped, resized to 128×128) and predicts φ and V<sub>tex</sub>/25. It is trained per fold on the 33 images of the 11 training surfaces. Its targets are the geometric proxies of Section 2.2, so these are weak labels. A GP is then fitted on ln Re, ln We, ln D<sub>0</sub> and the CNN descriptors.

Two ablations were added for this paper: the **Laan law alone** (Eq. 1 with A fitted on the 11 training surfaces in each fold) and the **GP without surface descriptors** (baseline GP on ln Re, ln We and ln D<sub>0</sub> only). An earlier **screen of 13 learners** used the same five inputs and the same protocols: GP, random forest, extra trees, CatBoost, XGBoost \[8\], LightGBM, a multilayer perceptron, a deep ensemble of five MLPs, an FT-Transformer, quadratic ridge, k-nearest neighbours, RBF support-vector regression and kernel ridge regression.

3.4 Prediction intervals

The nominal 90% interval is exp(μ ± 1.645 σ), where σ is the GP predictive standard deviation of ln β<sub>max</sub>. To test calibration without using the held-out surface, a nested experiment was run: for each of the 12 outer surfaces, the remaining 11 were split into three inner surface folds, and all four GP candidates were fitted in every inner fold (144 inner fits). Normalised residuals \|y − μ\|/σ from the inner held-out surfaces set the interval multiplier q as a finite-sample order statistic, and the same inner scores chose the candidate for that outer fold (first candidate within 0.001 macro RMSE of the best). This is empirical cross-validated calibration in the spirit of split conformal prediction \[9\], not a distribution-free guarantee: impacts share surfaces and a new surface is a distribution shift \[10\]. Selection on the same scores used for reporting would be optimistic \[11\]; the nested design avoids that.

3.5 Implementation and reliability checks

The models were implemented in Python with NumPy, SciPy and scikit-learn \[7\] and exported as float64 Cholesky factors, so that inference needs no pickled objects. A browser predictor runs the same GP in a JavaScript Web Worker. Each prediction returns reliability flags: inputs outside the measured range of D<sub>0</sub>, V, Re or We; a feature-space anomaly score at or above the training 95th percentile; and an explicit warning for smooth surfaces, whose historical interval coverage is poor. The optimiser reported bound or iteration-limit warnings in 12/48 outer and 48/144 inner fits; these are retained in the result files.

4\. Results

4.1 Model screen

The GP had the lowest error on unseen surfaces among the 13 learners (Table 3), at 0.040 against 0.046 for the best tree ensemble (random forest) and 0.051 for the best neural model (FT-Transformer). It was also best in-distribution (0.036 against 0.046 for XGBoost; 95% interval of the difference -0.015 to -0.005). Every learner's worst held-out surface was D50, the densest texture. Kernel ridge and RBF support-vector regression had the lowest REF-H errors (0.048 and 0.049) but among the worst LOSO errors (0.257 and 0.108). A single smooth plate is therefore a poor basis for choosing a model of textured surfaces.

**Table 3.** Screen of 13 learners on the same five inputs. RMSE of β<sub>max</sub>. Sorted by LOSO RMSE.

| **Model**             | **ID RMSE** | **LOSO RMSE** | **Worst held-out surface** | **REF-H RMSE** |
|-----------------------|-------------|---------------|----------------------------|----------------|
| Gaussian process      | 0.036       | 0.040         | 0.059 (D50)                | 0.059          |
| Random forest         | 0.044       | 0.046         | 0.080 (D50)                | 0.063          |
| Extra trees           | 0.044       | 0.046         | 0.096 (D50)                | 0.063          |
| FT-Transformer        | 0.045       | 0.051         | 0.098 (D50)                | 0.067          |
| CatBoost              | 0.045       | 0.052         | 0.114 (D50)                | 0.063          |
| Deep ensemble (5 MLP) | 0.045       | 0.053         | 0.101 (D50)                | 0.067          |
| XGBoost               | 0.046       | 0.056         | 0.115 (D50)                | 0.059          |
| LightGBM              | 0.051       | 0.059         | 0.136 (D50)                | 0.060          |
| Quadratic ridge       | 0.054       | 0.064         | 0.135 (D50)                | 0.070          |
| MLP                   | 0.052       | 0.070         | 0.167 (D50)                | 0.082          |
| kNN                   | 0.132       | 0.076         | 0.178 (D50)                | 0.061          |
| SVR (RBF)             | 0.045       | 0.108         | 0.305 (D50)                | 0.049          |
| Kernel ridge          | 0.044       | 0.257         | 0.825 (D50)                | 0.048          |

4.2 GP variants, physics backbone and ablations

Table 4 and Figure 1 compare the GP variants and ablations with the baseline. The Laan law on its own is far from adequate on these surfaces: with A fitted it gives an unseen-surface RMSE of 0.250 (MAPE 10.9%), six times the baseline GP. Using it as a backbone under the GP gave the lowest point estimate (0.0388 against 0.0400), but the difference, −1.2×10<sup>−3</sup> with 95% interval \[−3.0, +0.1\]×10<sup>−3</sup>, includes zero. The Matérn 5/2 kernel (−0.2, \[−0.9, +0.5\]), the interaction kernel (+1.2, \[−4.3, +7.6\]) and nested selection (+1.5, \[−4.0, +7.8\]) are all indistinguishable from the baseline. Under the pre-set rule the baseline GP is retained.

The SEM CNN branch lowered the error to 0.0384 and was better on 8 of 12 surfaces, but its interval \[−3.8, +0.5\]×10<sup>−3</sup> includes zero and its REF-H error was higher (0.065 against 0.059). It is kept as an experimental option. XGBoost on the same inputs was clearly worse on unseen surfaces (+16.4×10<sup>−3</sup>, interval \[+5.7, +27.8\]×10<sup>−3</sup>); the GP had the lower error on 9 of 12 surfaces and placed 87% of impacts within ±0.05 of the measurement, against 81%.

Removing the two texture descriptors raised the unseen-surface error from 0.040 to 0.046 (+16%) and the REF-H error from 0.059 to 0.069. The paired interval, \[−2.6, +14.4\]×10<sup>−3</sup>, still includes zero. With 12 surfaces the descriptors help on average but the gain is not established; Figure 2 shows why.

**Table 4.** GP variants, ablations and baselines under LOSO and on REF-H. Coverage is the share of impacts inside the 90% interval (GP models only).

| **Model**                                 | **LOSO RMSE** | **Macro RMSE** | **Worst surface** | **REF-H RMSE** | **REF-H coverage** |
|-------------------------------------------|---------------|----------------|-------------------|----------------|--------------------|
| Baseline GP, Matérn 3/2 (selected)        | 0.0400        | 0.0391         | 0.059 (D50)       | 0.0591         | 61.6%              |
| GP, Matérn 5/2                            | 0.0398        | 0.0389         | 0.058 (D50)       | 0.0590         | 63.2%              |
| GP, fluid–surface interaction kernel      | 0.0412        | 0.0387         | 0.082 (D50)       | 0.0621         | 57.6%              |
| Laan backbone + residual GP               | 0.0388        | 0.0383         | 0.051 (D50)       | 0.0593         | 60.8%              |
| SEM CNN descriptors + GP (experimental)   | 0.0384        | 0.0375         | 0.054 (D50)       | 0.0648         | 59.2%              |
| GP without surface descriptors (ablation) | 0.0462        | 0.0428         | 0.091 (D50)       | 0.0690         | —                  |
| XGBoost, same five inputs                 | 0.0563        | 0.0516         | 0.115 (D50)       | 0.0593         | —                  |
| Laan scaling law only (A fitted)          | 0.2503        | 0.2501         | 0.261 (S800)      | 0.2528         | —                  |

<img src="media/69323850b79c0e6abbe5005d76ee01beeb19a629.png" title="fig1_forest.png" style="width:6.2in;height:2.1776in" alt="Figure 1. Change in unseen-surface RMSE relative to the baseline GP, with paired surface-bootstrap 95% intervals (4,000 draws; XGBoost 2,000). Orange marks the only plotted interval that excludes zero. The Laan law alone (+210×10−3) is off the scale; its interval [+203, +217]×10−3 also excludes zero." />

**Figure 1.** Change in unseen-surface RMSE relative to the baseline GP, with paired surface-bootstrap 95% intervals (4,000 draws; XGBoost 2,000). Orange marks the only plotted interval that excludes zero. The Laan law alone (+210×10<sup>−3</sup>) is off the scale; its interval \[+203, +217\]×10<sup>−3</sup> also excludes zero.

4.3 Surface by surface

Figure 2 shows that D50 is the hardest surface for every learner in the screen and every GP variant (the Laan law alone is uniformly poor): its φ = 0.010 is the smallest in the set, so holding it out forces extrapolation in φ. There the descriptors matter most (baseline 0.059 against 0.091 without descriptors and 0.115 for XGBoost). The descriptors also help on D100, D400 and D600, but they hurt most on S50, D200 and S600, where the surface-blind GP was more accurate. The descriptors are proxies with a fixed track width per depth class, which plausibly limits how well they separate surfaces of the same depth.

<img src="media/931cc07c355134314f98825ab356823b33c84005.png" title="fig2_per_surface.png" style="width:6.2in;height:3.01339in" alt="Figure 2. RMSE for each held-out surface: baseline GP, GP without surface descriptors, and XGBoost on the same five inputs." />

**Figure 2.** RMSE for each held-out surface: baseline GP, GP without surface descriptors, and XGBoost on the same five inputs.

4.4 Where the error is: viscosity

Predictions on unseen surfaces lie close to the measurements across the full range (Figure 3a). The error, however, is not uniform across fluids (Figure 4). Unseen-surface RMSE falls from 0.062 for water to 0.016 for 91 wt% glycerol, as mean viscosity rises about 160-fold and β<sub>max</sub> falls. The GP assumes one noise level in ln β<sub>max</sub> for all fluids, so its intervals are too narrow for water (coverage 76%) and 20 wt% glycerol (85%), and too wide for 78 and 91 wt% (97% and 98%). Overall coverage of 90.1% hides this imbalance.

<img src="media/16c7e81bd0a5b5ffcf44b06064995b449662a231.png" title="fig3_parity.png" style="width:6.2in;height:3.27888in" alt="Figure 3. Predicted against measured βmax for the baseline GP. (a) Each textured surface predicted by a model that never saw it. (b) REF-H predicted by the model fitted on all textured surfaces. Colour: glycerol content." />

**Figure 3.** Predicted against measured β<sub>max</sub> for the baseline GP. (a) Each textured surface predicted by a model that never saw it. (b) REF-H predicted by the model fitted on all textured surfaces. Colour: glycerol content.

4.5 Calibration and the smooth plate

Per held-out surface, nominal 90% coverage ranged from 84% (D800) to 96%. In the nested experiment, where the interval multiplier was learned only from inner folds, coverage was 92.6% with per-surface values of 86.4%–99.2% and a mean width of 0.115 in β<sub>max</sub>. The nested selector chose the interaction kernel in 11 of 12 outer folds, yet its error (0.0415) was no better than the fixed baseline. The deployment multiplier learned from LOSO residuals is q = 1.645, essentially the Gaussian value 1.645: on average across textured surfaces, the GP's own variance is already calibrated.

On REF-H the baseline GP's RMSE was 0.059 (MAPE 1.9%) and its 90% intervals held for only 61.6% of impacts. The shortfall is almost entirely in the low-viscosity fluids: coverage was 36%, 24% and 56% for 0, 20 and 60 wt% glycerol, against 96% for both 78 and 91 wt% (Figure 4b). The general-purpose anomaly score does not single out REF-H, because its Re, We and D<sub>0</sub> are ordinary; REF-H is flagged only by the explicit smooth-surface rule.

<img src="media/08159dd38964a1c0245320be7bc0ef9561fa6fa4.png" title="fig4_by_fluid.png" style="width:6.2in;height:2.70919in" alt="Figure 4. Baseline GP by fluid. (a) RMSE on unseen textured surfaces (LOSO) and on REF-H. (b) Share of impacts inside the nominal 90% interval." />

**Figure 4.** Baseline GP by fluid. (a) RMSE on unseen textured surfaces (LOSO) and on REF-H. (b) Share of impacts inside the nominal 90% interval.

4.6 What the GP relies on

In the deployed baseline, fitted on all 1,498 textured impacts, the length scales on standardised inputs are 8.5 for ln We, 9.4 for ln Re, 12.7 for ln D<sub>0</sub>, 23.8 for V<sub>tex</sub> and 25.2 for φ. Shorter length scales mean faster variation of the prediction with that input, so the fluid-dynamic groups carry most of the signal and the texture descriptors act as slow corrections. This agrees with an earlier SHAP analysis on the same model, in which mean \|SHAP\| was about 0.12–0.13 for ln Re and ln We and about 0.003–0.005 for φ and V<sub>tex</sub>.

4.7 Implementation check

All 18 unit tests pass. They cover data integrity, physical identities, split isolation for impacts and images, kernel gradients, interval ordering, input validation and recomputation of the reported metrics. Across 30 test cases and all five exported GP models, the browser predictor matches Python to within 2.7×10<sup>−12</sup> in β<sub>max</sub> and 3.1×10<sup>−12</sup> in interval bounds, and it rejects six classes of invalid input. The CNN command-line tool, run on the original D200 SEM images, reproduces the named-surface prediction to 10<sup>−8</sup>.

5\. Discussion

**Why a GP.** The GP interpolates smoothly in the descriptors, whereas tree ensembles produce piecewise-constant responses. On a held-out surface at the edge of the descriptor range (D50), a tree model can only return values seen on its training surfaces, whereas the GP extrapolates more gently. With only 12 distinct surfaces, the data cannot support the extra flexibility of the interaction kernel, the physics backbone or the image branch; each helps on some surfaces and hurts on others.

**Physics.** Equation (1) was derived for smooth, partially wetting surfaces. On these superhydrophobic textures it explains the overall trend but leaves errors of about 11%. As a backbone it adds a physically sensible mean function, which would matter more with fewer data or further extrapolation in Re and We. Within the measured range, the GP learns the same trend from the data. The model is physics-informed through its inputs (dimensionless groups and geometric descriptors) and its logarithmic target rather than through a constraint.

**Viscosity.** The concentration of error and under-coverage in the low-viscosity fluids is consistent with a simple picture. When viscous dissipation is weak, spreading is limited by capillarity and by what happens at the contact line and over the texture, so the surface matters more and replicate scatter is larger. Viscous drops lose their energy in the bulk of the lamella, and the surface matters less. This interpretation is a hypothesis; it is testable by fitting a noise model that depends on the Ohnesorge number, which should restore coverage for water without widening the intervals for viscous fluids.

**The smooth plate.** REF-H is hydrophobic, not superhydrophobic, and φ = 1 lies outside the textured range. No input describes wettability, so the model cannot know that a drop on REF-H may meet a different contact-line condition. The fact that REF-H errors appear in the same low-viscosity fluids where texture matters most supports wettability as the missing input. Measured advancing and receding angles for every fluid–surface pair are the most direct fix.

**Limitations.** (i) Only 12 independent textured surfaces exist, so surface-level intervals are wide and several apparent gains cannot be confirmed. (ii) The texture descriptors use fixed track widths per depth class rather than per-surface measurements. (iii) REF-H was inspected during development and is not a blind test; no prospective test has been run. (iv) All surfaces share one substrate, laser and pattern family. (v) Some optimiser runs stopped at bounds or iteration limits. (vi) Ranking fixed candidates on the same LOSO scores used to report them carries some selection optimism, which the nested experiment bounds.

6\. Conclusions

On a public dataset of drop impacts on laser-channelled superhydrophobic aluminium, a five-input Gaussian process predicts β<sub>max</sub> on a surface it has never seen with RMSE 0.040 (MAPE 1.1%). That is 29% below XGBoost on identical inputs and six times more accurate than the Laan scaling law alone. Physics backbones, richer kernels, nested selection and an SEM-image CNN did not produce statistically supported improvements. The texture descriptors reduce error on average, but 12 surfaces are too few to confirm it. Prediction intervals are calibrated on average but not per fluid, and they fail on a smooth hydrophobic plate. Both failures point to the same next steps: a viscosity-dependent noise model, measured wettability as an input, and more surfaces, with a prospective test set held back before any further tuning.

Data and code availability

Impact data: Može et al., Mendeley Data, doi:10.17632/wsh8rxwd38.1 \[4\]. The analysis code, trained model bundles, all fold-level predictions, the browser predictor and the tests are in the Droplet Intelligence v2.1 package accompanying this paper. The two ablations added here (Laan law alone; GP without descriptors) and the figures are reproduced by ablation.py and figs.py.

Acknowledgements

The author thanks Može, Jereb, Lovšin, Berce, Zupančič and Golobič for making their dataset public. Analysis code and drafts of this paper were prepared with the assistance of an AI model (Claude, Anthropic); every number in the paper was regenerated from saved result files and checked against them.

References

1.  C. Josserand, S. T. Thoroddsen. Drop impact on a solid surface. *Annual Review of Fluid Mechanics* 48 (2016) 365–391.

2.  C. Clanet, C. Béguin, D. Richard, D. Quéré. Maximal deformation of an impacting drop. *Journal of Fluid Mechanics* 517 (2004) 199–208.

3.  N. Laan, K. G. de Bruin, D. Bartolo, C. Josserand, D. Bonn. Maximum diameter of impacting liquid droplets. *Physical Review Applied* 2 (2014) 044018. doi:10.1103/PhysRevApplied.2.044018

4.  M. Može, S. Jereb, R. Lovšin, J. Berce, M. Zupančič, I. Golobič. Dataset on droplet spreading and rebound behavior of water and viscous water-glycerol mixtures on superhydrophobic surfaces with laser-made channels. *Data in Brief* 61 (2025) 111697. doi:10.1016/j.dib.2025.111697. Data: doi:10.17632/wsh8rxwd38.1

5.  S. Jereb, J. Berce, R. Lovšin, M. Zupančič, M. Može, I. Golobič. Investigation of droplet spreading and rebound dynamics on superhydrophobic surfaces using machine learning. *Biomimetics* 10 (2025) 357. doi:10.3390/biomimetics10060357

6.  C. E. Rasmussen, C. K. I. Williams. *Gaussian Processes for Machine Learning*. MIT Press, 2006.

7.  F. Pedregosa et al. Scikit-learn: machine learning in Python. *Journal of Machine Learning Research* 12 (2011) 2825–2830.

8.  T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. *Proc. 22nd ACM SIGKDD* (2016) 785–794.

9.  A. N. Angelopoulos, S. Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511 (2021).

10. R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani. Conformal prediction beyond exchangeability. *Annals of Statistics* 51 (2023) 816–845.

11. G. C. Cawley, N. L. C. Talbot. On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research* 11 (2010) 2079–2107.

12. D. R. Roberts et al. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40 (2017) 913–929.

Appendix A. Corrections to the draft of 23 September 2026

The earlier draft (Droplet_Impact_Research_Paper.md) and the README.md and pitch_scripts/PITCH.md inside the four-part download of 23 September contain statements that are not supported by the data or result files. None of them should be used. The source of each error is given in brackets.

The pitch script saved in the project (claude/pitch_scripts.md) was re-checked against the result files on 24 September and updated: REF-H is now described as a held-out stress test rather than a blind test, the dataset is credited to Može et al. \[4\], and the viscosity finding of Section 4.4 is included. Its remaining figures are traceable: Laan R<sup>2</sup> = 0.71 is Eq. (1) with the published A = 1.24 on the textured impacts (with A refitted on held-out surfaces, R<sup>2</sup> = 0.66); 0.043 vs 0.039 is the earlier single-descriptor CNN (macro RMSE), not the v2.1 three-view CNN (0.0384); and 83–97% per-surface coverage refers to the earlier split-conformal intervals (nominal intervals: 84%–96%).

| **Item**              | **Draft of 23 Sept**                                                                                                                                                            | **Correct**                                                                                 |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Authorship of data    | Data Collection section presents the experiments as the author's own: femtosecond laser, 25 kHz camera, manual SEM segmentation (draft); "We measured 1,498 impacts" (PITCH.md) | Public dataset of Može et al. \[4\]: nanosecond fibre laser, 5,000 fps. No new experiments. |
| REF-H RMSE            | 0.0394                                                                                                                                                                          | 0.0591                                                                                      |
| β<sub>max</sub> range | 0.2–1.0                                                                                                                                                                         | 1.44–3.34                                                                                   |
| Surfaces              | Included "D1200"                                                                                                                                                                | S50–S800 and D50–D800; no D1200                                                             |
| Descriptors           | φ 0.3–0.8 and V<sub>tex</sub> 50–800 µm<sup>3</sup>, "measured from SEM"                                                                                                        | φ 0.01–0.93, V<sub>tex</sub> 0.44–24.75 µm; geometric proxies with fixed track widths       |
| Kernel and logs       | Matérn 5/2, base-10 logs                                                                                                                                                        | Matérn 3/2, natural logs                                                                    |
| Laan law alone        | LOSO RMSE 0.073                                                                                                                                                                 | 0.250                                                                                       |
| ID RMSE               | GPR 0.0205, XGBoost 0.0187 (XGBoost better)                                                                                                                                     | GPR 0.0363, XGBoost 0.0462 (GPR better)                                                     |
| Other models          | M3–M5 LOSO 0.052–0.054                                                                                                                                                          | Not produced by any run; replaced by Table 3                                                |
| Without descriptors   | "about 0.048"                                                                                                                                                                   | Measured 0.0462; interval includes zero                                                     |
| Within ±0.05          | GPR 97%, XGBoost 89%                                                                                                                                                            | GPR 87%, XGBoost 81%                                                                        |
| Physics backbone CI   | \[−0.0022, +0.0001\]                                                                                                                                                            | \[-0.0030, +0.0001\]                                                                        |
| Interval details      | Per-surface q values; "D1200 83%"                                                                                                                                               | Per-surface coverage 84%–96%; not reported per surface q                                    |
| Unsupported claims    | "99.6% accuracy" (PITCH.md); "~\$50 USD per droplet impact" (draft)                                                                                                             | No source; removed                                                                          |
| Authorship            | Claude listed as author                                                                                                                                                         | AI assistance disclosed in Acknowledgements                                                 |

Appendix B. Project log (summary of the working sessions)

This is a factual summary of the work done with the AI assistant between 23 and 24 September 2026.

- **23 Sept — analysis pipeline.** The pipeline for Phases 3–7 was automated on the earlier project: physics-residual models, paired bootstrap comparisons, split-conformal intervals, OOD scoring and SHAP. An audit found that the bootstrap resampled replicate conditions rather than surfaces; after the fix the Laan backbone was no longer significant, and the plain GP was kept. Conformal coverage was made honest by calibrating each surface on the other 11.

- **23 Sept — comparison and tools.** XGBoost was refitted on the exact GP inputs and a three-panel comparison figure was produced (LOSO 0.0400 against 0.0563; GP better on 9 of 12 surfaces). A Streamlit dashboard, a FastAPI service and a browser predictor were built; the browser matched Python to 10<sup>−12</sup>. Four poster panels and the pitch scripts were updated.

- **23 Sept — packaging and first draft.** All code and results were packaged. The single 53 MB archive exceeded the 30 MB upload limit, so it was sent as four parts. A research-paper draft was then written after the conversation context had been summarised; it contained the errors listed in Appendix A.

- **23 Sept — Astra question.** You asked about "Astra". My answers covered DataStax Astra DB with Claude and with GPT; you then clarified that you meant a ChatGPT model called Astra, which I did not identify. That question is unresolved. Those answers also contained inaccuracies: Claude's weights are not publicly available, and Anthropic does not offer an embeddings API.

- **23 Sept, 23:00.** You sent Droplet Intelligence v2.1. It was installed from requirements-sem.txt and served on port 8000. Health, prediction for all five models, custom surfaces, input validation (HTTP 415/422) and extrapolation flags all worked. All 18 tests passed, the JavaScript–Python parity check passed, and the SEM command-line tool worked on the original TIFFs (the bundled JPEG thumbnails are rejected by design). The REF-H error in the paper draft was flagged.

- **24 Sept.** This paper was rebuilt from the v2.1 result files. Two missing analyses were run (Laan law alone; GP without descriptors), four figures were generated from saved predictions, the dataset was traced to its published source, and every number was checked against the result files.
