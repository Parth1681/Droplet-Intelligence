# Droplet Intelligence v2 — implementation and evaluation report

Prepared for Parth Sharma · 23 September 2026

## Decision

**Retain the five-feature Matérn 3/2 GP as the default.** Four models were actually trained and evaluated. The Laan-residual candidate has the lowest fixed-candidate LOSO error, but its paired surface-bootstrap interval includes zero. The nested selector does not improve on the fixed baseline. This release improves reproducibility, model comparison, input handling, calibration isolation and deployment; it does not establish a statistically supported prediction-accuracy breakthrough.

## Measured results

| Model | LOSO RMSE | Macro surface RMSE | Worst surface RMSE | REF-H RMSE | REF-H interval coverage |
|---|---:|---:|---:|---:|---:|
| Baseline Matérn 3/2 GP | 0.039973 | 0.039136 | 0.059353 | 0.059096 | 61.6% |
| Matérn 5/2 GP | 0.039767 | 0.038911 | 0.057743 | 0.058951 | 63.2% |
| Fluid–surface interaction GP | 0.041156 | 0.038750 | 0.081560 | 0.062063 | 57.6% |
| Laan + residual GP | 0.038767 | 0.038257 | 0.051283 | 0.059279 | 60.8% |

All RMSE values are in the dimensionless beta target, not percentages. Point predictions are lognormal medians. REF-H was previously inspected during the original project; these are stress-test results, not a new blind evaluation.

### Paired comparisons to the fixed baseline

| Candidate | RMSE difference | Surface-bootstrap 95% interval |
|---|---:|---:|
| Matérn 5/2 GP | -0.000206 | [-0.000929, +0.000496] |
| Fluid–surface interaction GP | +0.001183 | [-0.004318, +0.007627] |
| Laan + residual GP | -0.001205 | [-0.002958, +0.000081] |

Bootstrap uses 4,000 paired resamples of all 12 surfaces with replacement (seed 23). Rows within each resampled surface stay together. Intervals quantify the saved prediction comparison, not all sources of retraining or model-development uncertainty. Ranking candidates on these same outer scores incurs selection optimism.

## Nested selection and calibration

A separate nested experiment ran **144 inner fits** (12 outer surfaces × 3 inner surface folds × 4 candidates) and reused the 48 independent outer fits. No outer surface enters an inner training set, candidate choice, or residual calibration pool.

- Nested selection RMSE: **0.041474**.
- Difference from fixed baseline, 95% surface-bootstrap interval: [-0.003964, +0.007769].
- Empirical interval coverage: **92.59%**; mean interval width: 0.115289.
- Per-surface coverage: 86.4%–99.2%.

The rule chooses the first candidate within 0.001 macro RMSE of the inner best, in baseline / Matérn 5/2 / structured / residual order. This tolerance was coded before seeing the nested outputs. The final deployment applies the same rule to all textured-surface LOSO folds and chooses the baseline. The nested figure describes the selection procedure, not the final single model.

Calibration uses normalized log residuals from inner held-out surface folds, then the finite-sample order-statistic rank. This is **empirical cross-validated calibration**, not ordinary split conformal: the inner models use smaller training sets than the outer model, samples share surfaces, and distribution shift can invalidate exchangeability. No distribution-free per-surface guarantee is claimed.

Deployment q is obtained from the chosen candidate’s full LOSO residuals and applied after fitting it on all textured rows. This CV-to-full-fit transfer is empirical and may change calibration. The live display says “90% target interval,” and smooth/OOD requests are explicitly flagged. The wide interval is a sensitivity band, not a guarantee.

## Data audit

- 1498 textured impacts, 12 surfaces, 297 replicate conditions.
- 125 separate REF-H impacts, 25 conditions; 1,623 impacts total.
- No missing cells or fully duplicated raw textured rows detected.
- Textured beta range: 1.4429–3.3422.
- Recomputed Re and We agree with provided columns to relative error below 5×10⁻⁷.

The supplied CSV contact-angle files include water angles for 12 surfaces and glycerol-mixture advancing angles for 6. The 91 wt.% static-angle table is not an advancing-angle table. The missing fluid–surface measurements were not fabricated or interpolated into the predictor.

The geometry descriptors use the original fixed track widths (45 µm for deep tracks, 30 µm for shallow tracks), phi=(max(spacing-width,0)/spacing)^2 and texture volume per area=(1-phi)×depth. Their units are µm, not µm³. This evaluation is conditional on those supplied descriptors. SEM extraction was not repeated, and held-out-surface independence of the historical width-estimation process is not established.

## Implemented candidate models

All candidates use natural-log Re, We and diameter in mm, plus phi and texture volume per area. Each fit estimates its own StandardScaler, target normalization, kernel hyperparameters and white-noise variance. Optimizer initialization is independent of all held-out labels. The target is ln(beta).

1. Baseline: constant × Matérn 3/2 ARD + white noise.
2. Smoothness alternative: constant × Matérn 5/2 ARD + white noise.
3. Structured GP: a·kf + b·ks + c·kf·ks + white noise, with shared fluid and surface ARD scales and positive amplitudes.
4. Physics residual: a positive fitted Laan parameter A plus the Matérn 3/2 GP residual.

The custom structured kernel is a positive-semidefinite sum/product construction. Its analytic gradients were checked against finite differences. No extra synthetic impact labels were generated. Exact GP inference retains all 1,498 training impacts.

Optimizer diagnostics are retained in every fold file: 12/48 outer fits and 48/144 inner fits emitted warnings, including bounds or iteration limits. These diagnostics should be reviewed before a journal submission; they are not silently discarded.

## Issues corrected in this release

| Earlier issue | Implemented correction |
|---|---|
| Global GP warm start touched all surfaces | Cold initialization within every fit |
| Other-surface residual models could include the outer test surface | Rebuild inner residual models excluding that outer surface |
| Model family chosen from the same scores used for reporting | Add separate nested selection evaluation and label fixed-candidate rankings |
| Interpolated/capped conformal quantile | Explicit sorted-score rank; insufficient calibration returns infinity |
| Smooth reference looked ordinary to anomaly detector | Explicit surface-class warning in addition to support and anomaly checks |
| Re<70 described as GP extrapolation | GP support comes from data, independent of scaling-law regime |
| Unspecified model/scaler/units at deployment | Versioned JSON and binary numerical bundle with hashes |
| Unsupported user inputs and ambiguous geometry | Validated units, positivity, finite values, bounded phi and explicit custom descriptors |
| Inconsistent narrative tables | Report generated directly from saved predictions and evaluation JSON |
| Browser/backend numerical mismatch risk | Independent numerical export tests and JavaScript parity checks |

The paper’s Matérn 5/2, log-base-10 baseline, 0.0394 REF-H RMSE, beta range 0.2–1.0 and cubic-micrometre texture units do not describe the shipped baseline/data. Use the values and definitions in this report. The original paper and original archives are preserved, not silently overwritten.

## Verification and delivery

- Twelve Python tests passed: data integrity, physical identities, invalid inputs, interval ordering, smooth warnings, kernel gradients/PSD, complete split isolation, quantile edge behavior, exported-vs-sklearn predictions and metric recomputation.
- JavaScript/Python parity: 24 cases across 4 models, including smooth reference, high viscosity and extrapolation. Maximum beta difference 1.42e-12; maximum interval-bound difference 1.44e-12. Six invalid JavaScript inputs were rejected.
- Model arrays are exported as JSON and little-endian float64 lower-triangular Cholesky data; no pickle execution is needed for inference.
- The live website computes the real GP in a Web Worker. No training job runs when a visitor predicts. All four trained candidates are available for comparison.
- A local Python HTTP API and matching website are included. Static hosting does not provide a remote Python API endpoint.
- Full browser UI automation was unavailable for this static-site preview profile. JavaScript syntax, prediction logic, API behavior and static asset references were checked; visual/browser interaction QA remains a limitation.
- WebMCP registration is feature-detected. A supported WebMCP browser validation context was unavailable; ordinary UI operation does not depend on it.

## Next useful experiments

Collect genuinely new surfaces spanning track width, pitch and depth, with per-fluid advancing/receding angles and measurement uncertainty. Reserve a prospective test before further tuning. Then compare descriptor uncertainty, a measured-wettability GP and heteroscedastic observation noise. A larger architecture alone is not supported as an improvement by the current dataset.

## Primary methods references

- https://scikit-learn.org/stable/modules/gaussian_process.html
- https://arxiv.org/abs/2107.07511
- https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf
- https://doi.org/10.1103/PhysRevApplied.2.044018
