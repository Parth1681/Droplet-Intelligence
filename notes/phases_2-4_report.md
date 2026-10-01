# Phases 2–4 report

Sep 23, 2026

**Answer.** The final model is a Gaussian process on log Re, log We, log D₀ and two SEM-derived surface descriptors (smooth-plateau fraction φ, texture volume). Its error on an unseen textured surface is 0.040, and 0.059 on the smooth REF-H surface (1.9% mean error).

- **Phase 2:** GPR beats XGBoost on every input set. Dimensionless inputs alone do not beat raw ones.
- **Phase 3:** only D₀, φ and texture volume earn their place. Re/We-derived groups add nothing, and inputs that push REF-H outside the training range break it (depth/D₀: REF-H error 0.61; contact angle: 0.39).
- **Phase 4:** a GPR on raw inputs scores 0.003 better on unseen textured surfaces, but its smooth-surface answer depends on a made-up spacing/depth (REF-H error 0.058–0.375). φ removes that choice, so it stays. Intervals are calibrated on textured surfaces (90% coverage) but overconfident on REF-H (62%).

## Phase 2 — Baselines

The learner matters more than the input set: GPR beats XGBoost on every input set, and replacing raw inputs with Re/We/Oh makes both worse.

**What was done.** The brief's three input sets (raw, the paper's dimensionless set, hybrid) × two learners, same protocol: grouped 5-fold CV (ID), leave-one-surface-out (LOSO, 12 folds) and REF-H predicted once. These sets have no descriptor for a smooth plate, so REF-H is typed in as spacing = depth = 0. GPR is fitted row-level exactly like the final model, with hyperparameters re-optimised in every fold.

| Learner | Inputs | ID RMSE | LOSO RMSE | Worst surface | REF-H RMSE | REF-H bias | Bias, top-We third |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPR | raw 7 (D₀, V, ρ, σ, μ, spacing, depth) | 0.037 | **0.037** | 0.063 D50 | 0.058 | +0.005 | +0.049 |
| GPR | hybrid (raw + Re, We, Oh) | 0.037 | 0.037 | 0.063 D50 | 0.058 | +0.005 | +0.049 |
| GPR | Re, We, Oh + spacing, depth (paper set) | 0.043 | 0.045 | 0.091 D50 | 0.069 | +0.031 | +0.079 |
| XGBoost | hybrid | 0.044 | 0.049 | 0.084 D50 | 0.064 | +0.009 | +0.057 |
| XGBoost | raw 7 | 0.046 | 0.051 | 0.098 D50 | 0.065 | +0.014 | +0.052 |
| XGBoost | Re, We, Oh + spacing, depth (paper set) | 0.051 | 0.054 | 0.111 D50 | 0.069 | +0.027 | +0.080 |
| Scaling law | β = a·We^b·Re^c, fitted | 0.134 | 0.131 | 0.157 | 0.106 | +0.029 | — |
| Scaling law | Laan et al. 2014, A refitted | 0.231 | 0.231 | 0.257 | 0.219 | −0.049 | — |

**Evidence.** GPR is 0.009–0.014 lower than XGBoost on LOSO for every input set. The paper's dimensionless set is the worst ML row for both learners and has the largest REF-H bias. Adding Re, We, Oh to raw inputs changes nothing for GPR. Scaling laws alone are 3–6× worse than any ML model. Every model over-predicts REF-H in the top We third (+0.05 to +0.08).

**Weaknesses.** The brief's premise that dimensionless inputs improve accuracy does not hold here: Re, We and Oh are smooth functions of the raw inputs, and only 5 fluid families exist. The best LOSO row (GPR on raw inputs, 0.037) cannot safely be used on a smooth surface: its REF-H error is 0.058 only because REF-H was typed as 0/0. Other equally arbitrary entries give 0.061, 0.059, 0.077 and 0.375. D50 is the worst surface for every model.

**Next.** Keep GPR. Replace spacing/depth with a descriptor that is defined for a smooth plate (Phase 3).

## Phase 3 — Physics features

Only three physics features earn their place: droplet size D₀, the smooth-plateau fraction φ and texture volume. Features that are exact functions of Re and We add nothing, and any feature that puts REF-H far outside the training range breaks the smooth-surface prediction.

**What was done.** Starting from a base of log Re, log We, φ and V\_tex, each candidate was added on its own, and φ and V\_tex were each removed from the final set. The learner is the final GPR, row-level, with hyperparameters re-optimised in every fold. Scores: LOSO RMSE (selection) and REF-H RMSE (never tuned on). Final set = base + log D₀: LOSO 0.040, REF-H 0.059.

| Feature | Equation | Physical meaning | Expected effect | LOSO | REF-H | Useful? |
| --- | --- | --- | --- | --- | --- | --- |
| base | log Re, log We, φ, V\_tex | inertia/viscosity, inertia/capillarity, surface | — | 0.048 | 0.055 | reference |
| + log D₀ | D₀ in mm | drop size beyond Re, We (gravity, texture scale) | small | **0.040** | 0.059 | **Yes**, best single addition (−16%) |
| + Bo | ρgD₀²/σ | gravity vs capillarity | small | 0.043 | 0.060 | Partly, a weaker proxy for D₀ |
| + log Oh | √We / Re | viscous vs inertial-capillary | regime marker | 0.048 | 0.055 | No, exact function of Re, We |
| + log P | We·Re^−2/5 | Laan capillary vs viscous regime | regime marker | 0.048 | 0.055 | No, linear in log Re, log We |
| + log β\_Laan | Padé law, A = 1.24 | physics prior for βmax | strong | 0.048 | 0.055 | No, fits this data poorly (R² 0.71) |
| + log μ | viscosity | fluid identity | none beyond Re | 0.048 | 0.055 | No |
| + spacing/D₀ | s / D₀ | texture scale vs drop | small | 0.048 | 0.063 | No, REF-H worse |
| + depth/D₀ | h / D₀ | groove depth vs drop | small | 0.050 | **0.609** | **Harmful**: REF-H at 0 is far outside the training range |
| + cos θ | measured θ\_adv; REF-H 114.5° assumed | wettability | large for REF-H | 0.068 | **0.392** | **Harmful**: training θ spans only 162–167° |
| final − φ | — | — | — | 0.046 | 0.066 | φ is useful (+0.006 on both) |
| final − V\_tex | — | — | — | 0.041 | 0.064 | V\_tex is useful for REF-H (+0.005) |

**Evidence.** Adding D₀ cuts the unseen-surface error by 16% and the D50 error from 0.104 to 0.059. Oh, P and the Laan prior change the fourth decimal at most. φ and V\_tex each lower REF-H error by 0.005–0.006.

**Weaknesses.** Contact angle is the physically right variable for REF-H, but the training surfaces barely vary in it (162–167°), so no model can learn its effect. Feeding REF-H's 114° in is pure extrapolation. The φ formula rests on SEM-measured track widths (±10 µm); REF-H results move by at most 0.005 across that range.

**Next.** Keep log Re, log We, log D₀, φ, V\_tex. Phase 4 tests whether physics structure (residual backbones, wettability corrections, monotonicity) improves generalisation beyond feature choice.

## Phase 4 — Attacking cross-surface generalisation

The GPR on SEM descriptors is the most robust option. Physics backbones help only slightly, and every attempt to inject wettability made REF-H 4–6× worse. Uncertainty is well calibrated on unseen textured surfaces (90% coverage) but overconfident on REF-H (62%).

**What was done.** Eight strategies from the brief, same protocol. The GPR variants share the final inputs unless stated. For the wettability tests REF-H's unknown contact angle was swept over 110–120° (a similar coating on smooth Al measures \~114.5°).

| Strategy | LOSO RMSE | Worst surface | REF-H RMSE | REF-H bias | Bias, top-We third |
| --- | --- | --- | --- | --- | --- |
| **GPR + SEM descriptors φ, V\_tex (final)** | **0.040** | 0.059 D50 | **0.059** | +0.004 | +0.041 |
| GPR on Laan-law residual (A refit) | 0.039 | 0.051 D50 | 0.059 | +0.004 | +0.042 |
| GPR on power-law residual | 0.041 | 0.066 D50 | 0.059 | +0.004 | +0.042 |
| GPR, raw spacing/depth (REF-H typed as 0/0) | 0.041 | 0.078 D50 | 0.059 | +0.008 | +0.050 |
| GPR, surface-blind (no surface inputs) | 0.046 | 0.091 D50 | 0.069 | +0.014 | +0.066 |
| XGBoost + SEM descriptors | 0.056 | 0.115 D50 | 0.059 | +0.010 | +0.049 |
| XGBoost, monotone in Re and We | 0.053 | 0.098 D50 | 0.066 | +0.010 | +0.056 |
| GPR + cos θ input (REF-H θ = 110 / 114.5 / 120°) | 0.047 | 0.098 D50 | 0.38 / 0.38 / 0.37 | −0.04 to −0.07 | −0.26 to −0.28 |
| Lee β₀ backbone + GPR residual (same sweep) | 0.039 | 0.050 D50 | 0.29 / 0.26 / 0.23 | +0.20 to +0.26 | +0.22 to +0.27 |

Follow-up: raw fluid/impact inputs (D₀, V, ρ, σ, μ) plus φ and V\_tex score LOSO 0.041, worst surface 0.069 (D50), REF-H 0.057 (bias +0.003). That is a tie with the final model, so the simpler dimensionless-plus-φ set stays.

**Uncertainty (final GPR, 90% predictive intervals):** coverage 90.1% on unseen textured surfaces, but only 61.6% on REF-H (mean width 0.082). Split-conformal recalibration on the LOSO residuals (quantile 1.63 ≈ the Gaussian 1.64) leaves REF-H coverage unchanged. The model does not know that REF-H is different.

**Evidence and reading**

1. **Surface information matters.** Dropping it raises LOSO error by 15% and REF-H error by 17%.
2. **Encoding matters more than accuracy suggests.** Raw spacing/depth scores like φ when REF-H is typed as 0/0, but other equally arbitrary choices move the raw-input GPR's REF-H error from 0.058 to 0.375 (tested: 0/0, 50/6, 800/6, 800/25, 5000/0 µm). φ has no free choice.
3. **Physics residuals give a small, consistent gain on the hardest surface.** Laan residual: worst surface 0.059 → 0.051, overall LOSO −0.001, which is within fold noise. REF-H unchanged.
4. **Static contact angle is the wrong wettability variable here.** Lee's β₀ correction predicts REF-H spreading 0.2–0.26 too high; as an ML input it extrapolates wildly. Smooth REF-H does not spread more at low We, which suggests the dynamic angle at maximum spread is far higher than 114° (HYPOTHESIS).
5. **Monotonic constraints don't help XGBoost;** the GPR is already smooth.

**Not run, with reasons.** Domain-adversarial training: 12 source domains that differ only slightly in βmax, so there is little domain signal to remove and a high risk of erasing useful surface information. Learned surface embeddings: 12 surfaces are too few to learn an embedding that transfers to a 13th. Feature disentanglement: the GPR's per-input length-scales already separate fluid, impact and surface effects.

## Weaknesses and next steps

The remaining gap is wettability, not geometry: the data cannot teach any model what a contact-angle change does, and the uncertainty does not warn about it.

| Weakness | Evidence | Next step |
| --- | --- | --- |
| REF-H over-predicted at high We | +0.041 bias in the top We third, in every model | Test the air-cushion slip hypothesis; add a dynamic-angle or slip descriptor |
| Intervals overconfident on REF-H | 62% coverage for a nominal 90% | Widen intervals when OOD is flagged, or calibrate on a held-out wettability class if one becomes available |
| No usable wettability input | training θ only spans 162–167°; REF-H error 0.23–0.39 when used | Needs dynamic contact angles at maximum spread, or more surface chemistries |
| D50 is always the worst surface | worst-surface RMSE 0.051–0.115 across methods | Densest texture: tracks merge, so φ ≈ 0 misses its roughness; measure roughness directly |
| Small n of surfaces | 12 textured + 1 smooth | Every cross-surface claim rests on 13 surfaces; report CIs (REF-H 95% CI 0.044–0.073) |
| Laan residual slightly better | LOSO 0.039 vs 0.040, worst surface 0.051 vs 0.059 | Adopt after the conference if it holds on repeat seeds |

Phases 5–6 (uncertainty and explainability) are partly covered above: coverage is measured, and the final GPR's length-scales (shorter = more influential) rank log We (8.5) and log Re (9.4) first, then log D₀ (12.7), with φ (25) and V\_tex (24) weakest. Surface effects on βmax are real but small. Code: `src/phases.py`, `src/phase_extra.py`; raw results in `results/phase2.json`, `phase3.json`, `phase4.json`.
