# Droplet spreading on unseen surfaces: project summary

Author: Parth Sharma, Thapar Institute of Engineering & Technology
Event: International Conference on Interfacial Phenomena in Droplets, IISc Bengaluru, 5–7 Oct 2026
Status on 24 Sept 2026: paper, A0 poster, pitch scripts and live browser model done.
Every number below comes from the v2.1.0 result files (`results/benchmark.json`, `models/release.json`, `results/sem/summary.json`) or `paper/numbers.json`, which is recomputed from them. This version replaces an earlier summary from 24 Sept that contained wrong figures (see the end).

## 1. Question and contribution
Can a model predict the maximum spreading ratio β_max of a droplet on a textured surface it has never seen?
- Test protocol: leave-one-surface-out (LOSO), one whole surface held out per fold. Models are chosen on this, not on random splits (each condition was repeated about five times, so random splits leak near-duplicates).
- Surface description: smooth-plateau fraction φ and texture volume per area, computed from laser-track widths read off SEM images (about 45 µm deep class, about 30 µm shallow class). The smooth plate is φ = 1, texture volume 0.
- Calibrated 90% intervals and reliability flags (measured-range checks and a Mahalanobis anomaly score).
- Experimental SEM branch: a 9,070-parameter CNN reads three magnifications (43×, 100×, 350×) and predicts φ and texture volume.

## 2. Data
Public dataset: Može et al., *Data in Brief* 61 (2025) 111697, University of Ljubljana. The experiments are theirs.

| Item | Value |
|---|---|
| Textured impacts | 1,498 (297 conditions, 12 laser-textured Al surfaces) |
| Smooth-plate impacts (REF-H) | 125 (25 conditions), never used to train or choose a model |
| Fluids | water, 20, 60, 78, 91 wt% glycerol |
| Viscosity | 0.94, 1.6, 9.1, 36.6, 151 mPa·s (160-fold range) |
| Drop diameter D₀ | 2.15–2.61 mm |
| Impact velocity V | 0.48–1.71 m/s |
| Re | 8.9–4,435 |
| We | 8.0–119 |
| β_max measured | 1.44–3.34 |

Surfaces: shallow (6 µm) S50–S800 and deep (25 µm) D50–D800, where the number is channel pitch in µm (50, 100, 200, 400, 600, 800).

| Surface | φ | Texture vol. (µm) | Surface | φ | Texture vol. (µm) |
|---|---|---|---|---|---|
| S50 | 0.16 | 5.04 | D50 | 0.01 | 24.75 |
| S100 | 0.49 | 3.06 | D100 | 0.30 | 17.44 |
| S200 | 0.72 | 1.67 | D200 | 0.60 | 9.98 |
| S400 | 0.86 | 0.87 | D400 | 0.79 | 5.31 |
| S600 | 0.90 | 0.59 | D600 | 0.86 | 3.61 |
| S800 | 0.93 | 0.44 | D800 | 0.89 | 2.73 |
| REF-H | 1.00 | 0 | | | |

## 3. Final model
Gaussian process, Matérn 3/2 kernel, on ln Re, ln We, ln D₀, φ and texture volume; target ln β_max. 90% interval = GP posterior sd × empirically calibrated q = 1.645.

## 4. Results

### 13 learners, same five inputs
| Model | LOSO RMSE | Random-split RMSE | Worst surface | REF-H RMSE |
|---|---|---|---|---|
| **Gaussian process** | **0.0400** | 0.0363 | 0.059 (D50) | 0.0591 |
| Random forest | 0.0462 | 0.0440 | 0.080 (D50) | 0.0628 |
| Extra trees | 0.0465 | 0.0437 | 0.096 (D50) | 0.0634 |
| FT-Transformer | 0.0505 | 0.0455 | 0.098 (D50) | 0.0671 |
| CatBoost | 0.0524 | 0.0446 | 0.114 (D50) | 0.0628 |
| Deep ensemble (5 MLP) | 0.0525 | 0.0450 | 0.101 (D50) | 0.0674 |
| XGBoost | 0.0563 | 0.0462 | 0.115 (D50) | 0.0593 |
| LightGBM | 0.0591 | 0.0509 | 0.136 (D50) | 0.0598 |
| Quadratic ridge | 0.0638 | 0.0541 | 0.135 (D50) | 0.0702 |
| MLP | 0.0701 | 0.0525 | 0.167 (D50) | 0.0820 |
| kNN | 0.0760 | 0.1322 | 0.178 (D50) | 0.0608 |
| SVR (RBF) | 0.1085 | 0.0454 | 0.305 (D50) | 0.0491 |
| Kernel ridge | 0.2568 | 0.0442 | 0.825 (D50) | 0.0475 |

GP is 29% better than XGBoost on unseen surfaces. D50 (φ = 0.01) is the hardest surface for every learner. Note SVR and kernel ridge do well on REF-H but fail on unseen textured surfaces.

### What did not help (Δ LOSO RMSE vs the GP, 95% paired surface-bootstrap CI, 4,000 draws, seed 23)
| Change | Δ RMSE | 95% CI |
|---|---|---|
| Matérn 5/2 kernel | −0.0002 | [−0.0009, 0.0005] |
| Laan backbone + residual GP | −0.0012 | [−0.0030, 0.0001] |
| SEM CNN descriptors + GP | −0.0016 | [−0.0038, 0.0005] |
| Fluid–surface interaction GP | +0.0012 | [−0.0043, 0.0076] |
| Nested model selection | +0.0015 | [−0.0040, 0.0078] |
| GP without surface descriptors | +0.0063 | [−0.0026, 0.0144] |
| XGBoost, same inputs | +0.0164 | [0.0057, 0.0278] (significantly worse) |

Only XGBoost is significantly different, and it is worse, so the plain GP is kept.

### Error per held-out surface (GP)
S50 0.050 · S100 0.034 · S200 0.040 · S400 0.036 · S600 0.035 · S800 0.031 · D50 0.059 · D100 0.035 · D200 0.029 · D400 0.041 · D600 0.040 · D800 0.039 · all 12: 0.040.
Without φ and texture volume, D50 rises from 0.059 to 0.091; the surface inputs hurt slightly on S50 and D200. With 12 surfaces this is a hint, not proof.

### Interval coverage (target 90%)
| Glycerol | Unseen textured | Smooth plate REF-H |
|---|---|---|
| 0 wt% (water) | 76% | 36% |
| 20 wt% | 85% | 24% |
| 60 wt% | 94% | 56% |
| 78 wt% | 97% | 96% |
| 91 wt% | 98% | 96% |
| **All** | **90.1%** | **61.6%** |

Nested CV: RMSE 0.0415, coverage 92.6% (per-fold 86–99%).

### Smooth plate REF-H
RMSE 0.059, mean error 1.9%, 62% coverage. Misses concentrate in thin fluids and at high We, where the model over-predicts. Likely cause: the smooth plate is hydrophobic, the textures superhydrophobic, and wettability is not an input.

### SEM CNN branch (experimental)
LOSO RMSE 0.0384 vs 0.0400 baseline, better on 8 of 12 surfaces, not significant (CI [−0.0038, 0.0005]). Reads REF-H as φ ≈ 0.71 because no training surface has φ above 0.93. REF-H RMSE 0.065, coverage 59%. Its intervals do not include CNN uncertainty.

### Physics
Re and We recomputed from the raw data match the dataset to 5×10⁻⁷. Laan scaling law: refitted constant A = 1.14; the law alone misses by about 11% on average.

## 5. Deliverables
| Item | Location |
|---|---|
| Paper (11 pages) | `paper/Droplet_Spreading_Paper_v2.docx` and `.pdf`; text in `paper/research_paper_v2.md` |
| A0 poster, submitted (portrait) | `poster/final_A0_portrait/Parth_Sharma_Droplet_Poster_A0_Portrait.pdf` |
| Earlier A0 landscape poster (28 Sept) | `poster/landscape_A0/` |
| Earlier A0 portrait poster + cover (24 Sept) | `poster/portrait_v1_with_cover/` |
| Pitch scripts | `notes/pitch_scripts.md` |
| Live website | https://iisc-droplet-parth1682.vercel.app (source `site/`) |
| Local demo app | `python -m droplet.serve --port 8000` in `Droplet_Intelligence_v2/` |

The browser model runs the trained baseline GP (same weights and Cholesky factor as the package) and matched the Python predictor on five test impacts to two decimals, e.g. water, D200, 2.5 mm, 1.5 m/s → 2.86 [2.81, 2.91].

## 6. Open items
- Cover image is 1132×1600 px, low for A0 print; supply a larger original if possible.
- TIET logo is 277×258 px; an official vector file would print better.
- ~~Paper author list~~: confirmed 1 Oct 2026. The .docx body, .docx metadata and PDF metadata all list Parth Sharma as sole author.

## 7. Corrections to the earlier 24 Sept summary
The first version of this summary (written the same day) had these errors: data cited as "Pode et al." (correct: Može et al.); kernel given as Matérn 5/2 (deployed model is Matérn 3/2); φ range given as 0.55–0.93 and texture volume in mm³ (correct: φ 0.01–1.0, texture volume 0.44–24.75 µm); diameter range 1.5–3.5 mm (correct: 2.15–2.61 mm); a 40 wt% fluid that does not exist; wrong viscosities for 20/60/78 wt%; invented per-surface RMSE, per-fluid coverage and model-comparison tables. All replaced above with values from the result files.

## 8. Extensions after submission (1 Oct 2026)
Source: `Droplet_Intelligence_v2/results/extensions/summary.json` and `REPORT.md`. Same protocol (LOSO, paired surface bootstrap 4,000 draws seed 23, REF-H never used for choices).

| Candidate | LOSO RMSE | 95% CI vs baseline | REF-H RMSE | REF-H coverage |
|---|---|---|---|---|
| baseline | 0.0400 | [+0.0000, +0.0000] | 0.0591 | 61.6% |
| image_phi | 0.0398 | [-0.0057, +0.0047] | 0.0595 | 66.4% |
| image_only | 0.0397 | [-0.0068, +0.0045] | 0.0636 | 61.6% |
| lee_beta0 | 0.0398 | [-0.0013, +0.0009] | 0.2279 | 0.8% |
| angle_input | 0.0480 | [+0.0004, +0.0171] | 0.3771 | 100.0% |

- **Image reader (accepted as an input path):** φ from one 43× SEM image; REF-H reads 1.000. Equivalent accuracy, not better; baseline stays default. Valid near 43× only. Tools: `droplet.predict_image`, website section "Drop in an SEM image".
- **Wettability (rejected on this data):** training angles too uniform; Lee correction over-predicts REF-H.
- **Video pipeline:** `droplet.video` (single video or `--watch` folder), website section "Drop in an impact video". Validated on synthetic videos only; raw recordings to be requested from Može et al.
- **Paper:** Sections 4.8–4.10 added; .docx/.pdf rebuilt (11 pages). The submitted poster is unchanged.
