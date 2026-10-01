# Pitch scripts: Physics-informed ML for cross-surface droplet impact

Interfacial Phenomena in Droplets 2026, IISc Bengaluru, 5–7 Oct. Final model: Gaussian process on log Re, log We, log D₀, φ, V_tex.
Every number below is from the final results (results/final/*.json) and matches the poster.

## 30 seconds (walk-up hook)
Droplet-impact models are usually tested on the same surfaces they were trained on. I asked a harder question: can a model predict how far a droplet spreads on a surface it has never seen?
I describe each laser-textured surface with one number read from its SEM image, the smooth-plateau fraction φ, and feed it to a Gaussian process with the Reynolds and Weber numbers.
On an unseen textured surface it predicts β_max to within 0.040, 29% better than XGBoost. On a smooth plate it never trained on, it is off by 1.9% on average. And it tells you when not to trust it.

## 1 minute (someone stops and asks "what is this?")
The data: about 1,500 impacts on 12 laser-textured aluminium surfaces, water to 91% glycerol, plus 125 impacts on a smooth plate that I kept aside as a blind test.
Most papers split rows at random, so every surface is in training. I hold out one whole surface at a time. That is the test I pick models on.
The key idea is the surface description. A smooth plate has no channel spacing or depth, so encoding it with made-up values moved its error by up to 0.022. Instead I measure track widths from SEM and compute φ, the fraction of smooth plateau. The smooth plate is simply φ = 1.
I compared 13 models. A Gaussian process wins: 0.040 on unseen surfaces and 1.9% error on the smooth plate. Putting a scaling law underneath it as a physics backbone did not give a significant gain, so I kept it simple.
Its 90% intervals hold 90% of the time on unseen textured surfaces, but only 62% on the smooth plate. That is the honest limit: wettability changes and no input captures it.

## 3 minutes (full walk-through, point at the poster)
1. **Question (panel 1).** Surface design needs predictions *before* a surface is made. So the test is a surface the model has never seen, not a random split.
2. **Data and physics (panels 1–2).** 1,498 textured impacts and 125 smooth impacts, We 8–119, Re 9–4435. I recompute Re and We, and they match the dataset to 5×10⁻⁷. Oh is a function of Re and We, so the physics set is really two groups. Classic scaling laws fit poorly here (Laan R² = 0.71) because they were derived on wettable surfaces.
3. **Reading the surface (panel 3).** SEM shows rough laser tracks and smooth plateaus that look like the smooth plate. I segment the images, get track widths of about 45 µm (deep) and 30 µm (shallow), and compute φ and V_tex for every surface. That includes the smooth plate: φ = 1, V_tex = 0.
4. **Evaluation (panel 4).** Each condition was repeated five times, so random splits leak near-identical twins. I use grouped CV, leave-one-surface-out (the selection criterion) and one blind prediction of the smooth plate that is never used for tuning.
5. **Models (panel 5).** 13 learners on the same inputs. The GP is best on unseen surfaces at 0.040, against 0.056 for XGBoost. Every model's worst surface is D50, the densest texture. I also tried physics-residual models: a power law under XGBoost and the Laan law under the GP. Neither was a significant gain once I resample whole surfaces, so the plain GP stays.
6. **Why the descriptor matters (panel 6).** With invented encodings, the smooth-plate error moved by up to 0.022. With φ, changing the measured track width by ±10 µm moves it by at most 0.005.
7. **Trust (panel 7).** On unseen textured surfaces the 90% intervals hold 90% of the time (83–97% per surface). The OOD detector flags D50 and S50, the hardest surfaces. On the smooth plate the intervals hold only 62% and nothing flags it. The likely cause is wetting chemistry (about 114° versus 160–166°), which no input describes.
8. **Blind test (panel 8).** Smooth plate RMSE 0.059, mean error 1.9%. The error sits at high We, where textured surfaces spread further. My hypothesis is less friction over the trapped air layer.
9. **SEM to prediction (panels 9–10).** A small CNN reads φ from SEM patches of surfaces it never saw, which costs 0.043 vs 0.039 on unseen surfaces. Everything runs locally as a dashboard and a REST API.
10. **Take-away.** Judge droplet models on unseen surfaces, describe surfaces with quantities every surface has, and report where the model stops being trustworthy.

## Live demo (≈60 s, laptop next to the poster)
1. Open the dashboard at **Predict**. Pick water, D200, 2.5 mm, 1.5 m/s: β_max ≈ 2.86, 90% band 2.81–2.91.
2. Drag the velocity slider through the shaded measured range and point at the band widening once it leaves the range.
3. Switch the surface to **REF-H**. The smooth-surface warning appears and tells you to use the wide band.
4. Open **Uncertainty & OOD**: the parity plot for REF-H and the coverage bars per surface.
5. If asked about deployment, open the **API** page and show the curl call.

## Likely questions (short, honest answers)
- **Why not deep learning?** I tested an FT-Transformer, a deep ensemble and an MLP on the same inputs. With 12 surfaces, all generalized worse than the GP on unseen surfaces.
- **Is this physics-informed?** Yes. The inputs are the dimensionless groups plus physically defined surface descriptors, and the target is log β_max. A scaling-law backbone was tested explicitly and gave no significant gain.
- **Why only 62% coverage on the smooth plate?** The intervals are calibrated on textured, superhydrophobic surfaces. The smooth plate differs in wettability, which no input describes. That is why the tool shows a wide band and a warning for smooth surfaces.
- **Would it work on other materials?** Only within the texture family it was trained on. The CNN, for example, reads the smooth plate as φ = 0.69, not 1.0. The next step is a wettability input.
- **How much does the surface matter?** Removing surface inputs nearly doubles in-distribution error (0.086 vs 0.051). But Re and We carry most of the signal: mean |SHAP| of about 0.12–0.13 each vs about 0.004 for φ and V_tex.
