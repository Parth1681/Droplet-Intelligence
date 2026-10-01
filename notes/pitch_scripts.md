# Pitch scripts: Physics-informed ML for cross-surface droplet impact

International Conference on Interfacial Phenomena in Droplets, IISc Bengaluru, 5–7 Oct 2026. Author: Parth Sharma (Thapar Institute).
Final model: Gaussian process on ln Re, ln We, ln D₀, φ, V_tex.
Numbers match the submitted A0 portrait poster (`poster/final_A0_portrait/`) and the paper (`paper/research_paper_v2.md`). Re-checked against the result files on 24 Sept 2026; layout references updated to the portrait poster on 1 Oct 2026.
Data: public dataset of Može et al., *Data in Brief* 61 (2025) 111697. The experiments are theirs; the surface-held-out evaluation and the models are ours.

## 30 seconds (walk-up hook)
Droplet-impact models are usually tested on the same surfaces they were trained on. We asked a harder question: can a model predict how far a droplet spreads on a surface it has never seen?
We describe each laser-textured surface by its smooth-plateau fraction φ, computed from track widths read off SEM images, and feed it to a Gaussian process with the Reynolds and Weber numbers.
On an unseen textured surface it predicts β_max to within 0.040, 29% better than XGBoost. On a smooth plate it never trained on, it is off by 1.9% on average. And it tells you when not to trust it.

## 1 minute (someone stops and asks "what is this?")
The data is a public set from Ljubljana: about 1,500 impacts on 12 laser-textured aluminium surfaces, water to 91% glycerol, plus 125 impacts on a smooth plate. We never used the smooth plate to train or choose a model.
Most papers, including the original ML study on this data, split rows at random, so every surface is in training. We hold out one whole surface at a time. That is the test we pick models on.
The key idea is the surface description. A smooth plate has no channel spacing or depth, so encoding it with made-up values moved its error by up to 0.022. Instead we estimate track widths from SEM (about 45 µm deep, 30 µm shallow) and compute φ, the fraction of smooth plateau. The smooth plate is simply φ = 1.
We compared 13 models. A Gaussian process wins: 0.040 on unseen surfaces and 1.9% error on the smooth plate. Putting a scaling law underneath it as a physics backbone did not give a significant gain, so we kept it simple.
Its 90% intervals hold 90% of the time on unseen textured surfaces, but only 62% on the smooth plate. The misses cluster in the thin fluids: water and 20% glycerol. That is the honest limit: wettability changes and no input captures it.

## 3 minutes (full walk-through, point at the poster)
Panel numbers refer to the submitted A0 portrait poster. Three columns, read top to bottom, left to right: column 1 is Introduction and Methods (panels 1–3), column 2 is inputs, evaluation and the first results (panels 4–7), column 3 is intervals, per-surface error and the smooth plate (panels 8–10, then Future work). The Conclusions band runs along the bottom.
1. **Background and objective (panel 1, top left).** Surface design needs predictions *before* a surface is made. So the test is a surface the model has never seen, not a random split. Point at the stats grid: 1,498 textured impacts and 125 smooth-plate impacts from Može et al. (2025), We 8–119, Re 9–4,435, water to 91% glycerol.
2. **Surface descriptors from SEM (panel 2).** Point at the three SEM photos: rough laser tracks and smooth plateaus that look like the smooth plate. From the images we get track widths of about 45 µm (deep) and 30 µm (shallow). From those we compute φ and V_tex for every surface; the smooth plate is simply φ = 1, V_tex = 0. With invented spacing/depth codes, its error moved by up to 0.022; with φ, ±10 µm in track width moves it by at most 0.005.
3. **Reading descriptors directly from images (panel 3, bottom of column 1).** A small CNN reads three SEM views of a surface it never saw and estimates φ and V_tex. The GP on those gets 0.038 against 0.040, better on 8 of 12 surfaces but not significant. It reads the smooth plate as φ ≈ 0.71, because no training surface is smoother than 0.93.
4. **Dimensionless inputs (panel 4, top of column 2).** We recompute Re and We, and they match the dataset to 5×10⁻⁷. Oh is a function of Re and We, so it adds nothing. Laan's scaling law with its published constant gives R² = 0.71 here, because it was derived on wettable surfaces.
5. **Evaluation (panel 5).** Point at the three-step flow. Each condition was repeated five times, so random splits leak near-identical twins. We use grouped CV and leave-one-surface-out (step 2, highlighted, the selection criterion). The smooth plate is held out and never used to train or choose a model. A change is accepted only if its 95% interval, from resampling whole surfaces, excludes zero.
6. **Comparison of 13 models (panel 6, Results).** 13 learners on the same inputs. The GP is best on unseen surfaces at 0.040 (1.1% mean error), against 0.056 for XGBoost.
7. **Variants tested (panel 7, bottom of column 2).** A Laan-law backbone, other kernels, nested selection and the CNN descriptors each change the error by under 0.002. Only XGBoost is significantly different, and it is worse. So the plain GP stays.
8. **Prediction intervals (panel 8, top of column 3).** On unseen surfaces 90.1% of impacts fall inside the 90% interval, but not evenly: 76% for water, 98% for 91% glycerol. A thin fluid is more sensitive to the surface, so one noise level is too narrow for it. The OOD check flags D50 and S50, the hardest surfaces.
9. **Error for each held out surface (panel 9).** D50, at the edge of the φ range, is the hardest surface for every learner. The texture inputs cut its error from 0.091 to 0.059, but they hurt on S50 and D200. With 12 surfaces, their benefit is a hint, not proof.
10. **Smooth plate REF-H (panel 10).** RMSE 0.059, mean error 1.9%, but only 62% inside the interval. Point at the warning box: the misses sit at high We and in water, 20% and 60% glycerol; at high We the model over-predicts, because textured surfaces spread further. The smooth plate is hydrophobic and the textures are superhydrophobic, so wettability is the missing input. Our hypothesis: less friction over the trapped air layer on the textures.
11. **Future work and Conclusions (bottom of column 3, bottom band).** Next: measured contact angles as an input, a noise level that depends on Oh, and new surfaces with a test set fixed before tuning. Close on the Conclusions band: judge droplet models on unseen surfaces, describe surfaces with quantities every surface has (φ = 1 for the smooth plate), and say where the model stops being trustworthy.


## New since the poster (30 seconds, say after panel 10 or when asked "what next?")
Since submitting I tested two ways to reach surfaces outside the dataset.
First, reading the surface straight from one 43× SEM image: the rough laser tracks and smooth plateaus are separated by local roughness, which gives φ without knowing spacing or depth. With that image φ the model is just as accurate on unseen surfaces, 0.0397 against 0.0400, and it reads the smooth plate correctly as φ = 1.00, where the CNN said 0.71.
Second, adding contact angle. It does not work on this data: the Lee correction over-predicts the smooth plate by about 0.22, and as an input the angle makes the model worse, because every training surface sits between about 160 and 167°. So wettability needs new surfaces with intermediate angles, which is the clearest next experiment.
I also automated the measurement: a pipeline reads D₀, V and β_max from an impact video and checks them against the prediction, with no manual steps.

## Live demo (≈90 s, laptop next to the poster)
Open https://iisc-droplet.vercel.app (or the Claude artifact). Everything runs in the browser.
1. **Predict.** Water, D200, 2.5 mm, 1.5 m/s: β_max ≈ 2.86, 90% band 2.81 to 2.91. Push V past 1.71 m/s: status turns to extrapolation with reasons.
2. **Compare all five models.** They agree to within about 0.01.
3. **Your image.** Click "Smooth plate": φ reads 1.000 from the image and the tool warns that wettability is not modelled. Click "D200 textured": φ 0.48, β ≈ 2.84.
4. **Video.** Click "Synthetic impact": it finds the drop and surface by itself and measures D₀ 2.50 mm, V 1.20 m/s, β_max 2.58 (true 2.6), then compares with the prediction.
5. If asked about evidence: **Evaluate** (per-surface error and coverage) and **SEM** (the CNN result).
Backup: local app, `python -m droplet.serve --port 8000` in Droplet_Intelligence_v2.

## Likely questions (short, honest answers)
- **Did you do the experiments?** No. The impacts are a public dataset from Može et al., University of Ljubljana (*Data in Brief*, 2025). My contribution is testing on unseen surfaces, the surface descriptors, the models and the uncertainty analysis.
- **Hasn't someone already used a GP on this data?** Yes. Jereb et al. (*Biomimetics*, 2025) chose a GP with a random 80/20 split. I test on a surface the model has never seen, which is the harder and more useful question.
- **Why not deep learning?** I tested an FT-Transformer, a deep ensemble and an MLP on the same inputs. With 12 surfaces, all generalized worse than the GP on unseen surfaces.
- **Is this physics-informed?** Yes. The inputs are the dimensionless groups plus physically defined surface descriptors, and the target is log β_max. A scaling-law backbone was tested explicitly and gave no significant gain. The Laan law alone misses by about 11% on average.
- **Is the smooth plate really a blind test?** It was never used to train or choose a model, but I did look at its results during development. So I call it a held-out stress test, not a fully blind one.
- **Why only 62% coverage on the smooth plate?** The intervals are calibrated on textured, superhydrophobic surfaces. The smooth plate differs in wettability, which no input describes. The shortfall is mostly in water and the 20% and 60% mixtures; the viscous fluids stay at 96%. That is why the tool shows a warning for smooth surfaces.
- **Would it work on other materials?** Only within the texture family it was trained on. The CNN, for example, reads the smooth plate as φ ≈ 0.71, not 1.0. The next step is a wettability input.
- **How much does the surface matter?** Re and We carry most of the signal: mean |SHAP| of about 0.12–0.13 each against about 0.004 for φ and V_tex. Removing the surface inputs raises the unseen-surface error from 0.040 to 0.046 and the smooth-plate error from 0.059 to 0.069. It helps most on D50, the most extreme texture. With 12 surfaces that is a strong hint, not proof.
- **What would you do next?** Measure advancing and receding angles for every fluid–surface pair, let the noise level depend on viscosity, and add new surfaces with a test set held back before any tuning.
- **Can it handle a new surface it has never seen?** Within the laser-textured family, yes: give it one 43× SEM image and it reads φ itself, with the same error as the geometry model (0.0397). Other materials or wettability are extrapolation, and the tool says so.
- **Why not add contact angle?** I tried. All training surfaces are superhydrophobic (about 160 to 167°), so the model cannot learn the effect, and the Lee correction over-predicts the smooth plate. It needs surfaces with intermediate angles.
- **Can it run on its own?** Yes. A watch mode measures each new impact video automatically and logs measured against predicted β_max. So far it is validated on synthetic videos with known answers; I am requesting raw recordings from the dataset authors to validate it on real footage.
