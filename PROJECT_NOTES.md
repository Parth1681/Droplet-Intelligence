# Droplet Intelligence: project notes

Research project by Parth Sharma: predict the maximum spreading ratio β_max = D_max/D_0 of a droplet impacting a laser-textured aluminium surface **that the model has never seen**. Presented as a poster at the International Conference on Interfacial Phenomena in Droplets, IISc Bengaluru, 5–7 Oct 2026 (sub-topic: Impact of Droplets onto Surfaces).

Status on 1 Oct 2026: the paper, the A0 portrait poster (the submitted version), the pitch scripts, the live website and the local demo app are all done. Work period: 23 Sept to 1 Oct 2026.

## Ground rules

1. **Numbers come only from the result files.** That means `Droplet_Intelligence_v2/results/*.json`, `Droplet_Intelligence_v2/models/release.json`, `Droplet_Intelligence_v2/results/sem/summary.json`, or `paper/numbers.json`, which `paper/paper_numbers.py` rebuilds from them. Never type a figure from memory. Earlier drafts contained invented numbers; the list of what was wrong is in `notes/research_summary.md` §7.
2. **The data are not Parth's experiments.** They are the public dataset of Može et al., *Data in Brief* 61 (2025) 111697 (Mendeley Data doi:10.17632/wsh8rxwd38.1), from the University of Ljubljana. Parth's contribution is the surface-held-out evaluation, the surface descriptors, the models and the uncertainty analysis.
3. **REF-H (the smooth plate) is never used to train or choose a model.** It was looked at during development, so call it a *held-out stress test*, not a blind test.
4. **Models are chosen on leave-one-surface-out (LOSO) RMSE.** A change is accepted only if its 95% paired bootstrap interval excludes zero. The bootstrap resamples whole surfaces (4,000 draws, seed 23). Otherwise the simpler model stays.
5. **Sole author:** Parth Sharma, Thapar Institute of Engineering and Technology, Patiala, Punjab, India.
6. **Poster rules** (conference spec plus Parth's requests):
   - A0 portrait.
   - Conference name prominent at the top, conference logo in the top-right corner, presenter name underlined.
   - Plain, non-"AI" wording.
   - **No hyphens or dashes in visible poster text.** The exceptions are the sample code REF-H and minus signs in maths. For example, the title reads "Physics Informed Machine Learning for Cross Surface Droplet Impact Prediction", and number ranges are written "8 to 119".

## Folder map

| Path | What it is |
|---|---|
| `Droplet_Intelligence_v2/` | **Main code package, v2.1.0.** It contains the data (`data/raw` = the 5 original CSVs, `data/experiments.csv`, `data/sem` = SEM thumbnails and CNN training arrays), the trained models (`models/`, float64 Cholesky bundles), every fold-level result (`results/`), the browser UI (`web/`), the tests and the docs. Its own `README.md` documents the API and how to retrain. |
| `paper/` | The paper: `Droplet_Spreading_Paper_v2.docx` / `.pdf`, its text `research_paper_v2.md`, `numbers.json` (the single source for paper and poster numbers), `paper_numbers.py`, `ablation.py`, `figs.py`, `build.js` (Node `docx`, writes the .docx) and `fig1`–`fig4` PNGs. |
| `poster/final_A0_portrait/` | **The submitted poster:** `Parth_Sharma_Droplet_Poster_A0_Portrait.pdf`. Built by `figs.py` → `build6.py` → `render.py`. |
| `poster/landscape_A0/` | The earlier A0 landscape version (28 Sept), built by `figs3.py`, `build4.py` and `render3.py`. |
| `poster/portrait_v1_with_cover/` | The first portrait poster plus a cover page (24 Sept). |
| `live_model/` | Early single-file browser predictor (`index.html`, `data.js`, `L.txt`). It runs the same baseline GP and matched Python on five test impacts to two decimals. Superseded by the website in `site/`. |
| `notes/` | `research_summary.md` (all key tables), `pitch_scripts.md` (30 s / 1 min / 3 min scripts, demo steps, likely questions), `phase1_audit_and_plan.md` and `phases_2-4_report.md` (the planning document). |
| `archive/v1_droplet/` | The earlier v1 workspace. `src/` contains the 13-model screen (`models_extra.py`, `ft_transformer.py`), the phase 2–4 experiments (`phases.py`, `phase_extra.py`), the CNN and OOD code. `results/` holds their outputs (including `zoo_table.csv`). `droplet_app/` is the Streamlit and FastAPI dashboard, the backup demo. It also contains `webdash/`, `live/` and the v1 `poster/`. |

## Final model and headline results

- **Model:** Gaussian process, Matérn 3/2 kernel with one length scale per input, plus white noise.
  - Inputs: ln Re, ln We, ln D_0 (mm), φ, V_tex. Target: ln β_max. The prediction is the lognormal median.
  - 90% interval: exp(μ ± 1.645 σ).
  - φ = (max(s − w, 0)/s)² and V_tex = (1 − φ)·h. Track width w is 45 µm for deep channels and 30 µm for shallow ones. REF-H has φ = 1 and V_tex = 0.
- **Unseen textured surfaces (LOSO):** RMSE 0.0400 (MAPE 1.1%).
  - XGBoost on the same inputs: 0.0563, with a 95% CI of the difference of [0.0057, 0.0278].
  - Grouped CV: 0.0363. Worst surface: D50 at 0.059.
- **Smooth plate REF-H:** RMSE 0.0591 (MAPE 1.9%). Only 61.6% of impacts fall inside the 90% interval; the misses are in water and the 20% and 60% glycerol mixtures.
- **Intervals on unseen surfaces:** 90.1% coverage overall, but 76% for water and 98% for 91 wt% glycerol.
- **Variants that gave no significant gain:** Matérn 5/2, Laan backbone + residual GP, SEM CNN descriptors, the fluid–surface interaction kernel and nested selection. The plain GP stays.
- **Checks:** 18 unit tests pass, and the JS↔Python parity check agrees to ≤3.1×10⁻¹².

Full tables are in `notes/research_summary.md` and `paper/research_paper_v2.md`.

## How to run (tested on 1 Oct 2026)

**Demo app, which Parth uses at the poster.** Python 3.10+:
```
cd Droplet_Intelligence_v2
python -m pip install -r requirements.txt
python -m droplet.serve --port 8000        # or double-click run_demo.bat on Windows
```
Open http://127.0.0.1:8000. As a check, POST to `/api/predict` with `{"D_mm":2.5,"V":1.5,"rho":997.2375,"mu":0.000943923,"sigma":0.07246,"surface":"D200"}`. It should return β_max 2.859, interval [2.807, 2.913], status `within_measured_support`.

**Website** (live at https://iisc-droplet-parth1682.vercel.app; root `vercel.json`, Vercel "services" mode):
- `site`: the repo-level `site/` folder, plain static files, built from `Droplet_Intelligence_v2/showcase/` by `python Droplet_Intelligence_v2/build_site.py`. It is the full workspace in the dark full-bleed design: live predictor, five-model compare, "Drop in an SEM image" (image GP in the browser), evaluation, the 1,623 impacts, SEM image model and methods. `stats.json` is generated from the result files. **Re-run it and commit `site/` after changing `showcase/`, `models/` or `results/`.** The same source builds the Claude artifact with `build_site.py --artifact DIR` (base64 model factors; the page falls back to them automatically). Public on `/(.*)`.
- `legacy_api`: `archive/v1_droplet/droplet_app/api.py` (FastAPI, the v1 model). Public on `/v1/api/*`, e.g. `POST /v1/api/predict`, docs at `/v1/api/docs`. Its `requirements.txt` is API only; the Streamlit dashboard uses `requirements-dashboard.txt`.
- No bindings: neither service calls the other. `.vercelignore` keeps the SEM TIFFs, paper, posters and the rest of the archive out of the upload.
- Checked locally: the built site predicts 2.859 [2.81, 2.91] for water/D200/2.5 mm/1.5 m/s in headless Chromium with no errors; `legacy_api` returns 2.859 at `/v1/api/predict`.

**Tests** (re-checked 1 Oct 2026 on Linux, Python 3.11: all 18 tests pass with the SEM originals in place, demo check returns 2.859 [2.807, 2.913], JS parity ≤3.1×10⁻¹²):
```
cd Droplet_Intelligence_v2
python -m pip install -r requirements-sem.txt     # adds torch + Pillow, needed by the SEM tests
python -m unittest discover -s tests -v            # 18 tests
node tests/check_parity.mjs                        # browser worker vs Python
```
Full retraining (benchmark, nested CV, export, SEM CNN, report) is described in `Droplet_Intelligence_v2/README.md`. It is CPU heavy and not needed to use the models.

**Paper numbers and figures.** Run from inside `paper/`, because the scripts use cwd-relative output names:
```
cd paper
python ablation.py        # slow: GP fits; rewrites ablation.json / ablation_preds.npz
python figs.py            # fig1–fig4 PNGs + forest_rows.json + by_fluid.json
python paper_numbers.py   # rebuilds numbers.json (verified to reproduce the current file exactly)
node build.js             # needs `npm install docx`; writes Droplet_Spreading_Paper_v2.docx
```
The PDF is a conversion of the .docx, for example `soffice --headless --convert-to pdf Droplet_Spreading_Paper_v2.docx`.

**Final poster.** Run from inside its folder; it needs the `fonts/` folder and the images next to it:
```
cd poster/final_A0_portrait
python figs.py            # matplotlib → f_*.svg (reads ../../paper/numbers.json and the v2 results)
python build6.py          # → poster.html
python render.py --pdf    # Playwright + Chromium → Parth_Sharma_Droplet_Poster_A0_Portrait.pdf + check.png
```
`render.py` prints the slack for each column. A negative slack, or `scroll` > 0, means a column overflows the A0 page. It needs `pip install playwright` and `playwright install chromium`. Verified on 1 Oct: running this chain from a clean copy reproduces `poster.html` and every figure exactly.

**Backup dashboard (v1):** `cd archive/v1_droplet/droplet_app`, then `pip install -r requirements-dashboard.txt` and `streamlit run Overview.py`, or run `run_dashboard.bat`.

## Notes on provenance and paths

- The scripts in `paper/` and `poster/` use repo-relative paths (`_ROOT` at the top of each file). `paper_numbers.py` now also regenerates three things that had been added to `numbers.json` by hand on 24 Sept:
  - the `zoo` key (13-model screen, from `archive/v1_droplet/results/zoo_table.csv`);
  - `mu_ratio`;
  - the per-fluid means and viscosity ranges.
- Scripts in `archive/` use paths relative to the repository root; run them from there.
- **Not included, because of size or duplication:**
  - The original dataset zip (148 MB). The five CSVs and the dataset description PDF are in `Droplet_Intelligence_v2/data/raw/`.
- **SEM originals now included (1 Oct 2026):** all 39 TIFFs (13 surfaces × 43/100/350×) are in `Droplet_Intelligence_v2/data/sem_original/`, where `droplet.sem` expects them. Every file matches the SHA-256 in `data/sem/manifest.json`, and `droplet.sem.prepare()` rebuilds `training_pixels.npz` from them with zero pixel difference.
  - Older duplicate zips, render-check screenshots, an older copy of the paper sources, superseded poster scripts (`build3.py`, `build5.py`) and CatBoost training logs.

## Extensions for new surfaces (1 Oct 2026)

Code: `droplet/extensions.py` (experiments), `droplet/predict_image.py` (tool), `droplet/extensions_report.py`. Results and the full table: `Droplet_Intelligence_v2/results/extensions/REPORT.md` and `summary.json`. Same protocol as the paper (LOSO, paired surface bootstrap 4,000 draws seed 23, REF-H never used for choices); the harness reproduces the baseline exactly (0.0400).

- **Step 1, physical image reader: accepted as an input path.** φ is measured from one 43× SEM image (local roughness over a 12.7 µm window, threshold calibrated on training surfaces only). GP on ln Re, ln We, ln D₀ and image φ, with no spacing or depth: LOSO 0.0397 vs 0.0400, difference interval [−0.0068, +0.0045], so equivalent, not better. Reads REF-H as φ = 1.000 (the CNN read 0.709). LOSO interval coverage 88.8%. Valid near 43× only; at 100× and 350× textured φ is under-read, so the tool refuses other pixel sizes.
  ```
  python -m droplet.predict_image --image new_43x.tif --D_mm 2.5 --V 1.5 --rho 997.2 --mu 0.000944 --sigma 0.0725
  ```
- **In the browser:** `showcase/imagereader.mjs` is a port of the reader and the image GP (`models/image_only.json` + `.L.bin`, exported by `python -m droplet.export_image_model`). It reproduces Python exactly on all 13 SEM images (φ difference 0, GP to 1e-10). It powers the "Drop in an SEM image" section of the website and the Claude artifact (https://claude.ai/artifact/8hNvWLNL4Yor9k3DGbhZhj). The interval uses a LOSO-calibrated multiplier (1.739, coverage 90.1%), in the CLI too.
- **Step 2, wettability: rejected on this data.** Lee β₀ correction: LOSO unchanged, but REF-H RMSE 0.228 (bias +0.22) at the literature angle and still 0.151 at 130°. Angle as a GP input: LOSO worse (0.0480, interval excludes zero). The training angles span only about 160 to 167°, so the data cannot teach a wettability effect; surfaces with intermediate angles are needed.

## Automatic impact video measurement (1 Oct 2026)

`droplet/video.py` measures a backlit side-view impact video with no manual steps: Otsu threshold, substrate = dark band across the width, drop = largest dark blob above it; D₀ from area, V from a line fit of the centroid before contact, β_max from the widest blob extent after contact. It then predicts β_max from D₀ and V and checks the measurement against the 90% interval.
```
python -m droplet.video --video impact.mp4 --fps 5000 --mm-per-px 0.02 --fluid 0 --surface D200
python -m droplet.video --watch incoming/ --fps 5000 --mm-per-px 0.02 --fluid 0 --surface D200   # automatic: every new video -> results.csv
```
Browser port: `showcase/video.mjs` (website section "Drop in an impact video"). Validated only on synthetic videos with known answers (`droplet/synthetic_video.py`, `tests/test_video.py`): D₀ and V exact, β_max within about 1% (frame sampling of the peak). The Zenodo liquid-film fluorescence videos (Drive folder) are a different problem (impact on a film, no visible drop edge, no scale or capture rate) and were not used. Real validation needs backlit videos with known D₀, V, β_max, e.g. raw videos from Može et al.

## Final poster layout (portrait, built by `build6.py`)

The header carries the conference name, with the Thapar logo top-left and the conference logo top-right. Below it come the title, the subtitle "Testing on laser textured aluminium by holding out one surface at a time", the underlined presenter name and the affiliation.

| Column | Panels |
|---|---|
| 1 | Introduction: 1 Background and objective. Methods: 2 Surface descriptors from SEM, 3 Reading descriptors directly from images |
| 2 | 4 Dimensionless inputs, 5 Evaluation. Results: 6 Comparison of 13 models, 7 Variants tested |
| 3 | 8 Prediction intervals, 9 Error for each held out surface, 10 Smooth plate REF-H, Future work, References |

A "Conclusions" band with 4 points runs along the bottom.

## Open items

1. ~~Pitch scripts out of date for the layout~~: done 1 Oct 2026. The 3-minute script now follows the portrait panels 1 to 10, Future work and the Conclusions band; numbers unchanged.
2. ~~Research summary deliverables table~~: done 1 Oct 2026, now uses repo-relative paths and the portrait poster.
3. **Conference logo.** The poster uses `logo_ws.png`, the droplet "D" mark from the earlier files. The organisers' email mentioned an attached official logo that has not been supplied yet. If it differs, replace `poster/final_A0_portrait/logo_ws.png` (the `.r` image in `build6.py`) and re-render.
4. **Print quality.** The TIET logo is 277×258 px and the cover image is 1132×1600 px. Larger or vector originals would print better at A0.
5. ~~Paper author list~~: confirmed 1 Oct 2026 (body text, .docx and PDF metadata: Parth Sharma, sole author).
6. **Planning doc diagram.** The architecture diagram in `notes/phase1_audit_and_plan.md` still shows the early design: a Laan prior and a spacing descriptor feeding the GP, and a conformal interval. The final model is the plain GP on the five inputs above.
7. **Research next steps** (from the paper; steps 1 and 2 of the extensions above address the first two in part):
   - measured advancing and receding contact angles for every fluid–surface pair, as a wettability input;
   - a noise model that depends on the Ohnesorge number, to fix water undercoverage;
   - new surfaces, with a test set fixed before any tuning;
   - re-testing the Laan backbone on repeat seeds.
