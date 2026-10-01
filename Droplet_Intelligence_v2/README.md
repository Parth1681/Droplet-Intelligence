# Droplet Intelligence v2.1

Working trained models, measured dataset, independent surface evaluation, browser prediction dashboard and Python API. Prepared for Parth Sharma on 23 September 2026.

**The baseline GP remains the default.** Four original candidates plus an experimental SEM CNN + GP were trained. No statistically supported accuracy improvement over the original GP was established. The release strengthens evaluation, calibration isolation, reproducibility and deployment. See `results/REPORT.md` for measured results and limitations.

## Run the working demo

Requires Python 3.10+ (tested on the version recorded in `models/release.json`). From this folder:

```bash
python -m pip install -r requirements.txt
python -m droplet.serve --port 8000
```

Open http://127.0.0.1:8000 . On Windows, `run_demo.bat` installs requirements and starts the same app. On macOS/Linux, run `sh run_demo.sh`. Internet access is needed for dependency installation; inference then works locally. Models are already trained; you do not need to retrain to use the demo.

The website now has a fifth SEM image-model view, in addition to: live prediction and model comparison; model/surface accuracy and interval coverage; filterable experimental data; architecture and reproducibility. The hosted site runs inference in the browser. The API below runs with the local Python server.

## API

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"D_mm":2.5,"V":1.5,"rho":997.2375,"mu":0.000943923,"sigma":0.07246,"surface":"D200"}'
```

`GET /api/health`, `GET /api/metadata`, `POST /api/predict`.
Invalid inputs return HTTP 422; a non-JSON content type returns 415. `model` can be `baseline`, `matern52`, `structured`, `physics_residual`, or `sem_gp`. For `surface="custom"`, explicitly supply `spacing`, `depth`, `phi`, and `texvol`. Do not attach geometry overrides to a named surface.

Units: D_mm in mm; velocity m/s; density kg/m³; viscosity Pa·s; surface tension N/m; spacing/depth/texture volume per area in µm; phi is a fraction. Core computations convert diameter to SI. Predictions are the lognormal median. The returned interval has a 90% empirical target; coverage is not guaranteed for a new surface or distribution shift.

## Reproduce training

```bash
python -m pip install -r requirements-sem.txt
python -m droplet.benchmark --workers 3
python -m droplet.nested --workers 3
python -m droplet.export
python -m droplet.sem --workers 2
python -m unittest discover -s tests -v
node tests/check_parity.mjs
python -m droplet.report
```

Node.js is needed only for the cross-language parity check. Set workers to 1 on a small computer. Exact GP training is CPU/memory intensive; this run used single-thread linear algebra per worker and three workers. Full reproduction includes 48 outer fits, 144 inner fits and four deployment fits. No GPU or paid API is required. Changing the model/data invalidates caches; changing unrelated Python files also invalidates the conservative outer cache. Saved logs include optimizer warnings.

## Files

- `droplet/`: ingestion, physics features, four models, benchmark, nested evaluation, export, numerical inference, API/server and report generation.
- `data/raw/`: original five CSVs supplied by the user, preserved byte-for-byte.
- `data/experiments.csv`: consolidated derived table with original target values.
- `models/`: five numerical model bundles plus the SEM encoder, integrity hashes and deployment metadata.
- `results/`: all outer/inner predictions, split membership, metrics, bootstrap comparisons, plots and report.
- `web/`: complete browser interface and float64 GP worker.
- `tests/`: Python research invariants and Node numerical parity.
- `docs/`: implemented architecture, source provenance and earlier design specification.

## Scope and provenance

Training uses 1,498 textured impacts; 125 REF-H impacts remain out of training. Fixed geometry proxies from the supplied earlier implementation are retained. Contact-angle measurements are incomplete and excluded from this release; do not impute them as if measured. Footer-free SEM training arrays, image thumbnails and trained CNN weights are included. Original TIFFs, original posters and unrelated archives are not duplicated in this package. Dataset redistribution rights remain those of its original authors; no new license or experimental authorship is asserted. See `docs/DATA_PROVENANCE.md`.

This is a research system. The number of independent surfaces is small, REF-H was already inspected, optimizer diagnostics exist, and nested CV is not a substitute for a prospective external test. Read the report before using the metrics in a paper or poster.

## SEM photo-trained extension

See `results/sem/REPORT.md` for the actual photo training and results. Install `requirements-sem.txt` before running the full 18-test suite or processing new images. Ordinary GP inference and the website do not require PyTorch.

```bash
python -m pip install -r requirements-sem.txt
python -m droplet.sem --workers 2
python -m droplet.sem_infer --images surface_43.tif surface_100.tif surface_350.tif --D_mm 2.5 --V 1.5
```

The new image branch is available as model `sem_gp`. For named surfaces the API uses their precomputed CNN descriptors. Custom descriptor inputs remain the caller’s responsibility; use `sem_infer` to compute them from actual images.
