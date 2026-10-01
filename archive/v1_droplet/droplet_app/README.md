# Droplet β_max: local dashboard + API

## Run (Windows)
Double-click `run_dashboard.bat`. It installs the packages, starts the API in a second window and opens the dashboard in your browser.

## Run (macOS / Linux)
    ./run_dashboard.sh

## Run by hand
    pip install -r requirements.txt
    uvicorn api:app --port 8000          # API, docs at http://localhost:8000/docs
    streamlit run Overview.py            # dashboard at http://localhost:8501

## Pages
1. Overview: final model, headline errors, and when not to trust it
2. Predict: β_max with a 90% interval and OOD warnings; velocity sweep
3. Benchmark: M1–M6 ladder, physics residual vs M2 with CIs, per-surface error
4. Uncertainty & OOD: parity with intervals, coverage per surface, OOD score vs error
5. Surfaces & SEM: φ / V_tex per surface, SHAP, CNN reading φ from SEM
(+ an API page with the curl/Python examples)

## Files
- `core.py`: numpy-only predictor (no sklearn needed; checked against sklearn to 1e-6 at export)
- `api.py`: FastAPI service
- `model/`: exported GP (bundle.npz) and metadata (meta.json)
- `data/`: result files the dashboard shows
- `export_model.py`: regenerates `model/` from the research repo (not needed to run the app)

Python 3.9+.
