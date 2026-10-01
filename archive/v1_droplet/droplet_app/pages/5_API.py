import json
import streamlit as st
from core import Predictor, fluids, surfaces

st.set_page_config(page_title="API · Droplet β_max", layout="wide")
st.title("API")
P = Predictor()
st.write("The same model is served over HTTP by `api.py` (FastAPI). Start it in a second terminal:")
st.code("cd droplet_app\nuvicorn api:app --port 8000", language="bash")
st.write("Interactive docs are then at http://localhost:8000/docs.")
st.subheader("Endpoints")
st.markdown("""
| Method | Path | Returns |
|---|---|---|
| GET | `/health` | model name and status |
| GET | `/fluids` | the five glycerol–water mixtures with ρ, σ, μ |
| GET | `/surfaces` | the 13 surfaces with pitch, depth, φ, V_tex |
| POST | `/predict` | β_max, 90% interval, wide (worst-surface) interval, Re, We, φ, OOD percentile, warnings |
""")
st.subheader("Example")
body = {"D_mm": 2.5, "V": 1.5, "fluid": list(fluids())[0], "surface": "D200"}
st.code(f"curl -X POST http://localhost:8000/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '{json.dumps(body)}'", language="bash")
fl = fluids()[body["fluid"]]; s = surfaces()["D200"]
st.write("Response:")
st.json(P.predict(2.5, 1.5, fl["rho"], fl["sigma"], fl["mu"], s["spacing_um"], s["depth_um"]))
st.subheader("From Python")
st.code("""import requests
r = requests.post("http://localhost:8000/predict",
                  json={"D_mm": 2.5, "V": 1.5, "fluid": "0 wt.% glycerol", "surface": "D200"})
print(r.json()["beta_max"])""", language="python")
