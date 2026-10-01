"""Droplet β_max dashboard. Run from inside droplet_app/:   streamlit run Overview.py"""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from core import Predictor

D = Path(__file__).resolve().parent / "data"
st.set_page_config(page_title="Droplet β_max", page_icon="💧", layout="wide")
P = Predictor(); p7 = json.load(open(D / "p7.json")); p5 = json.load(open(D / "p5.json"))
k = P.meta["model_key"]; row = next(d for d in p7["table"] if d["id"] == k)

st.title("Droplet β_max on laser-textured surfaces")
st.write("Predicts the maximum spreading ratio β_max = D_max/D₀ of a droplet hitting a laser-textured or smooth "
         "aluminium surface, from droplet size, speed, fluid and surface geometry. The surface enters through two "
         "descriptors measured from SEM images: the smooth-plateau fraction φ and the texture volume V_tex.")
c = st.columns(4)
c[0].metric("Final model", k, help=P.meta["model_name"])
c[1].metric("Error on an unseen textured surface (LOSO RMSE)", f"{row['LOSO']:.3f}")
c[2].metric("Error on the smooth plate, never trained on (REF-H RMSE)", f"{row['REFH']:.3f}")
c[3].metric("90% interval coverage, unseen textured surface", f"{p5[k]['cov_conformal_LOSO']:.0%}")
st.subheader("How the model was chosen")
st.markdown(f"""
- **Three tests, one rule.** Grouped 5-fold CV over replicate conditions (ID), leave-one-surface-out over the 12 textured
  surfaces (LOSO), and one blind prediction of the smooth REF-H plate. The model is picked on **LOSO** alone; REF-H is never used for tuning.
- **Rule:** {p7['rule']}.
- **Result:** `{k}`, {P.meta['model_name']}.
""")
st.subheader("Where it should not be trusted")
st.markdown(f"""
- **Smooth or hydrophilic surfaces.** All textured surfaces are superhydrophobic (θ ≈ 162–167°). On REF-H the 90% band
  covered only **{p5[k]['cov_conformal_REFH']:.0%}** of impacts. Use the wide (worst-surface) band there.
- **The OOD score does not catch this.** It sees only flow and geometry inputs, and REF-H looks ordinary to it.
- **Outside the training ranges** of Re, We and D₀; the Predict page warns when an input is outside them.
""")
st.caption("Pages: Predict · Benchmark · Uncertainty & OOD · Surfaces & SEM. Data: Mendeley droplet-impact dataset (Može et al.).")
