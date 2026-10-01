import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core import surfaces

D = Path(__file__).resolve().parents[1] / "data"
st.set_page_config(page_title="Surfaces · Droplet β_max", layout="wide")
st.title("Surfaces and what the model sees")
SF = pd.DataFrame(surfaces()).T.reset_index(names="surface")
st.write("Each surface is a square grid of laser tracks (deep ≈ 25 µm, shallow ≈ 6 µm) at a given pitch. The model does not see "
         "pitch and depth directly. It sees **φ**, the smooth-plateau area fraction (1 for the smooth plate), and **V_tex = (1 − φ) × depth**, "
         "both defined from SEM-measured track widths.")
c = st.columns(3)
for col, (img, cap) in zip(c, [("D200_43.jpg", "D200: deep tracks, 200 µm pitch"), ("S800_43.jpg", "S800: shallow tracks, 800 µm pitch"),
                                 ("REF-H_43.jpg", "REF-H: smooth plate (φ = 1)")]):
    if (D / img).exists(): col.image(str(D / img), caption=cap, use_container_width=True)
st.dataframe(SF.style.format({"spacing_um": "{:.0f}", "depth_um": "{:.0f}", "phi": "{:.3f}", "texvol_um": "{:.2f}"}),
             hide_index=True, use_container_width=True)

if (D / "p6.json").exists():
    st.subheader("What drives the prediction (SHAP)")
    p6 = json.load(open(D / "p6.json"))
    S = pd.DataFrame({"textured": p6["mean_abs_shap_textured"], "REF-H": p6["mean_abs_shap_REFH"]}).reset_index(names="feature")
    fig = px.bar(S.melt(id_vars="feature", var_name="set", value_name="mean |SHAP|"), y="feature", x="mean |SHAP|", color="set",
                 barmode="group", orientation="h", color_discrete_map={"textured": "#2f4574", "REF-H": "#eb6834"})
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fig, use_container_width=True)
    if (D / "shap.npz").exists():
        z = np.load(D / "shap.npz"); feats = list(p6["mean_abs_shap_textured"])
        f = st.selectbox("Dependence plot for", feats, key="dep")
        i = feats.index(f)
        fig = px.scatter(x=z["X"][:, i], y=z["sv"][:, i], labels={"x": f, "y": f"SHAP value of {f} (log β)"})
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fig, use_container_width=True)

if (D / "cnn_descriptors.json").exists():
    st.subheader("Reading φ straight from an SEM image (CNN)")
    C = json.load(open(D / "cnn_descriptors.json"))
    T = pd.DataFrame([{"surface": k, "φ from geometry": v["true"][0], "φ read by CNN": v["pred"][0],
                       "held out": "blind" if k == "REF-H" else "leave-one-surface-out"} for k, v in C.items()])
    fig = px.scatter(T, x="φ from geometry", y="φ read by CNN", text="surface", color="held out")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    fig.update_traces(textposition="top center"); fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("A small CNN trained on SEM patches of the other surfaces reads φ for the held-out one; REF-H is never shown in training.")
