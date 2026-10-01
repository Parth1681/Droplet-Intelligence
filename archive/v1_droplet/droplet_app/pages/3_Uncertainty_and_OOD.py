import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

D = Path(__file__).resolve().parents[1] / "data"
st.set_page_config(page_title="Uncertainty · Droplet β_max", layout="wide")
st.title("Uncertainty and out-of-distribution warnings")
p5 = json.load(open(D / "p5.json")); p7 = json.load(open(D / "p7.json")); k = p7["final"]; g = p5[k]
c = st.columns(3)
c[0].metric("90% band coverage, unseen textured surface (q from the other 11)", f"{g['cov_conformal_LOSO']:.0%}")
c[1].metric("90% band coverage, smooth REF-H", f"{g['cov_conformal_REFH']:.0%}")
c[2].metric("Wide (worst-surface) band, REF-H", f"{g['cov_surface_conformal_REFH']:.0%}")
st.write(f"Conformal multiplier on the GP std: **q = {g['q_conformal']:.2f}**, calibrated on leave-one-surface-out residuals. Coverage per surface uses a q calibrated on the other 11 surfaces only. "
         f"Lowest single-surface coverage: {g['worst_surface']} at {g['min_surface_coverage']:.0%}.")

L, R = st.columns(2)
lo = pd.read_csv(D / "loso_pred.csv"); rf = pd.read_csv(D / "refh_pred.csv")
with L:
    st.subheader("Parity with 90% intervals")
    which = st.radio("Set", ["LOSO (textured)", "REF-H (smooth, blind)"], horizontal=True, key="set")
    d = lo if which.startswith("LOSO") else rf
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.beta, y=d.pred, mode="markers", marker=dict(size=5, color=d.gly, colorscale="Blues", showscale=True,
                             colorbar=dict(title="glycerol %")), error_y=dict(type="data", symmetric=False, array=d.hi - d.pred,
                             arrayminus=d.pred - d.lo, thickness=0.6, color="rgba(47,69,116,0.25)"), name="impacts"))
    m = [d.beta.min(), d.beta.max()]; fig.add_trace(go.Scatter(x=m, y=m, mode="lines", line=dict(dash="dash", color="gray"), name="y = x"))
    fig.update_layout(xaxis_title="measured β_max", yaxis_title="predicted β_max", height=440, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with R:
    st.subheader("Coverage per held-out surface (q from the other 11)")
    cs = pd.Series(g["cov_conformal_per_surface"]).rename("coverage").reset_index()
    fig = px.bar(cs, x="index", y="coverage", labels={"index": "surface"}); fig.add_hline(y=0.9, line_dash="dash")
    fig.update_layout(height=440, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fig, use_container_width=True)

st.subheader("Does the OOD score warn when the error is high?")
O = pd.DataFrame(p5["ood"]); sp = p5["ood_spearman_flag_vs_rmse"]
fig = px.scatter(O, x="flagged", y="rmse", text="surface", labels={"flagged": "share of impacts flagged (≥ 95th pct)", "rmse": "RMSE"})
fig.update_traces(textposition="top center"); fig.update_layout(height=420, xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Spearman ρ across the 12 textured surfaces = {sp['rho']:+.2f} (p = {sp['p']:.2f}). REF-H shown for reference; "
           "the detector sees only geometry and flow inputs, so it cannot see the wettability change on the smooth plate.")
