import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import Predictor, fluids, surfaces

st.set_page_config(page_title="Predict · Droplet β_max", layout="wide")
P = Predictor(); FL, SF = fluids(), surfaces()
st.title("Predict β_max")

L, R = st.columns([1, 2])
with L:
    D_mm = st.number_input("Droplet diameter D₀ (mm)", 1.0, 4.0, 2.5, 0.05, key="d")
    V = st.slider("Impact velocity V (m/s)", 0.3, 2.5, 1.5, 0.02, key="v", help="Measured range: 0.48–1.71 m/s. Outside it the model extrapolates.")
    fl = st.selectbox("Fluid", list(FL) + ["Custom"], key="fl")
    if fl == "Custom":
        rho = st.number_input("ρ (kg/m³)", 800.0, 1400.0, 1000.0, key="rho")
        sigma = st.number_input("σ (N/m)", 0.02, 0.08, 0.072, 0.001, format="%.3f", key="sig")
        mu = st.number_input("μ (Pa·s)", 0.0005, 0.5, 0.001, 0.0005, format="%.4f", key="mu")
    else:
        rho, sigma, mu = FL[fl]["rho"], FL[fl]["sigma"], FL[fl]["mu"]
    sf = st.selectbox("Surface", list(SF) + ["Custom geometry"], index=list(SF).index("D200"), key="sf")
    if sf == "Custom geometry":
        sp = st.number_input("Track pitch (µm)", 0.0, 2000.0, 300.0, 10.0, key="sp")
        dp = st.number_input("Channel depth (µm)", 0.0, 40.0, 25.0, 1.0, key="dp")
    else:
        sp, dp = SF[sf]["spacing_um"], SF[sf]["depth_um"]

out = P.predict(D_mm, V, rho, sigma, mu, sp, dp)
with R:
    c = st.columns(3)
    c[0].metric("β_max", f"{out['beta_max']:.3f}")
    c[1].metric("90% interval", f"{out['interval90'][0]:.2f} – {out['interval90'][1]:.2f}")
    c[2].metric("OOD percentile", f"{out['ood_percentile']:.0f}")
    st.caption(f"Re = {out['Re']:.0f} · We = {out['We']:.1f} · φ = {out['phi']:.3f} · V_tex = {out['texvol_um']:.2f} µm · "
               f"wide (worst-surface) band {out['interval90_wide'][0]:.2f} – {out['interval90_wide'][1]:.2f}")
    for w in out["warnings"]: st.warning(w)
    # sweep over velocity
    vs = np.linspace(0.3, 2.5, 60); rows = [P.predict(D_mm, v, rho, sigma, mu, sp, dp) for v in vs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[vs, vs[::-1]], y=np.r_[[r["interval90"][1] for r in rows], [r["interval90"][0] for r in rows][::-1]],
                             fill="toself", line=dict(width=0), fillcolor="rgba(42,120,214,0.18)", name="90% interval"))
    fig.add_trace(go.Scatter(x=vs, y=[r["beta_max"] for r in rows], line=dict(color="#2f4574", width=3), name="β_max"))
    fig.add_trace(go.Scatter(x=[V], y=[out["beta_max"]], mode="markers", marker=dict(size=12, color="#eb6834"), name="this impact"))
    fig.add_vrect(x0=0.48, x1=1.71, fillcolor="rgba(27,175,122,0.07)", line_width=0, annotation_text="measured range", annotation_position="top left")
    fig.update_layout(xaxis_title="Impact velocity V (m/s)", yaxis_title="β_max", height=420, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
with st.expander("Same request through the API"):
    body = {"D_mm": D_mm, "V": V}
    body.update({"fluid": fl} if fl != "Custom" else {"rho": rho, "sigma": sigma, "mu": mu})
    body.update({"surface": sf} if sf != "Custom geometry" else {"spacing_um": sp, "depth_um": dp})
    import json
    st.code(f"curl -X POST http://localhost:8000/v1/api/predict -H 'Content-Type: application/json' -d '{json.dumps(body)}'", language="bash")
