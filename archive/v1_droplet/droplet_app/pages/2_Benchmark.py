import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

D = Path(__file__).resolve().parents[1] / "data"
st.set_page_config(page_title="Benchmark · Droplet β_max", layout="wide")
st.title("Benchmark")
p7 = json.load(open(D / "p7.json")); p34 = json.load(open(D / "p34.json"))
st.subheader("Model ladder (Phases 2–7)")
B = pd.DataFrame(p7["table"]).rename(columns={"id": "ID", "model": "Model", "ID": "ID RMSE", "LOSO": "LOSO RMSE",
                                               "LOSO_worst": "Worst surface", "REFH": "REF-H RMSE", "REFH_MAPE": "REF-H MAPE %",
                                               "REFH_bias": "REF-H bias"})
st.dataframe(B.style.format({c: "{:.4f}" for c in ["ID RMSE", "LOSO RMSE", "REF-H RMSE", "REF-H bias"]} | {"REF-H MAPE %": "{:.2f}"}),
             hide_index=True, use_container_width=True)
st.info(f"Final: **{p7['final']}**. Rule: {p7['rule']}.")

st.subheader("GPR vs XGBoost on identical inputs")
if (D / "fig_gpr_vs_xgb.png").exists():
    gx = json.load(open(D / "gpr_vs_xgb.json"))
    st.image(str(D / "fig_gpr_vs_xgb.png"), use_container_width=True)
    ci = gx["LOSO_GPR_minus_XGB_CI"]
    st.caption(f"Unseen surface: GPR {gx['GPR']['LOSO']:.3f} vs XGBoost {gx['XGB']['LOSO']:.3f} "
               f"(paired 95% CI of the difference {ci[0]:.3f} to {ci[1]:.3f}); smooth plate: {gx['GPR']['REFH']:.3f} vs {gx['XGB']['REFH']:.3f}.")

st.subheader("Physics residual vs M2 (Phase 3–4)")
rows = []
for k in ["M5", "M6", "GPR"]:
    v = p34[k]
    rows += [{"Model": k, "Test": "REF-H", "Δ RMSE vs M2": v["REFH_minus_M2"], "lo": v["REFH_minus_M2_CI95"][0], "hi": v["REFH_minus_M2_CI95"][1]},
             {"Model": k, "Test": "LOSO", "Δ RMSE vs M2": v["LOSO_minus_M2"], "lo": v["LOSO_minus_M2_CI95"][0], "hi": v["LOSO_minus_M2_CI95"][1]}]
C = pd.DataFrame(rows)
fig = px.scatter(C, x="Δ RMSE vs M2", y="Model", color="Test", error_x=C.hi - C["Δ RMSE vs M2"],
                 error_x_minus=C["Δ RMSE vs M2"] - C.lo, color_discrete_map={"REF-H": "#2f4574", "LOSO": "#2a78d6"})
fig.add_vline(x=0, line_dash="dash"); fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("95% CI from a paired bootstrap: LOSO resamples the 12 surfaces, REF-H its replicate conditions. Left of zero = lower error than M2.")

st.subheader("Per-surface LOSO error")
per = pd.DataFrame({k: p34[k]["LOSO_per_surface"] for k in ["M2", "M5", "M6", "GPR"]})
fig2 = px.bar(per.reset_index().melt(id_vars="index", var_name="Model", value_name="RMSE"), x="index", y="RMSE",
              color="Model", barmode="group", labels={"index": "Held-out surface"})
fig2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10)); st.plotly_chart(fig2, use_container_width=True)

if (D / "zoo_table.csv").exists():
    st.subheader("13-model zoo on the same features (earlier phase)")
    st.dataframe(pd.read_csv(D / "zoo_table.csv"), hide_index=True, use_container_width=True)
