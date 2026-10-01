"""Builds webdash/index.html (+ gp_L.bin) from the exported model and result files."""
import json, os
import numpy as np
import pandas as pd

ROOT = "/home/claude/droplet"; APP = f"{ROOT}/droplet_app"; OUT = f"{ROOT}/webdash"
b = np.load(f"{APP}/model/bundle.npz"); meta = json.load(open(f"{APP}/model/meta.json"))
L = b["L"]; n = len(L)
L[np.tril_indices(n)].astype("<f4").tofile(f"{OUT}/gp_L.bin")     # packed lower triangle, row-major

D = f"{APP}/data"
J = lambda f: json.load(open(f"{D}/{f}"))
lo, rf = pd.read_csv(f"{D}/loso_pred.csv"), pd.read_csv(f"{D}/refh_pred.csv")
sh = np.load(f"{D}/shap.npz")
r4 = lambda a: np.round(np.asarray(a, float), 5).tolist()
model = {"Xtr": r4(b["Xtr"]) if False else np.asarray(b["Xtr"]).tolist(), "alpha": np.asarray(b["alpha"]).tolist(),
         "mu_x": b["mu_x"].tolist(), "sd_x": b["sd_x"].tolist(), "ls": b["ls"].tolist(), "c": float(b["c"]),
         "noise": float(b["noise"]), "y_mean": float(b["y_mean"]), "y_std": float(b["y_std"]),
         "mcd_loc": b["mcd_loc"].tolist(), "mcd_prec": b["mcd_prec"].tolist(), "mah_sorted": r4(b["mah_sorted"]), "n": n}
data = {"meta": meta, "model": model, "p34": J("p34.json"), "p5": J("p5.json"), "p6": J("p6.json"), "p7": J("p7.json"),
        "gx": json.load(open(f"{ROOT}/results/final/gpr_vs_xgb.json")),
        "zoo": pd.read_csv(f"{D}/zoo_table.csv")[["name", "ID RMSE", "LOSO RMSE", "REF-H RMSE"]].round(4).to_dict("records"),
        "cnn": J("cnn_descriptors.json"),
        "loso": {c: (r4(lo[c]) if c != "surface" else lo[c].tolist()) for c in ["surface", "gly", "beta", "pred", "lo", "hi"]},
        "refh": {c: r4(rf[c]) for c in ["gly", "beta", "pred", "lo", "hi"]},
        "shap": {"sv": r4(sh["sv"]), "X": r4(sh["X"])}}
# per-impact absolute errors for the GPR vs XGB CDF
pg, px = np.load(f"{ROOT}/results/final/pred_GPR.npz"), np.load(f"{ROOT}/results/final/pred_XGB.npz")
from sys import path; path.insert(0, ROOT)
from src.data import load
t, _ = load()
data["gx_abs"] = {"GPR": r4(np.sort(np.abs(pg["loso"] - t.beta.values))), "XGB": r4(np.sort(np.abs(px["loso"] - t.beta.values)))}
tpl = open(f"{OUT}/template.html").read()
html = tpl.replace("/*%%DATA%%*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
open(f"{OUT}/index.html", "w").write(html)
print("index.html", os.path.getsize(f"{OUT}/index.html") // 1024, "KB; gp_L.bin", os.path.getsize(f"{OUT}/gp_L.bin") // 1024, "KB")
