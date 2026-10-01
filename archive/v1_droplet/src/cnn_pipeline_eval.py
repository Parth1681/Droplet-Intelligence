"""End-to-end check: SEM image -> CNN descriptors -> GPR -> beta_max, for surfaces the CNN and GPR never saw."""
import json
import numpy as np
from src.data import load
from src.models_extra import feats, F, gpr
from src.benchmark import metrics

t, r = load(); t = feats(t); r = feats(r.assign(spacing=0.0, depth=0.0))
cnn = json.load(open("results/cnn_descriptors.json"))
res = {}
for s in list(t["sample"].unique()) + ["REF-H"]:
    tr = t if s == "REF-H" else t[t["sample"] != s]
    te = (r if s == "REF-H" else t[t["sample"] == s]).copy()
    m = gpr().fit(tr[F].values, np.log(tr.beta.values))
    p_true = np.exp(m.predict(te[F].values))
    phi, tv = cnn[s]["pred"]; te["phi"], te["texvol"] = phi, tv * 25.0
    p_cnn = np.exp(m.predict(te[F].values))
    res[s] = dict(rmse_measured_desc=metrics(te.beta.values, p_true)["RMSE"],
                  rmse_cnn_desc=metrics(te.beta.values, p_cnn)["RMSE"])
    print(s, {k: round(v, 4) for k, v in res[s].items()}, flush=True)
json.dump(res, open("results/cnn_pipeline_eval.json", "w"), indent=1)
tex = [k for k in res if k != "REF-H"]
print("LOSO mean RMSE measured:", np.mean([res[k]["rmse_measured_desc"] for k in tex]),
      "CNN:", np.mean([res[k]["rmse_cnn_desc"] for k in tex]))
