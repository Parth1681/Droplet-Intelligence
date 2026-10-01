"""Follow-ups triggered by Phase 2: (a) REF-H encoding sensitivity of GPR on raw inputs;
(b) raw fluid/impact inputs + SEM surface descriptors (phi, texvol) instead of spacing/depth."""
import json, sys
import numpy as np
from src.phases import build, GPR, evaluate, RAW, rmse

t, r = build()
if sys.argv[1] == "enc":
    m = GPR(RAW).fit(t)
    out = {}
    for sp, dp in [(0, 0), (50, 6), (800, 6), (800, 25), (5000, 0)]:
        rr = r.assign(spacing=float(sp), depth=float(dp))
        out[f"{sp}/{dp}"] = rmse(np.exp(m.predict(rr)), rr.beta.values)
    print(json.dumps(out)); json.dump(out, open("results/phase_extra_enc.json", "w"))
else:
    row = evaluate("GPR | raw fluid/impact (D,V,rho,sigma,mu) + phi, texvol", lambda: GPR(["D", "V", "rho", "sigma", "mu", "phi", "texvol"]), t, r, intervals=True, do_id=True)
    json.dump([row], open("results/phase_extra_raw_phi.json", "w"), indent=1)
