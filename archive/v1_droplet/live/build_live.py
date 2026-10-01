"""Builds live/index.html from results on disk. Re-run + republish after each stage."""
import json, os, html, time, re
import numpy as np

ROOT = "archive/v1_droplet"
R = f"{ROOT}/results"; FR = f"{R}/final"


def J(p, d=None):
    try: return json.load(open(p))
    except Exception: return d


def esc(s): return html.escape(str(s))
def f4(x): return "—" if x is None else f"{x:.4f}"
def f3(x): return "—" if x is None else f"{x:.3f}"
def pct(x): return "—" if x is None else f"{x*100:.0f}%"


status = J(f"{FR}/status.json", {})
p34, p5, p6, p7 = (J(f"{FR}/{k}.json") for k in ["p34", "p5", "p6", "p7"])
p2b = J(f"{R}/phase2_benchmark.json", [])
extra = J(f"{ROOT}/live/extra_status.json", {})


def st(stage_keys, done_if):
    if done_if: return "done"
    if any(status.get(k, {}).get("state") == "running" for k in stage_keys): return "running"
    return "queued"


fit_running = any(v.get("state") == "running" for k, v in status.items() if k.startswith("fit_"))
PH = [
    ("Sep 23–24", "1. Numeric audit", "done", "Re/We recalc matches, distributions, REF-H schema, rebound label confirmed"),
    ("Sep 25–26", "2. Baselines M1–M5", "done", "Grouped-CV, LOSO and REF-H numbers for all baselines"),
    ("Sep 27–28", "3–4. Physics residual M5/M6", "running" if (fit_running and not p34) else st(["p34"], p34), "REF-H error vs M2, with CI"),
    ("Sep 29", "5. Uncertainty + OOD", st(["p5"], p5), "Conformal coverage on LOSO and REF-H"),
    ("Sep 30", "6–7. SHAP + benchmark table", st(["p6", "p7"], p6 and p7), "Final model chosen on evidence"),
    ("Oct 1–2", "9–10. API + dashboard", extra.get("api", "queued"), "5 pages running locally"),
    ("Oct 3–4", "11. Pitch + poster tie-in", extra.get("pitch", "queued"), "30 s / 1 min / 3 min scripts, rehearsed demo"),
]
LBL = {"done": "Done", "running": "Running", "queued": "Queued"}
n_done = sum(p[2] == "done" for p in PH)

# ---------------------------------------------------------------- sections
def table(head, rows, num_from=1, hi=None):
    h = "".join(f"<th{' class=n' if i >= num_from else ''}>{esc(c)}</th>" for i, c in enumerate(head))
    b = ""
    for r in rows:
        cls = ' class="hi"' if hi and hi(r) else ""
        b += f"<tr{cls}>" + "".join(f"<td{' class=n' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def pending(msg): return f'<p class="pend">{esc(msg)}</p>'


# Phase 2
rows2 = []
for d in p2b:
    rows2.append([esc(d["Model"]), f4(d["ID RMSE"]), f4(d["LOSO RMSE"]), f4(d["REF-H RMSE"]),
                  f'{d["REF-H CI95"][0]:.3f}–{d["REF-H CI95"][1]:.3f}'])
sec2 = table(["Model", "ID RMSE", "LOSO RMSE", "REF-H RMSE", "REF-H 95% CI"], rows2,
             hi=lambda r: r[0].startswith("M3"))

# Phase 3-4: CI plot of REF-H RMSE minus M2
def ci_plot(d):
    items = [(k, d[k]) for k in ["M5", "M6", "GPR"]]
    lo = min(min(v["REFH_minus_M2_CI95"][0], v["LOSO_minus_M2_CI95"][0]) for _, v in items)
    hi_ = max(max(v["REFH_minus_M2_CI95"][1], v["LOSO_minus_M2_CI95"][1]) for _, v in items)
    lo, hi_ = min(lo, -0.005) * 1.15, max(hi_, 0.005) * 1.15
    W, L, Rm, rowh = 640, 150, 30, 30
    H = 40 + len(items) * 2 * rowh + 30
    X = lambda v: L + (v - lo) / (hi_ - lo) * (W - L - Rm)
    s = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Difference in RMSE versus M2 with 95% CI">'
    ticks = np.linspace(lo, hi_, 5)
    for tv in ticks:
        s += f'<line x1="{X(tv):.1f}" x2="{X(tv):.1f}" y1="24" y2="{H-28}" class="grid"/>'
        s += f'<text x="{X(tv):.1f}" y="{H-10}" class="tick" text-anchor="middle">{tv:+.3f}</text>'
    s += f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="18" y2="{H-28}" class="zero"/>'
    s += f'<text x="{X(0)-6:.1f}" y="14" class="tick" text-anchor="end">better than M2</text>'
    s += f'<text x="{X(0)+6:.1f}" y="14" class="tick">worse</text>'
    y = 40
    for k, v in items:
        for lab, m, ci, cl in [("REF-H", v["REFH_minus_M2"], v["REFH_minus_M2_CI95"], "mk-ref"),
                               ("LOSO", v["LOSO_minus_M2"], v["LOSO_minus_M2_CI95"], "mk-lo")]:
            s += f'<text x="{L-12}" y="{y+4}" class="lab" text-anchor="end">{k} · {lab}</text>'
            s += f'<line x1="{X(ci[0]):.1f}" x2="{X(ci[1]):.1f}" y1="{y}" y2="{y}" class="whisk {cl}"/>'
            s += f'<circle cx="{X(m):.1f}" cy="{y}" r="6" class="{cl}"/>'
            y += rowh
    return s + "</svg>"


if p34:
    r34 = []
    for k in ["M2", "M5", "M6", "GPR"]:
        v = p34[k]
        ci = v.get("REFH_minus_M2_CI95")
        r34.append([f"<b>{k}</b> {esc(v['name'])}", f4(v["ID"]), f4(v["LOSO"]), f4(v["REFH"]),
                    "reference" if k == "M2" else f'{v["REFH_minus_M2"]:+.4f} [{ci[0]:+.3f}, {ci[1]:+.3f}]'])
    c = p34["M6_vs_GPR_LOSO_CI95"]; cr = p34["M6_vs_GPR_REFH_CI95"]
    sec34 = (table(["Model", "ID", "LOSO", "REF-H", "REF-H − M2 [95% CI]"], r34, hi=lambda r: "M6" in r[0])
             + f'<figure class="chart">{ci_plot(p34)}<figcaption>RMSE difference from M2; paired bootstrap, 2000 draws: LOSO resamples the 12 surfaces, REF-H its 25 replicate conditions. '
               f'Filled = REF-H, open = LOSO. Left of zero means better than M2.</figcaption></figure>'
             + f'<p class="note">Does the Laan backbone help the GP? M6 − plain GPR: LOSO [{c[0]:+.4f}, {c[1]:+.4f}] (surfaces resampled; M6 better on {p34.get("M6_wins_surfaces","?")} of 12), REF-H [{cr[0]:+.4f}, {cr[1]:+.4f}]. '
               f'{"Both intervals include 0: no measurable gain from the backbone." if (c[0] < 0 < c[1] and cr[0] < 0 < cr[1]) else "At least one interval excludes 0."}</p>')
else:
    sec34 = pending("Fitting M5 and M6 across 12 LOSO folds + 5 grouped-CV folds. The GPR fits take a few minutes each.")

# Phase 5
if p5:
    g, m6 = p5["GPR"], p5["M6"]
    r5 = [[lab, pct(g[k]), pct(m6[k])] for lab, k in [
        ("Nominal 90% band, LOSO", "cov_nominal_LOSO"), ("Conformal (normalised), LOSO, q from other 11 surfaces", "cov_conformal_LOSO"),
        ("Nominal 90% band, REF-H", "cov_nominal_REFH"), ("Conformal (normalised), REF-H", "cov_conformal_REFH"),
        ("Conformal (absolute), REF-H", "cov_abs_conformal_REFH"), ("Wide band (worst-surface 90th pct), REF-H", "cov_surface_conformal_REFH"),
        ("Lowest single-surface coverage, LOSO (q from other 11)", "min_surface_coverage")]]
    ood = p5["ood"]; sp = p5["ood_spearman_flag_vs_rmse"]
    ro = [[esc(o["surface"]), pct(o["flagged"]), f"{o['iso_med']:.0f}", f"{o['mah_med']:.0f}", f3(o["rmse"])] for o in ood]
    sec5 = (f'<h4>Coverage of 90% intervals (target 90%)</h4>'
            + table(["Interval method", "GPR final", "M6"], r5)
            + f'<p class="note">Conformal quantile q = {g["q_conformal"]:.2f} (GP std multiplier). Worst held-out surface: {esc(g["worst_surface"])} at {pct(g["min_surface_coverage"])}.</p>'
            + f'<h4>OOD flags per held-out surface</h4>'
            + table(["Surface", "Impacts flagged (≥95th pct)", "IsoForest pct", "Mahalanobis pct", "GPR RMSE"], ro,
                    hi=lambda r: r[0] == "REF-H")
            + f'<p class="note">Spearman ρ between flagged fraction and error across the 12 textured surfaces: {sp["rho"]:+.2f} (p = {sp["p"]:.2f}).</p>')
else:
    sec5 = pending("Waits for the M6 and GPR fits (needs their LOSO predictive std).")

# Phase 6-7
if p7:
    rb = [[f"<b>{esc(d['id'])}</b> {esc(d['model'])}", f4(d["ID"]), f4(d["LOSO"]), esc(d["LOSO_worst"]),
           f4(d["REFH"]), f"{d['REFH_MAPE']:.2f}%"] for d in p7["table"]]
    ci = p7["best_minus_runner_up_LOSO_CI95"]
    sec7 = (table(["Model", "ID", "LOSO", "Worst surface", "REF-H", "REF-H MAPE"], rb, hi=lambda r: f"<b>{p7['final']}</b>" in r[0])
            + f'<div class="verdict"><span class="k">Final model</span><span class="v">{esc(p7["final"])}</span>'
              f'<span class="why">Rule: {esc(p7["rule"])}. Lowest LOSO: {esc(p7["lowest_LOSO"])}; '
              f'{esc(p7["lowest_LOSO"])} − {esc(p7["runner_up"])} LOSO, 95% CI resampling the 12 surfaces: [{ci[0]:+.4f}, {ci[1]:+.4f}] → {"tie, keep the simpler model" if p7["tie"] else "clear winner"}.</span></div>')
    cold = J(f"{FR}/cold_start_check.json")
    if cold:
        sec7 += (f'<p class="note"><b>Independent checks.</b> A separate reviewer reproduced every number from the saved predictions and found no leakage '
                 f'(Laan A, scalers and OOD detector are refit inside each fold). Its finding changed the pick: the first version resampled replicate '
                 f'conditions for the LOSO comparison, which made M6 look significantly better. Resampling whole surfaces makes it a tie, so the rule keeps plain GPR. '
                 f'Cold-start GP fits (no warm start from the all-data kernel) reproduce LOSO exactly: GPR {cold["GPR_cold"]:.4f}, M6 {cold["M6_cold"]:.4f}.</p>')
else:
    sec7 = pending("Builds once all six models are fitted.")
if p6:
    imp = p6["mean_abs_shap_textured"]; impr = p6["mean_abs_shap_REFH"]; ard = p6["ard_length_scales"]
    mx = max(max(imp.values()), max(impr.values()))
    NM = {"logRe": "log Re", "logWe": "log We", "logD": "log D₀", "phi": "φ (smooth fraction)", "texvol": "V_tex"}
    bars = ""
    for k in sorted(imp, key=imp.get, reverse=True):
        bars += (f'<div class="bar"><span class="bl">{NM.get(k, k)}</span>'
                 f'<span class="bt"><i style="width:{imp[k]/mx*100:.1f}%"></i><em style="width:{impr[k]/mx*100:.1f}%"></em></span>'
                 f'<span class="bv">{imp[k]:.4f} / {impr[k]:.4f}</span><span class="ba">ℓ = {ard[k]:.2f}</span></div>')
    sec6 = (f'<div class="bars">{bars}</div><p class="note">Mean |SHAP| on log β<sub>max</sub>. Solid bar = textured surfaces, thin bar = REF-H. '
            f'ℓ = ARD length-scale in standardised units (shorter = more influential). {esc(p6["note"])}.</p>')
else:
    sec6 = pending("SHAP runs last (KernelExplainer on the GPR, ~5–10 min).")

sec910 = extra.get("api_html") or pending("Next after Phase 7: FastAPI service for the final model and a 5-page local dashboard.")
sec11 = extra.get("pitch_html") or pending("Last: 30 s / 1 min / 3 min pitch scripts and poster changes.")

# code excerpts
def code(path, pattern=None, maxl=60):
    try: src = open(path).read()
    except Exception: return ""
    if pattern:
        m = re.search(pattern, src, re.S); src = m.group(0) if m else src
    lines = src.splitlines()[:maxl]
    return esc("\n".join(lines))


CODE = [
    ("Protocol + model ladder", f"{ROOT}/src/final_pipeline.py", r'""".*?"""', 30),
    ("Phase 3–4: paired bootstrap vs M2", f"{ROOT}/src/final_pipeline.py", r"def grouped_boot_diff.*?return \[.*?\n", 20),
    ("Phase 5: conformal intervals", f"{ROOT}/src/final_pipeline.py", r"def p5.*?out\[k\]\[\"cov_surface_conformal_REFH\"\].*?\n", 40),
    ("Phase 6: SHAP", f"{ROOT}/src/final_pipeline.py", r"def p6.*?json.dump", 25),
    ("Backbones (phases.py)", f"{ROOT}/src/phases.py", r"class PowerBackbone.*?def __call__\(self, d\): return self._f\(d, self.A\)", 30),
]
code_html = "".join(f'<details><summary>{esc(t)} <span class="path">{esc(os.path.relpath(p, ROOT))}</span></summary><pre><code>{code(p, pat, n)}</code></pre></details>'
                    for t, p, pat, n in CODE)
try: log = esc("\n".join(open(f"{ROOT}/logs/final_pipeline.log").read().splitlines()[-14:]))
except Exception: log = ""

now = time.strftime("%d %b %Y, %H:%M", time.gmtime(time.time() + 5.5 * 3600))
tl = "".join(f'<li class="s-{s}"><span class="d">{esc(d)}</span><span class="p">{esc(p)}</span>'
             f'<span class="pill {s}">{LBL[s]}</span><span class="w">{esc(w)}</span></li>' for d, p, s, w in PH)

SECTIONS = [
    ("p2", "Phase 2 · Baselines M1–M5", "done", "Already complete. XGBoost/MLP ladder and the two pure-physics laws, same three protocols.", sec2),
    ("p34", "Phase 3–4 · Physics residual M5/M6 vs M2", PH[2][2], "Does putting a physics law underneath the learner reduce REF-H error relative to M2? Differences come with a condition-grouped 95% CI.", sec34),
    ("p5", "Phase 5 · Uncertainty + OOD", PH[3][2], "Are the 90% intervals honest on an unseen textured surface, and on the smooth plate? Does the OOD score warn when error is high?", sec5),
    ("p67", "Phase 6–7 · SHAP + benchmark", PH[4][2], "What the final model relies on, and the full table the pick is made from.", sec6 + sec7),
    ("p910", "Phase 9–10 · API + dashboard", PH[5][2], "Serving the chosen model locally.", sec910),
    ("p11", "Phase 11 · Pitch + poster tie-in", PH[6][2], "Scripts at three lengths and the poster changes the new results require.", sec11),
]
sec_html = "".join(f'<section id="{i}"><header><h2>{esc(t)}</h2><span class="pill {s}">{LBL[s]}</span></header>'
                   f'<p class="q">{esc(q)}</p>{body}</section>' for i, t, s, q, body in SECTIONS)

tpl = open(f"{ROOT}/live/template.html").read()
out = (tpl.replace("%%NOW%%", now).replace("%%TIMELINE%%", tl).replace("%%SECTIONS%%", sec_html)
          .replace("%%CODE%%", code_html).replace("%%LOG%%", log).replace("%%NDONE%%", str(n_done))
          .replace("%%NTOTAL%%", str(len(PH))))
open(f"{ROOT}/live/index.html", "w").write(out)
print("built", now, n_done, "/", len(PH))
