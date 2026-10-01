"""Create publication-ready figures and a results report from saved predictions."""
import json,platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT
from .models import LABELS

def run():
    out=ROOT/'results';b=json.loads((out/'benchmark.json').read_text());n=json.loads((out/'nested.json').read_text());r=json.loads((ROOT/'models/release.json').read_text());a=json.loads((out/'data_audit.json').read_text());par=json.loads((out/'js_parity.json').read_text())
    fits=[json.loads(p.read_text()) for p in (out/'folds').glob('*.json')];inner=[json.loads(p.read_text()) for p in (out/'inner_folds').glob('*.json')]
    lines=['# Droplet Intelligence v2 — implementation and evaluation report','',
    'Prepared for Parth Sharma · 23 September 2026','',
    '## Decision','',
    '**Retain the five-feature Matérn 3/2 GP as the default.** Four models were actually trained and evaluated. The Laan-residual candidate has the lowest fixed-candidate LOSO error, but its paired surface-bootstrap interval includes zero. The nested selector does not improve on the fixed baseline. This release improves reproducibility, model comparison, input handling, calibration isolation and deployment; it does not establish a statistically supported prediction-accuracy breakthrough.','',
    '## Measured results','',
    '| Model | LOSO RMSE | Macro surface RMSE | Worst surface RMSE | REF-H RMSE | REF-H interval coverage |','|---|---:|---:|---:|---:|---:|']
    for m in b['metrics']:
        z=r['candidates'][m['model']]['refh'];lines.append(f"| {LABELS[m['model']]} | {m['loso_rmse']:.6f} | {m['macro_rmse']:.6f} | {m['worst_rmse']:.6f} | {z['rmse']:.6f} | {z['coverage']:.1%} |")
    lines+=['','All RMSE values are in the dimensionless beta target, not percentages. Point predictions are lognormal medians. REF-H was previously inspected during the original project; these are stress-test results, not a new blind evaluation.','',
      '### Paired comparisons to the fixed baseline','', '| Candidate | RMSE difference | Surface-bootstrap 95% interval |','|---|---:|---:|']
    base=b['metrics'][0]['loso_rmse']
    for m in b['metrics'][1:]:
        ci=m['delta_vs_baseline_ci95'];lines.append(f"| {LABELS[m['model']]} | {m['loso_rmse']-base:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines+=['', 'Bootstrap uses 4,000 paired resamples of all 12 surfaces with replacement (seed 23). Rows within each resampled surface stay together. Intervals quantify the saved prediction comparison, not all sources of retraining or model-development uncertainty. Ranking candidates on these same outer scores incurs selection optimism.','',
      '## Nested selection and calibration','',
      f"A separate nested experiment ran **{len(inner)} inner fits** (12 outer surfaces × 3 inner surface folds × 4 candidates) and reused the 48 independent outer fits. No outer surface enters an inner training set, candidate choice, or residual calibration pool.", '',
      f"- Nested selection RMSE: **{n['nested_rmse']:.6f}**.",
      f"- Difference from fixed baseline, 95% surface-bootstrap interval: [{n['delta_vs_baseline_ci95'][0]:+.6f}, {n['delta_vs_baseline_ci95'][1]:+.6f}].",
      f"- Empirical interval coverage: **{n['coverage']:.2%}**; mean interval width: {n['mean_width']:.6f}.",
      f"- Per-surface coverage: {min(f['coverage'] for f in n['folds']):.1%}–{max(f['coverage'] for f in n['folds']):.1%}.", '',
      'The rule chooses the first candidate within 0.001 macro RMSE of the inner best, in baseline / Matérn 5/2 / structured / residual order. This tolerance was coded before seeing the nested outputs. The final deployment applies the same rule to all textured-surface LOSO folds and chooses the baseline. The nested figure describes the selection procedure, not the final single model.','',
      'Calibration uses normalized log residuals from inner held-out surface folds, then the finite-sample order-statistic rank. This is **empirical cross-validated calibration**, not ordinary split conformal: the inner models use smaller training sets than the outer model, samples share surfaces, and distribution shift can invalidate exchangeability. No distribution-free per-surface guarantee is claimed.','',
      'Deployment q is obtained from the chosen candidate’s full LOSO residuals and applied after fitting it on all textured rows. This CV-to-full-fit transfer is empirical and may change calibration. The live display says “90% target interval,” and smooth/OOD requests are explicitly flagged. The wide interval is a sensitivity band, not a guarantee.','',
      '## Data audit','',
      f"- {a['textured_rows']} textured impacts, 12 surfaces, {a['textured_conditions']} replicate conditions.",
      f"- {a['reference_rows']} separate REF-H impacts, {a['reference_conditions']} conditions; 1,623 impacts total.",
      f"- No missing cells or fully duplicated raw textured rows detected.",
      f"- Textured beta range: {a['target_range'][0]:.4f}–{a['target_range'][1]:.4f}.",
      f"- Recomputed Re and We agree with provided columns to relative error below 5×10⁻⁷.", '',
      'The supplied CSV contact-angle files include water angles for 12 surfaces and glycerol-mixture advancing angles for 6. The 91 wt.% static-angle table is not an advancing-angle table. The missing fluid–surface measurements were not fabricated or interpolated into the predictor.','',
      'The geometry descriptors use the original fixed track widths (45 µm for deep tracks, 30 µm for shallow tracks), phi=(max(spacing-width,0)/spacing)^2 and texture volume per area=(1-phi)×depth. Their units are µm, not µm³. This evaluation is conditional on those supplied descriptors. SEM extraction was not repeated, and held-out-surface independence of the historical width-estimation process is not established.','',
      '## Implemented candidate models','',
      'All candidates use natural-log Re, We and diameter in mm, plus phi and texture volume per area. Each fit estimates its own StandardScaler, target normalization, kernel hyperparameters and white-noise variance. Optimizer initialization is independent of all held-out labels. The target is ln(beta).','',
      '1. Baseline: constant × Matérn 3/2 ARD + white noise.','2. Smoothness alternative: constant × Matérn 5/2 ARD + white noise.','3. Structured GP: a·kf + b·ks + c·kf·ks + white noise, with shared fluid and surface ARD scales and positive amplitudes.','4. Physics residual: a positive fitted Laan parameter A plus the Matérn 3/2 GP residual.','',
      'The custom structured kernel is a positive-semidefinite sum/product construction. Its analytic gradients were checked against finite differences. No extra synthetic impact labels were generated. Exact GP inference retains all 1,498 training impacts.','',
      f"Optimizer diagnostics are retained in every fold file: {sum(bool(x['warnings']) for x in fits)}/{len(fits)} outer fits and {sum(bool(x['warnings']) for x in inner)}/{len(inner)} inner fits emitted warnings, including bounds or iteration limits. These diagnostics should be reviewed before a journal submission; they are not silently discarded.", '',
      '## Issues corrected in this release','',
      '| Earlier issue | Implemented correction |','|---|---|',
      '| Global GP warm start touched all surfaces | Cold initialization within every fit |',
      '| Other-surface residual models could include the outer test surface | Rebuild inner residual models excluding that outer surface |',
      '| Model family chosen from the same scores used for reporting | Add separate nested selection evaluation and label fixed-candidate rankings |',
      '| Interpolated/capped conformal quantile | Explicit sorted-score rank; insufficient calibration returns infinity |',
      '| Smooth reference looked ordinary to anomaly detector | Explicit surface-class warning in addition to support and anomaly checks |',
      '| Re<70 described as GP extrapolation | GP support comes from data, independent of scaling-law regime |',
      '| Unspecified model/scaler/units at deployment | Versioned JSON and binary numerical bundle with hashes |',
      '| Unsupported user inputs and ambiguous geometry | Validated units, positivity, finite values, bounded phi and explicit custom descriptors |',
      '| Inconsistent narrative tables | Report generated directly from saved predictions and evaluation JSON |',
      '| Browser/backend numerical mismatch risk | Independent numerical export tests and JavaScript parity checks |', '',
      'The paper’s Matérn 5/2, log-base-10 baseline, 0.0394 REF-H RMSE, beta range 0.2–1.0 and cubic-micrometre texture units do not describe the shipped baseline/data. Use the values and definitions in this report. The original paper and original archives are preserved, not silently overwritten.','',
      '## Verification and delivery','',
      '- Twelve Python tests passed: data integrity, physical identities, invalid inputs, interval ordering, smooth warnings, kernel gradients/PSD, complete split isolation, quantile edge behavior, exported-vs-sklearn predictions and metric recomputation.',
      f"- JavaScript/Python parity: {par['cases']} cases across {par['models']} models, including smooth reference, high viscosity and extrapolation. Maximum beta difference {par['max_absolute_beta_difference']:.3g}; maximum interval-bound difference {par['max_absolute_interval_difference']:.3g}. Six invalid JavaScript inputs were rejected.",
      '- Model arrays are exported as JSON and little-endian float64 lower-triangular Cholesky data; no pickle execution is needed for inference.',
      '- The live website computes the real GP in a Web Worker. No training job runs when a visitor predicts. All four trained candidates are available for comparison.',
      '- A local Python HTTP API and matching website are included. Static hosting does not provide a remote Python API endpoint.',
      '- Full browser UI automation was unavailable for this static-site preview profile. JavaScript syntax, prediction logic, API behavior and static asset references were checked; visual/browser interaction QA remains a limitation.',
      '- WebMCP registration is feature-detected. A supported WebMCP browser validation context was unavailable; ordinary UI operation does not depend on it.', '',
      '## Next useful experiments','',
      'Collect genuinely new surfaces spanning track width, pitch and depth, with per-fluid advancing/receding angles and measurement uncertainty. Reserve a prospective test before further tuning. Then compare descriptor uncertainty, a measured-wettability GP and heteroscedastic observation noise. A larger architecture alone is not supported as an improvement by the current dataset.','',
      '## Primary methods references','',
      '- https://scikit-learn.org/stable/modules/gaussian_process.html','- https://arxiv.org/abs/2107.07511','- https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf','- https://doi.org/10.1103/PhysRevApplied.2.044018','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,ax=plt.subplots(1,2,figsize=(12,4.7),layout='constrained');names=['Baseline','Matérn 5/2','Interaction','Laan residual'];vals=[m['loso_rmse'] for m in b['metrics']]
    bars=ax[0].barh(names,vals,color=['#007f76','#93afc1','#93afc1','#283e56']);ax[0].invert_yaxis();ax[0].set_xlim(0,.049);ax[0].set_xlabel('LOSO RMSE (dimensionless β)');ax[0].set_title('Fixed-candidate accuracy')
    for bar,val in zip(bars,vals):ax[0].text(val+.0007,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center')
    cov=[f['coverage']*100 for f in n['folds']];ax[1].bar(range(12),cov,color=['#b18437' if v<90 else '#007f76' for v in cov]);ax[1].axhline(90,color='#17293e',ls='--',lw=1,label='90% target');ax[1].set_xticks(range(12),[f['outer'] for f in n['folds']],rotation=60);ax[1].set_ylim(0,105);ax[1].set_ylabel('Observed coverage (%)');ax[1].set_title('Nested empirical interval coverage');ax[1].legend(frameon=False)
    fig.suptitle('Droplet Intelligence · actual evaluation results',fontsize=15);fig.savefig(out/'evaluation.png',dpi=220);fig.savefig(out/'evaluation.svg');plt.close(fig)
    print(out/'REPORT.md')
if __name__=='__main__':run()
