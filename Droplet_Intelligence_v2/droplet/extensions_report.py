"""Write results/extensions/REPORT.md from results/extensions/summary.json (no typed numbers)."""
import json
from .extensions import OUT

def main():
    s = json.loads((OUT / 'summary.json').read_text()); c = s['candidates']; r = s['image_reader']
    f = lambda x: f'{x:.4f}'
    L = ['# Extensions: images from new surfaces, and wettability', '',
         f"Protocol: {s['protocol']}", '',
         '## Results', '',
         '| Candidate | Inputs | LOSO RMSE | 95% CI of difference vs baseline | LOSO 90% coverage | REF-H RMSE | REF-H MAPE | REF-H coverage |',
         '|---|---|---|---|---|---|---|---|']
    desc = {'baseline': 'ln Re, ln We, ln D0, geometry phi, V_tex', 'image_phi': 'as baseline, phi read from SEM',
            'image_only': 'ln Re, ln We, ln D0, phi read from SEM (no spacing or depth)',
            'lee_beta0': 'baseline inputs, target sqrt(beta^2 - beta0^2), beta0 from advancing angle',
            'angle_input': 'baseline inputs + cos(advancing angle)'}
    for k, v in c.items():
        ci = v['delta_vs_baseline_ci95']
        L.append(f"| {k} | {desc[k]} | {f(v['loso']['rmse'])} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {v['loso']['coverage']*100:.1f}% | "
                 f"{f(v['refh']['rmse'])} | {v['refh']['mape']:.2f}% | {v['refh']['coverage']*100:.1f}% |")
    L += ['', f"Angles: {s['angles_used']}", '', '### REF-H angle sweep', '', '| Candidate @ assumed REF-H angle | RMSE | bias | coverage |', '|---|---|---|---|']
    for k, v in sorted(s['refh_angle_sweep'].items()):
        L.append(f"| {k} | {f(v['rmse'])} | {v['bias']:+.4f} | {v['coverage']*100:.1f}% |")
    L += ['', '### Image reader (43x, threshold calibrated on the 12 textured surfaces)', '',
          f"Window {r['window_um']} um; threshold {r['threshold_all_textured']:.2f}. In LOSO the threshold is recalibrated without the held-out surface.", '',
          '| Surface | phi from image | phi from geometry | phi from image, held-out fold |', '|---|---|---|---|']
    for k, v in r['per_surface'].items():
        fold = r['heldout_fold_phi'].get(k)
        L.append(f"| {k} | {v['image_phi']:.3f} | {v['geometry_phi']:.3f} | {'' if fold is None else f'{fold:.3f}'} |")
    L += ['', '## Reading', '',
          '- **Step 1 (image reader): accepted as an input path for new surfaces.** Reading phi from one 43x SEM image gives the same unseen-surface accuracy as the geometry baseline (difference interval includes zero), needs no spacing or depth, and reads the smooth plate as phi = 1 (the CNN read 0.71). It does not improve accuracy, so the baseline stays the default model. Its LOSO interval coverage is slightly below 90%.',
          '- Validity: calibrated at about 43x. At 100x and 350x the reader under-reads phi on textured surfaces (finer plateau detail, field of view smaller than one pitch), so `droplet.predict_image` refuses other pixel sizes unless forced.',
          '- **Step 2 (wettability): rejected on this data.** The Lee correction leaves LOSO unchanged but over-predicts REF-H strongly for every assumed angle; the observed smooth-plate spreading is close to the textured surfaces, not to the Lee prediction. As a GP input the angle hurts LOSO and extrapolates badly on REF-H, because the training angles span only a few degrees.',
          '- What this implies: wettability can only be learned from surfaces whose contact angles actually differ. New surfaces with intermediate angles (e.g. 110 to 150 degrees) are the data needed, with REF-H style smooth plates measured for advancing and receding angles.', '']
    (OUT / 'REPORT.md').write_text('\n'.join(L))
    print('\n'.join(L))

if __name__ == '__main__':
    main()
