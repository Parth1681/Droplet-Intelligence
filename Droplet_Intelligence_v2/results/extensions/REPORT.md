# Extensions: images from new surfaces, and wettability

Protocol: LOSO on 12 textured surfaces; paired surface bootstrap 4000 draws seed 23 vs baseline; REF-H predicted after training on all textured surfaces, not used for any choice.

## Results

| Candidate | Inputs | LOSO RMSE | 95% CI of difference vs baseline | LOSO 90% coverage | REF-H RMSE | REF-H MAPE | REF-H coverage |
|---|---|---|---|---|---|---|---|
| baseline | ln Re, ln We, ln D0, geometry phi, V_tex | 0.0400 | [+0.0000, +0.0000] | 90.1% | 0.0591 | 1.94% | 61.6% |
| image_phi | as baseline, phi read from SEM | 0.0398 | [-0.0057, +0.0047] | 88.0% | 0.0595 | 1.95% | 66.4% |
| image_only | ln Re, ln We, ln D0, phi read from SEM (no spacing or depth) | 0.0397 | [-0.0068, +0.0045] | 88.8% | 0.0636 | 2.09% | 61.6% |
| lee_beta0 | baseline inputs, target sqrt(beta^2 - beta0^2), beta0 from advancing angle | 0.0398 | [-0.0013, +0.0009] | 90.1% | 0.2279 | 11.49% | 0.8% |
| angle_input | baseline inputs + cos(advancing angle) | 0.0480 | [+0.0004, +0.0171] | 89.2% | 0.3771 | 15.33% | 100.0% |

Angles: Water advancing angle per surface (file 03); REF-H assumed 114.5 deg (literature), swept 100 to 130.

### REF-H angle sweep

| Candidate @ assumed REF-H angle | RMSE | bias | coverage |
|---|---|---|---|
| angle_input@100 | 0.3891 | -0.0177 | 100.0% |
| angle_input@114.5 | 0.3771 | -0.0477 | 100.0% |
| angle_input@130 | 0.3364 | -0.1155 | 100.0% |
| lee_beta0@100 | 0.3060 | +0.2981 | 0.0% |
| lee_beta0@110 | 0.2516 | +0.2430 | 0.8% |
| lee_beta0@114.5 | 0.2279 | +0.2188 | 0.8% |
| lee_beta0@120 | 0.1997 | +0.1898 | 2.4% |
| lee_beta0@130 | 0.1512 | +0.1390 | 8.0% |

### Image reader (43x, threshold calibrated on the 12 textured surfaces)

Window 12.7 um; threshold 12.32. In LOSO the threshold is recalibrated without the held-out surface.

| Surface | phi from image | phi from geometry | phi from image, held-out fold |
|---|---|---|---|
| D100 | 0.170 | 0.303 | 0.146 |
| D200 | 0.482 | 0.601 | 0.471 |
| D400 | 0.780 | 0.788 | 0.784 |
| D50 | 0.217 | 0.010 | 0.185 |
| D600 | 0.776 | 0.856 | 0.772 |
| D800 | 0.883 | 0.891 | 0.887 |
| S100 | 0.528 | 0.490 | 0.539 |
| S200 | 0.650 | 0.722 | 0.664 |
| S400 | 0.806 | 0.856 | 0.814 |
| S50 | 0.471 | 0.160 | 0.483 |
| S600 | 0.882 | 0.902 | 0.888 |
| S800 | 0.861 | 0.926 | 0.863 |
| REF-H | 1.000 | 1.000 |  |

## Reading

- **Step 1 (image reader): accepted as an input path for new surfaces.** Reading phi from one 43x SEM image gives the same unseen-surface accuracy as the geometry baseline (difference interval includes zero), needs no spacing or depth, and reads the smooth plate as phi = 1 (the CNN read 0.71). It does not improve accuracy, so the baseline stays the default model. Its LOSO interval coverage is slightly below 90%.
- Validity: calibrated at about 43x. At 100x and 350x the reader under-reads phi on textured surfaces (finer plateau detail, field of view smaller than one pitch), so `droplet.predict_image` refuses other pixel sizes unless forced.
- **Step 2 (wettability): rejected on this data.** The Lee correction leaves LOSO unchanged but over-predicts REF-H strongly for every assumed angle; the observed smooth-plate spreading is close to the textured surfaces, not to the Lee prediction. As a GP input the angle hurts LOSO and extrapolates badly on REF-H, because the training angles span only a few degrees.
- What this implies: wettability can only be learned from surfaces whose contact angles actually differ. New surfaces with intermediate angles (e.g. 110 to 150 degrees) are the data needed, with REF-H style smooth plates measured for advancing and receding angles.
