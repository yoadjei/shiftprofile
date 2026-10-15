# Metric Provenance

Every metric used in this study is established. The study introduces no new
metric; its novelty is the joint protocol. This file records where each one
comes from, which implementation is authoritative here, and what is known to
go wrong with it.

| Metric | Defining work | Implementation | Known pitfalls |
|---|---|---|---|
| Brier score | Brier 1950 | `metrics/calibration.py` | Multiclass form is 2x sklearn's binary form; pinned by `test_library_agreement.py` |
| NLL | standard | `metrics/calibration.py` | Infinite for a zero-probability true class; clipped at 1e-12 |
| ECE (equal-mass, 15 bins) | Naeini et al. 2015; Nixon et al. 2019 for equal-mass | `metrics/calibration.py` | Bin-dependent; biased upward at small n; blind to ranking. Never reported alone |
| Debiased ECE | Kumar, Liang & Ma 2019 | `metrics/calibration.py` | Squared-ECE bias correction reported on the ECE scale; floored at zero |
| AURC | Geifman & El-Yaniv | `metrics/calibration.py` | Sensitive to ties in confidence; stable sort used |
| Temperature scaling | Guo et al. 2017 | `metrics/calibration.py` | MUST be fitted on held-out in-distribution data only. Warns when the fitted value is pinned at a search bound, which means the optimum lies outside the range and the fit failed |
| Effective robustness | Taori et al. 2020 | `metrics/robustness.py` | Probit-space linear fit needs >= 2 reference models; fragile with few |
| Removal-based faithfulness | Samek et al.; ROAD (Rong et al. 2022) | `metrics/faithfulness.py`, masking in `curves.py` | Masked inputs are off-manifold, and worse under shift. **The pitfall in its final form, measured under `curves-v3`:** the metric is sound as a *comparison against a geometry-matched control* and invalid as a *level*. A per-pixel-i.i.d. control does not hold mask geometry fixed, so the level it produces carries a mask-shape artefact reaching **0.20** — a model-free coarse blob (`random_lowres_2`) reads −0.19932 under `black`, larger than any explainer effect measured here, and the level's sign flips against the matched comparison in three of four imputation schemes. Report explainer minus a control of the same geometry (`<name>_rolled`, `random_lowres_<grid>`); never report the raw level. Second pitfall, fixed: the mask selected the LOWEST-magnitude pixels until `curves-v3` (see the retraction below), and the i.i.d. control could not detect it, because for an i.i.d. map the bottom *k* pixels are as random as the top *k*. Pin the selection direction with a test against a map whose maximum and minimum are in known places |
| Attribution stability | Alvarez-Melis & Jaakkola 2018 | `metrics/stability.py` | Confounded by changed predictions; conditioned on unchanged prediction, with the conditioning count reported beside every value |
| Worst-group accuracy | Sagawa et al. 2020 | `metrics/subgroup.py` | Meaningless below ~200 rows per group; enforced by refusal, not by convention |
| Equal-opportunity gap | Hardt et al. 2016 | `metrics/subgroup.py` | Conflicts with per-group calibration when base rates differ (Kleinberg et al.; Pleiss et al. 2017) |

## Documented discrepancies

- **Brier, ours vs sklearn.** sklearn's `brier_score_loss` is the binary
  single-class MSE. The multiclass sum-of-squares form used here is exactly
  twice that on binary problems. Pinned in `test_library_agreement.py`.
- **Faithfulness across libraries.** The same removal-based faithfulness metric
  is known to return different values in AIX360 and Quantus. This study uses one
  implementation throughout and reports it as such; cross-library agreement is
  explicitly NOT claimed.
- **ECE bin sensitivity.** Measured on calibrated synthetic data (n=5000):

  | bins | 5 | 10 | 15 | 30 |
  |---|---|---|---|---|
  | ECE | 0.01444 | 0.0161 | 0.01587 | 0.02345 |

  The E6 ablation reports this sensitivity on real data.

- **Normalised zero is the dataset mean, not black.** Found twice in this
  codebase, in two modules, under two names, each time creating an arm that
  silently duplicated the one it was meant to contrast with:

  | Site | Spelled | Actually was | Now |
  |---|---|---|---|
  | `curves.impute` | `zero` | `mean` | refused; `black` added at −mean/std |
  | `explain.integrated_gradients` | `black` | `mean` | `mean`; a real `black` added |

  The imputation case was caught because `mean` and `zero` agreed to five
  decimals on all 120 curves of the four-arm E6 sweep, which two distinct
  schemes cannot do by chance; the IG baseline was then found by looking for the
  same shape elsewhere. The constant now lives in one function,
  `data.cifar.normalised_black`, and both call sites read it from there. Pinned
  by tests asserting the two schemes and the two baselines differ — the test the
  original lacked, and without which a duplicate arm is invisible.

  The IG case is the more consequential of the two: attribution is
  `grad × (x − baseline)`, so a pixel at the baseline value scores zero however
  much the model depends on it, and `mean` removal imputation then moves a
  removed pixel to exactly that baseline. Baseline and imputation are not
  independent knobs.

- **RETRACTED 2026-10-09 — the table below was computed with an inverted mask.**
  `curves._get_removal_mask` selected the pixels of *lowest* attribution
  magnitude rather than the highest (0 of 52 selected pixels in the top 52 by
  magnitude; all 52 in the bottom 52), so every curve under `curves-v1` and
  `curves-v2` is the removal curve for deleting the background. The numbers are
  kept here, struck through in prose rather than deleted, because the pattern
  they form is a useful record of what an inverted instrument looks like: the
  ordering tracks where each method's **minimum** sits, not its maximum.
  Grad-CAM's low region is one large contiguous backdrop, so deleting it
  preserved the prediction almost perfectly and it scored worst of all.

  Not to be cited. Superseded by whatever `curves-v3` produces.

  **Removal-based faithfulness is imputation-dependent, measured.** On the pilot
  grid (`resnet18`, seed 0, 1000 images, 3 families × 3 severities + clean):

  | | `blur` | `mean` | `uniform_noise` | `black` |
  |---|---|---|---|---|
  | Grad-CAM | −0.061 | −0.291 | −0.293 | −0.295 |
  | Integrated Gradients | −0.068 | −0.109 | −0.068 | −0.025 |
  | ratio Grad-CAM : IG | **0.91** | 2.66 | 4.31 | **11.95** |
  | `random` explainer | ≤ 9e-5 | ≤ 9e-5 | ≤ 9e-5 | ≤ 9e-5 |

  Two reversals, not one. The **ranking** reverses: Grad-CAM scores better than
  IG under `blur` and twelve times worse under `black`. The **sign of
  degradation** reverses too: for `fog`, the clean-to-severity-5 change is
  −0.0499 under `blur` and +0.0480 under `mean` for Grad-CAM, and −0.0487
  against +0.0350 for IG — both explainers, both beyond their own half-widths,
  same images and same attributions.

  79 of the 80 attribution measurements fail to clear zero, at 2.2 to 25.6
  half-widths below it, while the `random` explainer's own 40 sit at |·| ≤ 9e-5.
  So the implementation is sound and the result is a property of the measurement:
  it ranks Integrated Gradients and Grad-CAM *below* noise. Whether that is mask
  geometry is what the P3b controls (`random_lowres_<k>`, `<name>_rolled`) are
  for; until they report, the geometry account is an inference and is labelled as
  one.

---

- **CORRECTED 2026-10-15 — the `curves-v3` recomputation. These numbers
  supersede the retracted block above.** The grid was rerun with the selection
  direction fixed: 510 units, 0 failures, 7325.1s, `resnet18`, seed 0, 1000
  images, 10 cells (clean + 3 families × 3 severities), 4 imputation schemes,
  10 explainer arms. Full tables in `docs/results.md`, Part 2. Nothing below is
  shared with the block above; it is a different computation on the same images
  and the same attributions.

  **1. The geometry artefact is real and it is the size of the whole effect.**
  `random_lowres_<grid>` is noise on a grid × grid lattice, upsampled the way
  Grad-CAM's 4×4 map is, and never touches the model. It is monotone from grid 2
  to grid 16 under all four schemes:

  | imputation | grid 2 | 4 | 8 | 16 | 32 | span (2→32) |
  |---|---|---|---|---|---|---|
  | `black` | −0.19932 | −0.13929 | −0.07275 | −0.02958 | +0.00531 | 0.2046 |
  | `uniform_noise` | −0.18550 | −0.15432 | −0.11924 | −0.06960 | −0.00027 | 0.1852 |
  | `mean` | −0.15983 | −0.12500 | −0.09769 | −0.07262 | +0.00164 | 0.1615 |
  | `blur` | +0.04398 | +0.04112 | +0.02673 | +0.00974 | +0.01548 | 0.0285 |

  A coarse blob with no attribution content in it reads −0.19932 under `black`.
  The "unfaithfulness" the retracted block reported, and the residual negative
  level the corrected raw metric still reports, are of that order. Mask shape
  alone accounts for them.

  **2. Against a geometry-matched control the metric works.** Explainer minus a
  control of the same mask geometry, positive in 11 of 12 comparisons:

  | imputation | grad_cam − grad_cam_rolled | grad_cam − random_lowres_4 | IG − IG_rolled |
  |---|---|---|---|
  | `mean` | +0.11523 (9.3 half-widths) | +0.10078 (8.7) | +0.07389 (9.1) |
  | `blur` | +0.09648 (8.7) | +0.09209 (8.8) | +0.07100 (7.3) |
  | `black` | +0.10068 (8.6) | +0.09194 (8.5) | −0.00398 (0.6, **unresolved**) |
  | `uniform_noise` | +0.12083 (10.4) | +0.11050 (10.2) | +0.04855 (7.6) |

  Every cell but `black`/IG is 10 of 10 cells beating its matched control. The
  two controls are independently constructed — one keeps Grad-CAM's exact map
  and displaces it, the other never sees the model — and agree to within 0.40 to
  1.17 half-widths across all four schemes. `black`/IG at 0.6 half-widths is
  neither a pass nor a failure and is reported as unresolved.

  **3. The level and the comparison disagree in sign.** Against the per-pixel-
  i.i.d. control, Grad-CAM still reads negative under three of four schemes —
  −0.04735 (`black`), −0.02421 (`mean`), −0.04382 (`uniform_noise`) — and
  positive only under `blur` (+0.13321). Same attributions, same images, same
  removal curves as the matched table above. The raw level is the quantity that
  changes sign with the imputation scheme; the matched comparison does not, and
  no matched comparison is negative beyond its half-width. This is the whole
  content of the "Known pitfalls" entry: a level computed against a control of
  the wrong geometry is not a measurement of faithfulness.

  **4. The matched metric reads aim, not shape.** TEST 1 of the mechanism report
  displaces an attribution map by increasing amounts. Grad-CAM's advantage over
  its own displaced copy grows from −0.04357 at 5.73 px to −0.18904 at 17.90 px
  under `mean` (4.3×) and from −0.03410 to −0.15359 under `blur` (4.5×), against
  half-widths of about 0.0095; IG grows 1.7× under both. A map with the right
  shape but no aim would be penalised flatly by displacement. (The verdict column
  of that log prints "SHRINKS with displacement" on all four rows — a label bug
  that compared signed swing where `gain` is negative in the working case, fixed
  in the code at 77781e8, after the log was written.)

  **Caveat carried.** Single seed: `resnet18`, seed 0, 1000 images. The geometry
  account rests on one model whose Grad-CAM map is 4×4, and whether the artefact
  scales the same way at other attribution resolutions is untested here.
