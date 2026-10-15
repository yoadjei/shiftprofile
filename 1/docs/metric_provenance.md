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
| Removal-based faithfulness | Samek et al.; ROAD (Rong et al. 2022) | `metrics/faithfulness.py`, masking in `curves.py` | Masked inputs are off-manifold, and worse under shift. Never reported raw — always relative to a random-attribution control measured in the same cell. **Two pitfalls found here, both by reading code rather than results.** (1) That control is drawn per-pixel i.i.d., so it does not hold mask geometry fixed, and — worse — it is insensitive to the direction of the ranking, so it cannot detect an inverted mask. (2) The mask selected the LOWEST-magnitude pixels until `curves-v3`; see the retraction below. Pin the selection direction with a test against a map whose maximum and minimum are in known places |
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
