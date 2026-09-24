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
| Removal-based faithfulness | Samek et al.; ROAD (Rong et al. 2022) | `metrics/faithfulness.py` | Masked inputs are off-manifold, and worse under shift. Never reported raw — always relative to a random-attribution control measured in the same cell |
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
