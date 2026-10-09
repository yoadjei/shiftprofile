# Pre-Registration — Reliability Degradation Under Distribution Shift

**Status:** DRAFT. Not binding until committed with a git tag `prereg-frozen` and no full-grid data has
been observed.
**Date drafted:** 2026-09-22
**Gate:** P2. Depends on the P1 novelty gate, which passed; that assessment is held internally.
**Protocol:** the committed configs under `configs/`, the fixed evaluation indices, and the pipeline in
`shiftprofile/` are the operative specification. Where this document and the code disagree, this document
governs the analysis and the code is the bug.

---

## 0. What this document is for

Everything in §2–§7 is confirmatory: fixed before the full grid is run, tested as stated, reported whether
or not it comes out favourably. Anything not in this document that appears in the paper is exploratory and
will be labelled as such in the text, not merely in an appendix.

The point is not ceremony. The study's central risk (audit §31, attack 20 — "engineering, not research")
is answered only by having committed to a falsifiable prediction before seeing the data.

---

## 1. Distinction this study rests on

Two things are called "perturbation" in this literature and they are not the same axis:

- **Explanation perturbation** — how much an input is masked in order to *compute* or *smooth* an
  attribution. Varied by Zhang et al. (arXiv 2506.19630) and by arXiv 2511.10439.
- **Input distribution shift** — how far the *data* has drifted from the training distribution, graded by
  corruption severity or domain distance. **This study varies only this.**

Every hypothesis below concerns input distribution shift. No claim is made about explanation perturbation
intensity.

---

## 2. Primary hypothesis

**H3.** As input distribution shift severity increases, attribution faithfulness degrades faster than
accuracy, and the severity at which faithfulness collapses is predicted by in-distribution calibration.

Split into two independently testable parts. Both are primary; H3a passing and H3b failing is a reportable
result, not a failure of the study.

### H3a — differential degradation rate

*Faithfulness declines with severity at a steeper normalised rate than accuracy.*

- **Outcome:** `relative_faithfulness` = (removal AUC of a random-attribution control) − (removal AUC of
  the explainer), per cell. Normalised to its clean-cell value within (model, seed).
- **Comparator:** accuracy, normalised the same way within the same cell.
- **Test:** mixed-effects model
  `normalised_metric ~ severity * metric_type + C(model) + C(shift_family) + (1 | seed)`,
  where `metric_type ∈ {faithfulness, accuracy}`. H3a predicts a **negative** coefficient on the
  `severity × metric_type[faithfulness]` interaction.
- **Decision rule:** supported if that interaction coefficient's 95% CI excludes zero in the predicted
  direction after Holm correction within this research question.

### H3b — calibration predicts faithfulness onset

*In-distribution calibration predicts the severity at which an explainer stops beating its random control.*

- **Outcome (onset severity):** for each (model, seed, shift family), the smallest severity at which the
  bootstrap 95% CI of `relative_faithfulness` includes zero. If no severity qualifies, the observation is
  **right-censored** at the maximum severity run. Censoring is expected and is handled, not discarded.
- **Predictors:** in-distribution ECE and in-distribution AURC, measured on the clean cell of the same
  (model, seed).
- **Test:** Cox proportional-hazards model over onset severity with the two predictors, validated by
  **leave-one-shift-family-out** — fit on four families, predict the fifth.
- **Decision rule:** supported if the out-of-fold concordance index exceeds 0.6 and its bootstrap CI
  excludes 0.5. A model that cannot beat chance out-of-fold is reported as not predictive, regardless of
  its in-sample fit.
- **Insufficient-events rule:** if fewer than 10 uncensored onset events are observed across the whole
  grid, H3b is declared untestable at this scale and reported as such. It is not rescued by loosening the
  onset definition after the fact.

---

## 3. Secondary hypotheses

Confirmatory, but subordinate. Holm correction applied within this family, separately from H3.

- **S1 — residual coupling.** After regressing out accuracy change, faithfulness change and calibration
  change remain correlated across cells within model. Tests whether "everything just tracks accuracy"
  (audit §31, attack 18).
- **S2 — stability vs faithfulness.** Attribution stability and faithfulness do not degrade at the same
  rate; they measure different things and may trade off.
- **S3 — intervention transfer.** Temperature scaling (fitted in-distribution only) and AugMix change
  faithfulness or stability. Paired comparison, Holm-corrected.
- **S4 — subgroup gaps (tabular, secondary).** Per-group ECE gap widens faster with domain-shift distance
  than per-group accuracy gap. **Declared secondary in advance** per requirement R5 of the P1 gate, so
  that running the tabular track in parallel is not read as scope creep.

---

## 4. Parameters the pilot resolves — rules, not outcomes

Three parameters cannot be fixed before the P3 pilot. What is pre-registered is the **decision rule**, so
the choice cannot be made to suit the results. Each is decided on pilot data only, recorded in this file
by amendment, and frozen before any full-grid cell is filled.

| Parameter | Decision rule | Decided on |
|---|---|---|
| Primary faithfulness metric (ROAD-style vs single-deletion) | Whichever has the narrower bootstrap interval half-width relative to its clean→severity-5 change on pilot data. Tie → single-deletion, as the cheaper. | Pilot cells only |
| ViT-Tiny input resolution (112 / 160 / 224) | Lowest resolution whose clean accuracy falls within 1 point of the ResNet-18 baseline (E0). Escalate if none does. | Pilot cells only |
| Attribution sample size (1000 vs 2000 images) | 1000 if the faithfulness interval half-width is under 10% of the clean→severity-5 change; otherwise 2000. | Pilot cells only |

The secondary metric in each pair is still computed and reported in the E6 measurement-validity ablation.
Choosing the primary does not discard the alternative.

---

## 5. Metric collapse rule

Fixed in advance to prevent post-hoc metric selection:

> Any pair of metrics with |Spearman ρ| > 0.9 across all pilot cells collapses to one. The retained metric
> is the cheaper to compute; the dropped one moves to the appendix with its correlation stated.

Applied once, to pilot data, before the full run. Not re-applied afterwards.

---

## 6. Analysis plan

- **Unit of analysis:** the cell — (track, model_id, seed, shift_family, severity). Correlations are
  computed over cells **within** a model. Correlating across the three models is prohibited: with three
  points there is no correlation to report (audit §31, attack 15).
- **Intervals:** paired bootstrap over evaluation samples within seed, 2000 resamples, on the committed
  fixed indices. Hierarchical bootstrap for headline numbers. Across-seed variation reported separately.
- **Non-independence:** seed enters as a random effect. Cells within a seed are not independent and are
  never treated as such (audit §31, attack 16).
- **Corrections:** Holm within each research question — H3 family and secondary family corrected
  separately.
- **Effect sizes:** standardised differences and slopes with intervals, reported alongside every p-value.
- **Rank stability:** Kendall tau across seeds, explainers and faithfulness metrics, with bootstrap rank
  intervals.

### Estimator reporting (added 2026-09-23, before any data)

The mixed model is fitted by an ordered ladder, REML first, falling through
optimisers on singular-matrix failures and only then to ML. This is not a
detail: REML gives unbiased variance components and ML biases them downward,
worst with few groups — which is this study's regime at five seeds. On a
synthetic 5-seed grid, ML reported a seed variance of 0.000572 against REML's
0.000714, about 20% lower.

Accordingly:

- **The estimator that actually ran is reported for every fitted model**, in the
  results table, not a footnote. Metrics fitted by different estimators are not
  comparable across rows, and the ladder makes that possible without saying so.
- **A fit that falls through to ML is flagged**, and any variance component it
  produced is reported with that caveat attached.
- **A fit whose seed variance collapses to the boundary is not reported as a
  result.** It converged in name only — all variance went to the residual and
  none to the grouping. Both conditions emit warnings in code.

**Precision caveat, found on synthetic data before any real run.** When a large
unmodelled factor drives the metric — exactly the decoupling this study exists
to detect — no REML variant converged, and the accuracy coefficient moved from
its true 0.5 to 0.74 with a residual of 0.29. Diagnostically that draw had a
chance correlation between the unmodelled trend and accuracy change, which the
large factor then amplified. The lesson is not that the model is wrong but that
**the accuracy coefficient loses precision precisely when decoupling is strong**.
The confidence interval on that coefficient is therefore reported always, and
the residual-after-accuracy result is never read from a point estimate alone.

### Validity controls (enforced in code, not by discipline)

1. Faithfulness is **always** reported relative to a random-attribution control measured in the same cell.
   `relative_faithfulness()` raises if the control is absent.
2. The imputation scheme is a required argument with no default, recorded in every result record.
3. Stability is computed **only** on unchanged-prediction samples, with the conditioning subset size
   reported beside every value.

These are pinned by three named tests. Deleting or weakening any of them is a protocol change requiring an
amendment to this document, not a code change.

---

## 7. Scope: primary versus secondary cells

- **Primary (Version A).** 3 vision models × 5 seeds × (1 clean + 5 corruptions × 3 severities) = 240
  cells. All confirmatory tests above are computed on this grid.
- **Pre-specified secondary (Version B).** Extension to all 15 corruptions × 5 severities, CIFAR-10.1,
  Waterbirds, and the ACS tabular track. Declared here **before** any extension cell is filled, so that
  accumulated compute quota strengthens the paper without turning into an unregistered garden of forking
  paths. Reported as confirmatory-secondary, distinguished from primary in every table.

Compute accrues at roughly 30 GPU-hours per week. Which extension cells get filled is a resource outcome,
not an analytic choice, and will be reported as a completion table showing exactly which cells exist.
Cells that were not run are stated as such; silent truncation is treated as a defect.

---

## 8. Declared outcomes, including the unwelcome ones

Registered in advance so that no result requires reframing (audit §27):

| Outcome | What is reported |
|---|---|
| H3a and H3b both supported | The headline: faithfulness degrades faster than accuracy, and cheap in-distribution calibration forecasts its collapse |
| H3a supported, H3b not | Decoupling is real but not predictable from calibration. A clean null against the two small positive studies, with the prediction attempt reported in full |
| Neither supported; all metrics track accuracy | "Accuracy on the line" extends to calibration and explanations under corruption. Useful: cheap accuracy monitoring suffices. The reduced metric set is the contribution |
| Faithfulness intervals exceed the effect | Documented negative result: current faithfulness metrics cannot support claims under shift. Framing 3 of audit §40 |
| Conclusions flip with faithfulness metric choice | Rank instability becomes the headline, with bootstrap rank intervals |
| Fewer than 10 uncensored onset events | H3b declared untestable at this scale, reported with the observed event count |

**What would falsify the study's premise:** if `relative_faithfulness` is indistinguishable from zero at
*every* severity including clean, the explainers carry no measurable signal over random on these models
and no degradation claim is possible. This is checked at E1, before the grid is filled, and would trigger
a pivot rather than a null result.

---

## 9. Amendment log

Amendments are appended here with date and reason, never by editing the text above. An amendment made
after full-grid data is observed converts every affected test to exploratory.

| Date | Section | Change | Reason | Data seen at time of amendment |
|---|---|---|---|---|
| 2026-10-09 | §0, status header | Record that no `prereg-frozen` tag has ever been cut, so §2–§7 are not yet binding | A1 | Pilot and E6 sweeps only; no Version A cell filled |
| 2026-10-09 | §8 | Record that rows 4 and 5 of the declared-outcomes table both fired | A2 | As above |
| 2026-10-09 | §8, falsifier | Observed condition is stronger than the registered one: reliably negative, not indistinguishable from zero | A3 | As above |
| 2026-10-09 | §4, attribution sample size | Escalation rule 1000 → 2000 recorded with its measured consequence; reported, not taken | A4 | As above |
| 2026-10-09 | §6, validity control 2 | Imputation scheme `zero` retired: it duplicated `mean` exactly, so the four-arm ablation varied three arms | A5 | Four-arm E6 sweep, 120 curves |
| 2026-10-09 | §3 / E6 | IG baseline `black` renamed `mean`; a real `black` added. The axis was also unreachable from any config | A6 | As above |

**No amendment above changes a hypothesis, an outcome definition, a test or a decision rule.** Each
records a fact about the protocol or the instrument. The §2 hypothesis revision the pilot points to is
deliberately **not** made here: the control that decides it (P3b) has not yet run, and amending the
hypotheses before seeing it would be the post-hoc move this document exists to prevent.

### Notes on amendments

**A1 — the protocol was never frozen.** §0 makes §2–§7 confirmatory "before the full grid is run",
and binding only once committed under a `prereg-frozen` tag. `git tag -l` is empty and no Version A
cell has been filled. So nothing observed so far is full-grid data, and the pilot has been doing
exactly the job §4 assigns it: resolving parameters that could not be fixed in advance. This is
recorded because the distinction decides whether the work below is an amendment or a draft revision,
and the answer should not depend on anyone's memory.

**A2 — which declared outcomes fired.** Row 4, *"faithfulness intervals exceed the effect"*: the P3
gate passes 3 of 22 resolvable pairs. Row 5, *"conclusions flip with faithfulness metric choice"*:
the imputation scheme is part of the metric's definition, and it reverses both the ranking and the
sign.

- **Ranking.** Grad-CAM against Integrated Gradients, as a ratio of mean faithfulness: 0.91 under
  `blur` (Grad-CAM better), 2.66 under `mean`, 4.31 under `uniform_noise`, 11.95 under `black`.
- **Sign.** For `fog`, the clean-to-severity-5 change is −0.04988 under `blur` and +0.04796 under
  `mean` for Grad-CAM, and −0.04866 against +0.03502 for IG. Both explainers, both beyond their own
  half-widths, same images and same attributions. This fails E6's own agreement criterion
  (*"degradation direction agrees across 2 metrics and 2 imputations"*).

Both rows were registered in advance, so neither outcome requires reframing — which is what §8 was
for.

**A3 — the falsifier as written was too weak.** §8 declares the premise falsified if
`relative_faithfulness` is *indistinguishable from zero* at every severity. What was observed is
different and stronger: reliably **negative**. Counting only the measurements that involve an actual
attribution method — 2 explainers × 4 schemes × 10 cells = 80 — 79 fail to clear zero, and the
per-severity aggregates sit between 2.2 and 25.6 half-widths below it. The `random` explainer's own
40 measurements sit at |·| ≤ 9e-5 under all four schemes, so the instrument is working and is not
merely insensitive: it ranks structured attributions *below* noise.
"No signal" and "signal of the wrong sign" are different findings and the registered text admits only
the first. Recorded rather than rewritten.

**A4 — the sample-size escalation, and why it is reported instead of taken.** §4's rule is: 1000
images if the half-width is under 10% of the clean-to-severity-5 change, otherwise 2000. The gate
fails, so the rule permits 2000. Its measured consequence: the half-width falls as 1/√n, so 2000
narrows it by 1.41×, moving 3 of 22 pairs passing to 5 of 22 (the two additions are IG on
`gaussian_noise` under `mean` and under `black`; three further pairs land at 0.101–0.103 and still
fail). The worst pair, `black` / IG / `defocus_blur` at a ratio of 0.990, needs a 9.9× narrowing and
therefore about 98× the images — roughly 98,000, against a CIFAR-10 test set of 10,000. It is not
expensive, it is unreachable on this dataset. Escalating would also leave untouched the two findings
that do not depend on precision at all: that nothing beats its control, and that the sign reverses
with the scheme.

**A5 — `zero` was `mean`.** The scheme filled masked pixels with zero *in normalised space*, and
normalisation maps the dataset mean to zero, so it was mean imputation under a second name. The two
agreed to five decimals across all 120 curves of the sweep, which two distinct schemes cannot do by
chance. E6 reported four arms and varied three, and the duplicate inflated the gate's denominator.
`zero` is now refused rather than aliased, because a cached `zero` curve is a `mean` curve filed
under the wrong name. Replaced by `black`, which is black in pixel space, about −1.99 per channel
once normalised.

**A6 — the IG baseline carried the same misnaming, on an E6 axis.** `integrated_gradients` built its
`"black"` baseline as `torch.zeros_like(x)` — the dataset mean again — so the ablation's
"black vs blur" contrast was "mean vs blur". The baseline is not cosmetic: attribution is
`grad × (x − baseline)`, so a pixel at the baseline value scores zero however much the model depends
on it, and `mean` removal imputation then moves a removed pixel to exactly that baseline. `mean` is
now the frozen protocol's baseline under its true name, `black` means black, and the producer version
was bumped to `explain-v3` so the same spelling cannot resolve to both artifacts.

Two further facts belong on the record. First, the axis was **unreachable**: `explainer_options` has
carried `baseline` in the cache key from the start on the strength of a comment saying E6 varies it,
while the filler forwarded only `ig_steps` from the config, so every run attributed at the default and
the ablation existed only in the key. Second, the conflation was found twice independently, in two
modules, under two names. The constant now lives in one function, `data.cifar.normalised_black`, and
both call sites read it from there.

**Outstanding.** The §2 revision this points to is held until the P3b geometry controls report. The
single-seed caveat applies to everything above: all of it is `resnet18`, seed 0, 1000 images.
