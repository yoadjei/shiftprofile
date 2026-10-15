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

---

| Date | Section | Change | Reason | Data seen at time of amendment |
|---|---|---|---|---|
| 2026-10-09 | §9, entries A2 and A3 | **Retracted.** The faithfulness measurements they rest on were computed with an inverted removal mask | A7 | As A7 |

**A7 — the instrument was inverted, and A2 and A3 are withdrawn.**
`curves._get_removal_mask` selected the pixels of **lowest** attribution magnitude rather than the
highest. It built `-|attr| + position` and took `topk(largest=True)`, and the largest value of
`-|attr|` is the smallest `|attr|`. Measured on continuous data with no ties: 0 of 52 selected pixels
were in the top 52 by magnitude and all 52 were in the bottom 52. Every removal curve written under
`curves-v1` and `curves-v2` is the curve for deleting the background, so every faithfulness number
this document has recorded answers a question nobody asked.

Withdrawn with it:

- **A2's two reversals.** The ranking reversal (Grad-CAM 0.91× IG under `blur`, 11.95× under `black`)
  and the sign reversal on `fog` are both artefacts. Grad-CAM's *low* region is a large contiguous
  backdrop, so deleting it preserved the prediction almost perfectly and it scored worst; Integrated
  Gradients' low region is scattered, so it scored less badly. The ordering was a fact about where
  each method's **minimum** sits.
- **A3's "reliably negative at 2.2 to 25.6 half-widths".** Removing unimportant pixels barely moves
  the probability, so `model_auc` stayed high and `random_auc − model_auc` was negative by
  construction. The sign of the entire result was the defect.
- Everything the P3b run reported, including the geometry ladder, the geometry-matched comparison and
  the "anti-faithfulness" of Grad-CAM. §8 rows 4 and 5 are therefore **not** established as having
  fired, and the pivot they justified is unjustified until the grid is recomputed.

**A4 stands** as an arithmetic statement about how half-widths scale, but the ratios it quotes come
from the same curves and are not to be reused. **A5 and A6 stand in full**: both misnamings are
properties of the code, found by reading it, and neither depends on a curve.

**What the controls could not catch, which is the methodological lesson.** All three of §6's validity
controls are random-attribution controls. For an i.i.d. map the bottom *k* pixels are as random as the
top *k* — the single case an inverted comparison leaves untouched. So the control read ≈0 exactly as a
working control should, under all four imputation schemes, while every real explainer read negative.
The instrument appeared sound and the explainers appeared to fail, and the agreement of the control
with its own expectation was read as evidence that the machinery was trustworthy. **A control that
cannot fail in the same way as the thing it controls is not a control for that failure.** §6 is
amended to say so, and the function now has the five tests it never had.

**Status of the study.** Unaffected: the calibration track, which is computed from predictions and
touches no curve, and whose clean-to-severity-5 ladder stands. Void pending recomputation: every
faithfulness, gate and rank result. The §2 revision remains unwritten, which in hindsight is the one
procedural decision in this sequence that paid for itself.

---

| Date | Section | Change | Reason | Data seen at time of amendment |
|---|---|---|---|---|
| 2026-10-15 | §9, entry A7 | A7's "void pending recomputation" is **resolved**. The grid was recomputed under `curves-v3`; which withdrawals are reinstated and which stand is recorded below | A8 | Full P3b validity grid under `curves-v3`: 510 units, 0 failed, 7325.1s |
| 2026-10-15 | §4, row 1 (primary faithfulness metric) | Recorded as **not executable as written**: the pair it chooses between was never built. An amendment is recommended and awaits the author; none is made here | A8 | As above |
| 2026-10-15 | §4, row 3 (attribution sample size) | Recorded as **not decidable** from the present gate. Left at 1000 by default, as undecided rather than chosen | A8 | As above |

**A8 — the grid was recomputed, and A7's "void" resolves three ways, not one.**
The inverted-mask fix shipped as `curves-v3` and the whole P3b validity grid was rerun: 510 units,
0 failures, 7325.1s. `resnet18`, seed 0, 1000 images, 10 cells (clean + 3 families × 3 severities),
4 imputation schemes, 10 explainer arms. A7 declared every faithfulness, gate and rank result "void
pending recomputation". Some of it is reinstated in corrected form, some is reinstated only as an
observation with its interpretation withdrawn, and some stays withdrawn.

**Reinstated in corrected form: the geometry ladder.** `random_lowres_<grid>` is model-free by
construction — noise on a grid × grid lattice, upsampled the way Grad-CAM's 4×4 map is. It is
monotone from grid 2 to grid 16 under all four schemes, and the grid-2-to-grid-32 span is 0.2046
(`black`), 0.1852 (`uniform_noise`), 0.1615 (`mean`) and 0.0285 (`blur`). A model-free coarse blob
reads **−0.19932** under `black`. That is the full magnitude of the apparent unfaithfulness the raw
metric reports for real explainers, produced by mask shape alone with no model input at any point.

**Reinstated in corrected form: the geometry-matched comparison, which is now the working
instrument.** An explainer minus a control of the same mask geometry is positive in 11 of 12
comparisons:

| imputation | grad_cam − grad_cam_rolled | grad_cam − random_lowres_4 | IG − IG_rolled |
|---|---|---|---|
| `mean` | +0.11523 (9.3 half-widths) | +0.10078 (8.7) | +0.07389 (9.1) |
| `blur` | +0.09648 (8.7) | +0.09209 (8.8) | +0.07100 (7.3) |
| `black` | +0.10068 (8.6) | +0.09194 (8.5) | −0.00398 (0.6, **unresolved**) |
| `uniform_noise` | +0.12083 (10.4) | +0.11050 (10.2) | +0.04855 (7.6) |

Every cell except `black`/IG is 10 of 10 cells beating its matched control. The two controls are
independent in construction — one keeps Grad-CAM's exact map and displaces it, the other never sees
the model — and they agree to within 0.40 to 1.17 half-widths across all four schemes. The `black`/IG
row is the one unresolved comparison: at 0.6 half-widths it is neither a pass nor a failure, and it is
reported as unresolved rather than rounded to either.

**Reinstated as an observation, withdrawn as a finding: A3.** Against the pre-registered
per-pixel-i.i.d. control, Grad-CAM still reads **negative** under three of four schemes — −0.04735
(`black`), −0.02421 (`mean`), −0.04382 (`uniform_noise`) — and positive only under `blur`
(+0.13321). So the raw reading that prompted A3 does reproduce for Grad-CAM, and under `black` and
`uniform_noise` it is negative beyond its own half-width at every severity. A3's **quantities** do
not reproduce: it reported 79 of 80 attribution measurements failing to clear zero at 2.2 to 25.6
half-widths, where the corrected run puts 36 of those 80 cells above zero. And A3's **reading** is
withdrawn outright. It took the negative sign to be a property of the explainers. It is a property of
the control: a per-pixel-i.i.d. control does not hold mask geometry fixed, and matching the control on
geometry reverses the sign in three of four schemes on the same attributions and the same images.

**Still withdrawn: A2's ranking reversal, and the "anti-faithfulness" of Grad-CAM.** Grad-CAM beats
both of its matched controls under all four schemes, at 8.5 to 10.4 half-widths; there is no
anti-faithfulness to explain. A2's ratio table (Grad-CAM : IG at 0.91 under `blur` rising to 11.95
under `black`) does not reproduce and cannot even be restated on the corrected data: the two
explainers' raw means have opposite signs under two of the four schemes, so the ratio is not defined.
On the matched metric there is no ranking reversal at all — Grad-CAM's margin over its own geometry
control exceeds IG's margin over its own under every scheme.

**A2's imputation-dependence survives, but it is now a different finding.** As written, A2 said the
imputation scheme reverses the *ranking of the explainers* and the *sign of degradation with
severity*. Both of those are artefacts of the inverted mask and stay withdrawn. What survives is
narrower and sharper: the scheme reverses the sign of the **raw** metric (Grad-CAM, +0.13321 under
`blur` against negative under the other three), while the **matched** metric agrees in sign across
all four schemes and no matched comparison is negative beyond its half-width. The axis on which the
conclusion flips is therefore the **control**, not the imputation and not the explainer ranking.
Imputation still matters — it sets the size of the geometry artefact, 0.2046 under `black` against
0.0285 under `blur` — but it is no longer where the conclusion changes.

**§8 rows 4 and 5 fire again, on corrected data.** Row 4, *"faithfulness intervals exceed the
effect"*: the P3 gate passes 14 of 85 resolvable pairs. Row 5, *"conclusions flip with faithfulness
metric choice"*: the sign flips between the raw and the matched metric in three of four schemes.
The control is part of the metric's definition — §6's first validity control says faithfulness is
*always* reported relative to a control — so a change of control is a change of metric. Both rows
were registered in advance, which is what §8 was for.

**Independent confirmation that the matched instrument reads aim, not shape.** TEST 1 of the
mechanism report displaces an attribution map by increasing amounts and asks whether displacing it
further costs more. Grad-CAM's advantage over its own displaced copy grows from −0.04357 at 5.73 px
to −0.18904 at 17.90 px under `mean` (4.3×) and from −0.03410 to −0.15359 under `blur` (4.5×),
against half-widths of about 0.0095. IG grows 1.7× under both. A map that merely had the right
*shape* would be penalised flatly by displacement; this one is penalised in proportion to how far it
is moved, which is a statement about where it points. (Caveat for anyone reading that log: its verdict
column prints "SHRINKS with displacement" on all four rows, which is the opposite of the measurement.
It compared signed swing, and `gain` is negative precisely in the working case. The label is fixed in
the code at 77781e8; the log predates the fix.)

**§4 row 1 cannot be executed as written.** It names a choice between a "ROAD-style" and a
"single-deletion" faithfulness metric, with a decision rule over the pair. Only one faithfulness
metric exists in the code: a removal-curve AUC. `metrics/faithfulness.py` exposes `rank_features`,
`removal_auc` and `relative_faithfulness`, and no ROAD or single-deletion implementation exists
anywhere in the tree. The pair was never built, so the rule has nothing to choose between, and §4's
closing sentence — "the secondary metric in each pair is still computed and reported in the E6
measurement-validity ablation" — describes an artefact that does not exist.

**Recommendation on §4 row 1 — requiring the author's approval, not a decision taken here.** Replace
the axis. Make the pre-registered choice the **control** — per-pixel-i.i.d. versus geometry-matched —
rather than two removal metrics that were never implemented. The recommendation is made because the
data decide that axis decisively where they cannot decide the registered one: the per-pixel-i.i.d.
control admits a model-free artefact of up to 0.2046, which is larger than any explainer effect
measured here, while the geometry-matched control separates explainer from control at 8.5 to 10.4
half-widths in 11 of 12 comparisons. This amendment does **not** make that change, and the
distinction matters. §4 licenses *applying* its rules to pilot data and recording the outcome by
amendment; it does not license *replacing* a rule with a different one chosen after seeing the data
that would decide it, which is the post-hoc move §4 exists to prevent. No Version A cell has been
filled (A1), so nothing is being rewritten under full-grid data — but the rule change still has to be
an explicit, author-approved act on the record, not an amendment that looks like an application. The
choice between amending §4, implementing the missing metric, and deleting the row belongs to the
author. It is recorded here so the gap is on the record whichever way it goes.

**§4 row 3 is not decidable from this gate, and is not decided.** The P3 gate fails at 14 of 85
resolvable pairs, and the composition of the 14 is the reason it cannot carry a sample-size decision:

| what passes | count |
|---|---|
| model-free `random_lowres_*` arms | 9 |
| `grad_cam_rolled` — a real map, displaced off its target | 3 |
| a correctly-aimed explainer (`black`/grad_cam, `mean`/grad_cam) | 2 |

All 14 passes are `gaussian_noise`; `blur` passes nothing. The gate is computed on the **raw** metric,
against the per-pixel-i.i.d. control that A8 has just shown to be the defective one. A gate whose
passes are 12/14 arms carrying no correctly-aimed attribution is reporting how sharply a corruption
moves the removal curve, not how precisely a faithfulness claim can be made. §4's 1000-vs-2000 rule is
therefore recorded as **undecided**: the sample size stays at 1000 by default, not by the rule. A
matched-metric gate is being added separately; the decision waits on it. A4 stands unchanged as the
arithmetic of how half-widths scale, and its ratios, which came from the voided curves, remain
unusable.

**Calibration is unchanged, as it has been across every run.** Accuracy 0.8610 → 0.5487, Brier
0.2166 → 0.6810, ECE_em 0.0516 → 0.2514, ECE_db 0.0764 → 0.2650, AURC 0.0315 → 0.2702, over
severities 0, 1, 3, 5. It is computed from predictions and reads no curve, which is why A7 did not
touch it and why it is the one track to have come through two instrument defects intact.

**§2 is still not rewritten, and that is deliberate.** A7 called holding it the one procedural
decision in the sequence that paid for itself, and the reasons to hold it have not expired: the §4
metric decision above is a recommendation awaiting the author, and the matched-metric gate has not
reported. Revising the hypotheses before both land would be exactly the post-hoc move this document
exists to prevent. §2 is held, not forgotten.

**Single-seed caveat, unchanged and applying to everything above.** All of it is `resnet18`, seed 0,
1000 images: one architecture, one seed, one dataset. Nothing here is a claim about across-seed or
across-model variation, and §6's mixed-effects machinery has not been exercised on any of it. The
geometry account rests on a single model's 4×4 Grad-CAM resolution, and whether the artefact scales
the same way at other resolutions is untested.
