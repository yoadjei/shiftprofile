# Pre-Registration — Reliability Degradation Under Distribution Shift

**Status:** DRAFT. Not binding until committed with a git tag `prereg-frozen` and no full-grid data has
been observed.
**Date drafted:** 2026-09-22
**Last amended:** 2026-10-15 (A9: §2 rewritten onto the geometry-matched metric; §4 row 1 re-axed onto
the control. Read §9 A9 before §2 or §4.)
**Gate:** P2. Depends on the P1 novelty gate, which passed; that assessment is held internally.
**Protocol:** the committed configs under `configs/`, the fixed evaluation indices, and the pipeline in
`shiftprofile/` are the operative specification. Where this document and the code disagree, this document
governs the analysis and the code is the bug.

---

## 0. What this document is for

Everything in §2–§7 is confirmatory: fixed before the full grid is run, tested as stated, reported whether
or not it comes out favourably. Anything not in this document that appears in the paper is exploratory and
will be labelled as such in the text, not merely in an appendix.

**One qualification, added 2026-10-15 by A9.** §2 was rewritten after the P3b pilot and before any
Version A cell was filled. The pilot is not the full grid, and no `prereg-frozen` tag has ever been cut
(A1), so §2 remains confirmatory with respect to Version A. It is **not** confirmatory with respect to
the pilot: every pilot figure §2 quotes is labelled exploratory and generating at the point it is
quoted, so a reader can see exactly what was known when each hypothesis was written. A9 records what
changed and on whose decision.

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

## 2. Primary hypotheses

**Rewritten 2026-10-15 by amendment A9.** From 2026-09-22 to 2026-10-15 this section carried one primary
hypothesis, H3, in two parts, both stated on `relative_faithfulness`. The P3b pilot established that
`relative_faithfulness` is not a valid level: the control it is measured against does not hold mask
geometry fixed, and a hypothesis stated on an invalid outcome cannot be tested whatever the data say.

H3 is **restated, not deleted**. Its structure is unchanged — two independently testable parts, the same
mixed-effects specification, the same Cox model with leave-one-shift-family-out, the same decision rules,
the same insufficient-events rule. Only the outcome variable moves, and each consequence of the move is
stated where it falls.

Two validity hypotheses are added ahead of it and are primary. They are what the pilot actually
establishes, and H3 cannot be read without them: H3 asks the original question with a corrected
instrument, and H1 and H2 are the evidence that the instrument needed correcting.

### Definitions used throughout §2

| Name | Definition |
|---|---|
| `raw_faithfulness` | `removal_auc(control) − removal_auc(explainer)` with the control drawn **per-pixel i.i.d.** This is `metrics.faithfulness.relative_faithfulness`, and it is the outcome this section carried before A9. Invalid as a level; see H1a. |
| `matched_faithfulness` | `raw_faithfulness(explainer) − raw_faithfulness(control)` for a control of the **same mask geometry**. The per-pixel-i.i.d. term is bitwise identical in both halves — same `control_seed`, same images, same imputation, same model — so it cancels exactly, leaving `removal_auc(control) − removal_auc(explainer)` with a control of the explainer's own shape. Computed per image and bootstrapped paired; `pilot_report.matched_rows`. |
| geometry-matched control | Either `<name>_rolled` — the explainer's own map displaced by a seeded per-image offset drawn from the h·w−1 non-identity translations, so the value multiset and the spatial autocorrelation are identical and only registration to the image is destroyed — or `random_lowres_<g>` — noise on a g×g lattice, ReLU'd and bilinearly upsampled through Grad-CAM's exact pipeline, reading nothing from the model, the image or the label. |
| native map resolution | The spatial grid an attribution is computed on before upsampling. ResNet-18's Grad-CAM map is **4×4**: the CIFAR stem preserves 32×32 and `layer2`–`layer4` each halve it. ViT-Tiny at `img_size=112`, patch 16, is **7×7**. Integrated Gradients is pixel-level on both. |

### What is already known, and what that makes it

Everything quoted below from the P3b pilot is **exploratory and hypothesis-generating**. It is one
architecture (`resnet18`), one seed (0), 1000 images, 10 cells, three shift families and four imputation
schemes, run under `curves-v3` on 2026-10-15 — 510 units, 0 failed. It is quoted at each hypothesis so a
reader can see exactly what was known at the moment that hypothesis was written; that is the only thing
that makes the confirmatory/exploratory line checkable rather than asserted. **No hypothesis below is
supported or refuted by it.** Each is tested on Version A (§7), on seeds and architectures the pilot has
not touched. Every pilot figure is checkable against `docs/results.md` Part 2 and
`docs/runs/2026-10-15_curves-v3_pilot_validity.log`.

---

### H1 — the geometry artifact

#### H1a — the artifact is the size of the effect

*A control that does not hold mask geometry fixed admits an apparent faithfulness effect at least as
large as the largest real one.*

- **Outcome:** two quantities per (model, seed, imputation):
  - **A** = max over the model-free `random_lowres_<g>` arms of |mean `raw_faithfulness`|. No model, no
    image and no label enters any of these arms, so whatever they register is mask shape and nothing else.
  - **E** = max over the two explainers and their matched controls of |mean `matched_faithfulness`|.
- **Test:** paired bootstrap over evaluation images of `A − E`, 2000 resamples, seeded, within
  (model, seed, imputation) on the committed fixed indices.
- **Decision rule:** supported for an imputation scheme if the 95% CI of `A − E` excludes zero and is
  positive. Reported **per scheme and as a count of schemes, never pooled**: the pilot shows the
  artifact's size is a strong function of the scheme, and pooling would average a real dependence away.
- **Primary scheme:** `mean`, the one `configs/version_a.yaml` runs. Any further scheme is reported
  alongside it, not substituted for it.
- **What would falsify it:** `A − E` negative, or indistinguishable from zero, in the primary scheme.
  Then the per-pixel-i.i.d. control admits no artifact large enough to matter and the pivot this document
  makes is unwarranted.

**Pilot (exploratory, seed 0).** The model-free ladder, each value a mean over the pilot's 10 cells:

| imputation | grid 2 | 4 | 8 | 16 | 32 | span (2→32) |
|---|---|---|---|---|---|---|
| `black` | −0.19932 | −0.13929 | −0.07275 | −0.02958 | +0.00531 | 0.2046 |
| `uniform_noise` | −0.18550 | −0.15432 | −0.11924 | −0.06960 | −0.00027 | 0.1852 |
| `mean` | −0.15983 | −0.12500 | −0.09769 | −0.07262 | +0.00164 | 0.1615 |
| `blur` | +0.04398 | +0.04112 | +0.02673 | +0.00974 | +0.01548 | 0.0285 |

Monotone toward zero from grid 2 to grid 16 under all four schemes; grid 32 breaks monotonicity only
under `blur`. Against the largest matched explainer effect in the same scheme:

| imputation | A (model-free) | E (largest matched explainer effect) | A − E |
|---|---|---|---|
| `black` | 0.19932 | 0.10068 | **+0.0986** |
| `uniform_noise` | 0.18550 | 0.12083 | **+0.0647** |
| `mean` | 0.15983 | 0.11523 | **+0.0446** |
| `blur` | 0.04398 | 0.09648 | −0.0525 |

Three of four schemes in the predicted direction and `blur` against it. The headline claim that the
artifact is larger than any explainer effect in the study is true **globally** — 0.19932 against a largest
matched effect of 0.12083 and a largest raw explainer magnitude of 0.13321 — and is **not** true scheme by
scheme. Under `blur` the artifact is roughly half the effect. Stated here rather than left for a reader to
discover, because H1a is registered per scheme precisely so that this cannot be averaged out of sight.

#### H1b — the artifact scales with the attribution's native resolution

*The geometry account makes a prediction that an "attributions are simply unfaithful under shift" account
does not: the artifact is a function of how coarse the mask is, so a model whose attribution map is finer
carries a smaller artifact.*

- **Prediction, stated before the data exist.** ResNet-18's Grad-CAM map is 4×4. ViT-Tiny at
  `img_size=112`, patch 16, is 7×7 — 49 tokens against 16. The geometry account predicts the artifact
  measured on ViT-Tiny is **smaller in magnitude** than on ResNet-18, by about what the measured
  grid-4-to-grid-8 segment of the ladder spans. The rival account predicts **no relationship with
  resolution at all**.
- **Outcome:** |mean `raw_faithfulness`| of the model-free arm at each model's own native resolution,
  compared across models within (seed, imputation).
- **Test:** paired bootstrap over evaluation images of the difference between the two models' artifacts,
  within (seed, imputation); reported per seed, aggregated across seeds by §6's mixed-effects model.
- **Decision rule:** the geometry account is supported if that difference's 95% CI excludes zero in the
  predicted direction. It is **falsified** if the CI excludes zero in the opposite direction, or if it
  includes zero at a resolution that could have detected the predicted difference. A null at a resolution
  too coarse to detect the predicted size is reported as *underpowered*, not as evidence for either
  account — which is why the power statement below is written now and not afterwards.
- **Power, stated in advance.** The ladder's own grid-4-to-grid-8 step is the size of the effect this
  predicts, and in the pilot that step is clearly resolved under two schemes and marginal under two:

  | imputation | grid 4 | grid 8 | step | larger per-arm half-width | step / half-width |
  |---|---|---|---|---|---|
  | `black` | −0.13929 | −0.07275 | 0.06654 | 0.01969 | 3.4 |
  | `uniform_noise` | −0.15432 | −0.11924 | 0.03508 | 0.01226 | 2.9 |
  | `mean` | −0.12500 | −0.09769 | 0.02731 | 0.01876 | 1.5 |
  | `blur` | +0.04112 | +0.02673 | 0.01439 | 0.01106 | 1.3 |

  Under `mean` alone — the only scheme `configs/version_a.yaml` currently runs — this prediction is tested
  at about 1.5 per-arm half-widths and could fail to resolve for want of precision rather than for want of
  an effect. The half-widths quoted are each arm's own, not the half-width of the paired difference, which
  is tighter and is the interval the test actually uses; the paired statistic is not computed by the
  present report (see the implementation gap below).
- **Genuinely untested, and currently unrunnable.** `vit_tiny` has never been trained or run. Two
  blockers, both verified in the code for this amendment: `models.vision.gradcam_target_layer` returns
  `model.layer4[-1]` behind a `hasattr(model, "layer4")` check and otherwise raises, so `grad_cam` and
  `grad_cam_rolled` cannot run on a ViT at all; and `data.cifar.to_normalised_tensor` converts
  `(N, 32, 32, 3)` uint8 to `(N, 3, 32, 32)` and never resizes, while `vit_tiny` is built from
  `timm`'s `vit_tiny_patch16_224` at `img_size=112`. 80 of Version A's 240 cells are ViT-Tiny's and none
  of them can run. §4 row 2 is the decision that unblocks them and it is unresolved.
- **Implementation gap, recorded before it is hit.** `explain.LOWRES_GRIDS` is `(2, 4, 8, 16, 32)`. There
  is no `random_lowres_7` arm, so ViT-Tiny's native resolution has no exact model-free match on the
  existing ladder. Either 7 is added to the grid set, or the prediction is read against the
  grid-4-to-grid-8 interpolation above. The choice is made before any ViT cell is filled and recorded by
  amendment. Separately, `pilot_report.matched_pairs` constructs only explainer-versus-control pairs, so
  the grid-to-grid paired difference this test needs does not yet exist in code.

---

### H2 — the pre-registered control is blind to the defect it exists to catch

Two parts. H2a is a prediction about Version A. H2b is not a prediction: it is a property of the control,
with a demonstration this project has already paid for, and it is registered as a requirement on the
instrument rather than as something to be tested.

#### H2a — raw and matched disagree in sign

*Changing only the control reverses the sign of the faithfulness conclusion, on the same images, the same
attributions and the same removal curves.*

- **Outcome:** per (model, seed, imputation, explainer), the signs of mean `raw_faithfulness` and mean
  `matched_faithfulness`.
- **Counted as a disagreement** only when both readings exclude zero in their own 95% bootstrap CI and the
  signs differ. A raw reading that cannot clear zero is not a sign, and is reported as unresolved.
- **Decision rule:** supported if at least one (imputation, explainer) cell disagrees, per that
  definition, in a majority of the (model, seed) pairs run. Reported as a count out of the cells run, with
  the full per-cell table, never as a proportion alone.
- **Second, structural rule — the one that discriminates between accounts.** A generic "the control
  matters" account predicts disagreements scattered across explainers. The geometry account of H1b
  predicts them **concentrated in the explainer with the coarser native map**. H2a is recorded as
  *strongly* supported only if every disagreeing cell belongs to the coarser-mapped explainer, and the
  distribution of disagreements across explainers is reported whatever it is.
- **Why primary and not a footnote.** §6's first validity control requires that faithfulness is *always*
  reported relative to a control. The control is therefore part of the metric's definition, and a sign
  that depends on it is §8 row 5 — *conclusions flip with faithfulness metric choice* — firing on the
  control axis.

**Pilot (exploratory, seed 0).** 3 of 8 (imputation, explainer) cells disagree, and all three are Grad-CAM:

| imputation | explainer | raw | matched (vs own `_rolled`) | signs |
|---|---|---|---|---|
| `black` | grad_cam | −0.04735 | +0.10068 | **disagree** |
| `mean` | grad_cam | −0.02421 | +0.11523 | **disagree** |
| `uniform_noise` | grad_cam | −0.04382 | +0.12083 | **disagree** |
| `blur` | grad_cam | +0.13321 | +0.09648 | agree |
| `black` | integrated_gradients | −0.04214 | −0.00398 | agree (matched unresolved at 0.6 half-widths) |
| `mean` | integrated_gradients | +0.02605 | +0.07389 | agree |
| `uniform_noise` | integrated_gradients | +0.00203 | +0.04855 | agree |
| `blur` | integrated_gradients | +0.07246 | +0.07100 | agree |

Resolution of the raw end of the three disagreements, as multiples of the mean of that arm's per-cell
half-widths: `uniform_noise` 5.2, `black` 2.4, `mean` 1.4. Under `mean` the raw end is the weakest of the
three and is not resolved at every severity individually — the per-cell values at severities 3 and 5 are
−0.00829 ± 0.01779 and −0.00867 ± 0.01677. The matched end is resolved in all three, at 8.6 to 10.4
half-widths. The concentration in Grad-CAM is what H1b independently predicts and is the reason the
structural rule above is registered.

#### H2b — the per-pixel-i.i.d. control cannot move under the defect it controls for

For a per-pixel-i.i.d. attribution map the bottom *k* pixels are as random as the top *k*. Inverting the
attribution ranking permutes which control pixels are removed and leaves the distribution of the result
untouched. The control's reading is therefore **invariant to inversion of the ranking** — the single most
likely implementation error a removal metric admits.

This is not a thought experiment. This project ran its entire pilot with `curves._get_removal_mask`
selecting the **lowest**-magnitude pixels: measured on continuous data with no ties, 0 of 52 selected
pixels were in the top 52 by magnitude and all 52 were in the bottom 52. **All three of §6's
pre-registered validity controls passed throughout**, because all three are random-attribution controls.
The control read ≈0 before the fix (|·| ≤ 9e-5, A7) and ≈0 after it (per-scheme means +0.00002 to
+0.00007 under `curves-v3`, largest single cell +0.00017). A control whose reading cannot move under the
defect it exists to catch is not a validity control for that defect. A7 records the retraction;
`tasks/lessons.md` L5 records the rule and its cost.

**What this registers.** At least one control in every faithfulness comparison must be **asymmetric under
the metric's own ordering** — it must have a top and a bottom, so that inverting the selection moves it.
`<name>_rolled` and `random_lowres_<g>` are; the per-pixel-i.i.d. control is not. This is a fourth
validity control and belongs in §6. It is **recommended here and not made here**, on the same reasoning
A8 applied to §4 row 1: the author approved a change to §2 and §4, and a §6 protocol change is not
smuggled in alongside it. A9 lists it as outstanding before a freeze.

---

### H3 — the original question, asked with the corrected instrument

**H3.** As input distribution shift severity increases, attribution faithfulness degrades faster than
accuracy, and the severity at which faithfulness collapses is predicted by in-distribution calibration.

Unchanged in substance from the 2026-09-22 text. Split into the same two independently testable parts.
Both are primary; H3a passing and H3b failing is a reportable result, not a failure of the study.

#### H3a — differential degradation rate

*Faithfulness declines with severity at a steeper normalised rate than accuracy.*

- **Outcome:** `matched_faithfulness`, normalised to its clean-cell value within
  (model, seed, imputation, explainer, control).
  **Changed from `relative_faithfulness`** (A9). Justification: the registered outcome is a level computed
  against a control of the wrong geometry, and the pilot's model-free ladder puts the artifact in that
  level at up to 0.19932 — larger than any explainer effect the study has measured. A degradation *rate*
  read off that level is a rate at which mask geometry's contribution changes with severity, which is not
  what H3a claims. Nothing else about H3a changes.
- **Comparator:** accuracy, normalised the same way within the same cell. **Unchanged.**
- **Test:** mixed-effects model
  `normalised_metric ~ severity * metric_type + C(model) + C(shift_family) + (1 | seed)`,
  where `metric_type ∈ {faithfulness, accuracy}`. **Unchanged.** H3a predicts a **negative** coefficient
  on the `severity × metric_type[faithfulness]` interaction.
- **Decision rule:** supported if that interaction coefficient's 95% CI excludes zero in the predicted
  direction after Holm correction within this research question. **Unchanged.**
- **New refusal, forced by the new outcome.** Normalising to the clean cell is undefined when the clean
  `matched_faithfulness` is within its own bootstrap half-width of zero: the ratio is then a ratio to
  noise and its variance is unbounded. Any (model, seed, imputation, explainer, control) whose clean cell
  fails that check is **excluded from H3a and reported as excluded, with its clean value and
  half-width** — not rescued by a floor, a shrinkage constant or a change of normaliser after the fact.
  The pilot's one candidate is `black` / integrated_gradients against `integrated_gradients_rolled`:
  −0.00398 over 10 cells at 0.6 half-widths, with 1 of 10 cells beating its control. Whether its *clean
  cell specifically* fails the check is not determined by the pilot output, which reports the 10-cell
  mean.
- **Each control reported separately.** Grad-CAM has two matched controls, `grad_cam_rolled` and
  `random_lowres_4`, built independently — one keeps Grad-CAM's exact map and displaces it, the other
  never sees the model. The fit is run once per (explainer, control) pair and every pair is reported. A
  disagreement between a method's two controls is a result about the controls and is reported as one.
  On the pilot the two agree to within 0.40 to 1.17 half-widths across all four schemes.

#### H3b — calibration predicts faithfulness onset

*In-distribution calibration predicts the severity at which an explainer stops beating a control of its
own geometry.*

- **Outcome (onset severity):** for each (model, seed, shift family, explainer, control), the smallest
  severity at which the bootstrap 95% CI of `matched_faithfulness` includes zero. If no severity
  qualifies, the observation is **right-censored** at the maximum severity run. Censoring is expected and
  is handled, not discarded.
  **Changed from `relative_faithfulness`** (A9). Justification: the onset as registered was the severity
  at which an explainer stopped beating a *per-pixel-i.i.d.* control, and a model-free blob drives that
  quantity to −0.19932 without seeing a model, an image or a label. The registered onset was therefore a
  statement about the severity at which mask geometry overwhelms the comparison. That is a property of the
  mask, not of the explainer, and it is not what H3b claims to predict.
- **Predictors:** in-distribution ECE and in-distribution AURC, measured on the clean cell of the same
  (model, seed). **Unchanged** — the calibration track is the one track neither instrument defect touched.
- **Test:** Cox proportional-hazards model over onset severity with the two predictors, validated by
  **leave-one-shift-family-out** — fit on four families, predict the fifth. **Unchanged.**
- **Decision rule:** supported if the out-of-fold concordance index exceeds 0.6 and its bootstrap CI
  excludes 0.5. A model that cannot beat chance out-of-fold is reported as not predictive, regardless of
  its in-sample fit. **Unchanged.**
- **Insufficient-events rule:** if fewer than 10 uncensored onset events are observed across the whole
  grid, H3b is declared untestable at this scale and reported as such. It is not rescued by loosening the
  onset definition after the fact. **Unchanged.**
- **Expected to be censored, said in advance.** On the pilot, Grad-CAM beats both of its matched controls
  in 10 of 10 cells under all four schemes, and Integrated Gradients beats its own in 10 of 10 under
  `blur`, `mean` and `uniform_noise`. No severity's matched CI includes zero in any of those arms, so
  every pilot observation is right-censored and the pilot yields **zero uncensored onset events**. If
  Version A behaves like the pilot, the insufficient-events rule fires and H3b is declared untestable at
  this scale. That is registered here as the expected outcome, so that an H3b result, should one appear,
  cannot be read as having been fished for — and so that its absence is a pre-registered finding rather
  than a silence.

---

### Status of each hypothesis

| Hypothesis | Status for Version A | Pilot evidence |
|---|---|---|
| H1a — artifact size | Confirmatory | Exploratory, generating. Seed 0: A − E positive in 3 of 4 schemes, negative under `blur` |
| H1b — artifact scales with native resolution | Confirmatory, and **currently unrunnable** — `vit_tiny` is blocked | None. No ViT cell has ever been computed. The prediction rests on the ResNet-18 ladder alone |
| H2a — raw and matched disagree in sign | Confirmatory | Exploratory, generating. Seed 0: 3 of 8 cells, all Grad-CAM |
| H2b — the i.i.d. control cannot move under inversion | Not a hypothesis. A property of the control, demonstrated by a defect that occurred (A7, L5) | Established, not predicted |
| H3a — differential degradation rate | Confirmatory, outcome changed by A9 | Not measured. §6's mixed model has been exercised on synthetic data only |
| H3b — calibration predicts onset | Confirmatory, outcome changed by A9 | Exploratory: 0 uncensored events on seed 0 under every scheme |

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

**Row 1's axis was replaced on 2026-10-15 by amendment A9, after the pilot, by an explicit decision of
the author.** The rule it previously carried chose between two faithfulness metrics, only one of which
was ever implemented, so it could not be executed on any data and never could have been. A9 states what
was replaced, why, and why it is recorded as a rule change rather than as an application of the rule that
stood. Read A9 before reading row 1. The superseded text is preserved there verbatim.

| Parameter | Decision rule | Decided on |
|---|---|---|
| **Faithfulness control** — per-pixel i.i.d. vs geometry-matched *(re-axed by A9)* | A control is **admissible** only if an arm that reads nothing from the model, the image or the label — constructed at the explainer's own native map resolution — reads within its own bootstrap half-width of zero against it. Among admissible controls the primary is the one whose half-width is the smaller fraction of its clean→severity-5 change. If neither is admissible, the removal metric is reported as invalid as a level and no primary is named. | Pilot cells only |
| ViT-Tiny input resolution (112 / 160 / 224) | Lowest resolution whose clean accuracy falls within 1 point of the ResNet-18 baseline (E0). Escalate if none does. | Pilot cells only |
| Attribution sample size (1000 vs 2000 images) | 1000 if the faithfulness interval half-width is under 10% of the clean→severity-5 change; otherwise 2000. Read on the **matched** metric — `pilot_report.gate_verdicts(value_key="matched")` — not the raw one, per A8 and A9. | Pilot cells only |

The secondary option in each pair is still computed and reported in the E6 measurement-validity ablation.
Choosing the primary does not discard the alternative. On row 1 this is load-bearing rather than
decorative: the raw-versus-matched contrast H2a tests is the difference of the two readings and needs
both, which is why `configs/version_a.yaml` carries the pre-registered `random` arm alongside the matched
controls.

### Resolutions as of 2026-10-15

| Row | Status | Recorded by |
|---|---|---|
| 1 — faithfulness control | **Resolved: geometry-matched.** Against the per-pixel-i.i.d. control, `random_lowres_4` — a model-free arm at Grad-CAM's own 4×4 resolution — reads −0.13929 (`black`), −0.15432 (`uniform_noise`), −0.12500 (`mean`) and +0.04112 (`blur`), which is 3.7 to 12.6 times its own half-width from zero in every scheme. The per-pixel-i.i.d. control is inadmissible. The geometry-matched control is primary; the per-pixel-i.i.d. reading continues to be computed and reported. **Caveat:** the geometry-matched control's own admissibility check is trivially satisfied for `random_lowres_4` against itself and has **not** been run against an independently seeded draw at the same grid. That non-trivial check is listed in A9 as outstanding. | A9 |
| 2 — ViT-Tiny input resolution | **Unresolved, and now blocking a primary prediction.** No ViT-Tiny cell has ever been computed, so the rule's input — clean accuracy at 112 / 160 / 224 — does not exist. Two code-level blockers, verified in A9: `gradcam_target_layer` raises for any architecture without `layer4`, and `to_normalised_tensor` emits 32×32 while `vit_tiny` is built at `img_size=112`. 80 of Version A's 240 cells cannot run, and H1b — the geometry account's sharpest falsifiable prediction — cannot be tested at all until this row is decided. | A8, A9 |
| 3 — attribution sample size | **Undecided. 1000 by default, not by the rule.** The raw-metric P3 gate passes 14 of 85 resolvable pairs and 12 of those 14 passes carry no correctly-aimed attribution, so it cannot carry a sample-size decision. The matched-metric gate now exists — `pilot_report.gate_verdicts(value_key="matched")`, printed as `P3 GATE (GEOMETRY-MATCHED)` — and **has not been run**: it does not appear in `docs/runs/2026-10-15_curves-v3_pilot_validity.log`. The decision waits on running it. A4 stands as the arithmetic of how half-widths scale; its ratios came from the voided curves and remain unusable. | A8, A9 |

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

---

| Date | Section | Change | Reason | Data seen at time of amendment |
|---|---|---|---|---|
| 2026-10-15 | §4, row 1 | **Axis replaced.** The pre-registered parameter is now the faithfulness **control** (per-pixel i.i.d. vs geometry-matched), not the faithfulness metric. Resolved in favour of geometry-matched | A9 | Full P3b validity grid under `curves-v3`, `resnet18` seed 0 only. No Version A cell filled |
| 2026-10-15 | §2, whole section | **Rewritten.** H3a and H3b restated on `matched_faithfulness`; validity hypotheses H1a, H1b, H2a added as primary and H2b added as a requirement on the instrument | A9 | As above |
| 2026-10-15 | §4, row 2 | Recorded as **unresolved and now blocking a primary prediction** (H1b) and 80 of Version A's 240 cells | A9 | As above |
| 2026-10-15 | §4, row 3 | Unchanged from A8 — **undecided**. Records that the matched-metric gate now exists in code and has not been run | A9 | As above |
| 2026-10-15 | §0, status header | Pointer to A9 added, and §0's confirmatory claim qualified with respect to the pilot | A9 | As above |

**A9 — §4 row 1's axis is replaced by decision, and §2 is rewritten on the result.**

A8 recommended re-axing §4 row 1 and explicitly declined to take the decision. **The author has now
approved it.** This amendment executes it. Four things have to be said plainly, because the value of this
log depends on their being said rather than inferred.

**1. This is a rule being replaced after the data that would decide it were seen.** §4 licenses
*applying* its rules to pilot data and recording the outcome by amendment. It does not license
*replacing* a rule with a different one chosen after seeing the pilot, which is the post-hoc move §4
exists to prevent. That is what is being done here. Calling it anything else would make this document
worth less than no document.

**2. It is an explicit, author-approved act, and it is not presented as an application of the rule that
stood.** Row 1 as written chose between two *faithfulness metrics*. Row 1 as it now stands chooses
between two *controls*. Those are different questions and no reading of the old rule produces the new
one. The superseded text, preserved verbatim:

```
| Primary faithfulness metric (ROAD-style vs single-deletion) | Whichever has the narrower bootstrap
interval half-width relative to its clean→severity-5 change on pilot data. Tie → single-deletion, as the
cheaper. | Pilot cells only |
```

**3. No Version A cell has been filled.** A1 established that `git tag -l` is empty, that no
`prereg-frozen` tag has ever been cut, and that §2–§7 are therefore not yet binding. Nothing here is a
rewrite under full-grid data. That is what makes the change *permissible*; it is not what makes it
*legitimate*, which is point 2.

**4. The justification is that the registered rule was unexecutable from the day it was written — not
that the author preferred its replacement's outcome.** Only one faithfulness metric has ever existed in
this tree: a removal-curve AUC. `metrics/faithfulness.py` exposes `rank_features`, `removal_auc` and
`relative_faithfulness`; there is no ROAD implementation and no single-deletion implementation anywhere
in it. The rule had nothing to choose between on 2026-09-22 and nothing to choose between on 2026-10-15,
and no quantity of data would have changed that. A rule that cannot be executed is not evidence that was
overridden; it is a gap, and the replacement is justified by the gap. It is **not** justified by the
geometry result being more favourable than the alternative — which is the suspicion a reader is entitled
to hold, and the reason points 1 to 3 are stated before this one.

**What the new rule resolves to, and on what.** The new rule asks whether a control admits an artifact:
a control is admissible only if an arm reading nothing from the model, the image or the label, built at
the explainer's own native map resolution, reads within its own half-width of zero against it. Applied to
the pilot, `random_lowres_4` — Grad-CAM's own 4×4 resolution, no model input at any point — reads
−0.13929 (`black`), −0.15432 (`uniform_noise`), −0.12500 (`mean`), +0.04112 (`blur`), at 3.7 to 12.6
times its own half-width from zero in every scheme. The per-pixel-i.i.d. control is inadmissible. The
geometry-matched control is primary. One honest gap in that application: the geometry-matched control's
own admissibility check is trivially satisfied for `random_lowres_4` against itself and has never been
run against an independently seeded draw at the same grid, which is the only form of the check that could
fail. It is listed below as outstanding.

**What §2 now says, and what each part is.** H3 is restated, not deleted: same two parts, same
mixed-effects specification, same Cox model with leave-one-shift-family-out, same decision rules, same
insufficient-events rule. The outcome moves from `relative_faithfulness` to `matched_faithfulness`, and
that move forces exactly one new rule — H3a's refusal to normalise against a clean cell within its own
half-width of zero. H1a, H1b and H2a are added as primary and confirmatory for Version A; H2b is added
as a requirement on the instrument rather than a hypothesis, because the defect it describes has already
occurred. Every pilot figure quoted in §2 is labelled exploratory and generating at the point it appears,
with the number given, so a reader can reconstruct what was known when each hypothesis was written. That
distinction is the whole purpose of this document and §2 now carries it explicitly rather than relying on
§0.

**One claim in the recommendation A8 made is narrowed here.** A8 wrote that the per-pixel-i.i.d. control
"admits a model-free artefact of up to 0.2046, which is larger than any explainer effect measured here".
That is true globally — 0.19932 against a largest matched explainer effect of 0.12083 and a largest raw
explainer magnitude of 0.13321 — and it is **not** true scheme by scheme. Under `blur` the artifact at
grid 2 is +0.04398 against a largest matched effect of +0.09648, roughly half. H1a is therefore
registered per imputation scheme with the count of schemes reported, and is explicitly never pooled. The
narrowing does not change the recommendation: `blur` is also the scheme under which the whole geometry
ladder is flattest, spanning 0.0285 against 0.1615 to 0.2046 for the other three.

**Newly blocked: multi-seed aggregation, and with it the seed replication every §2 hypothesis needs.**
`pilot_report.require_one_model_and_seed` now refuses any set of faithfulness rows spanning more than one
(model_id, seed). The refusal is correct, and it was added because both alternatives are silent:

- `matched_rows` and `gate_verdicts` key their lookups on (explainer[, reference], imputation,
  shift_family, severity), with neither `model_id` nor `seed` in the key, so across three seeds the last
  one silently wins — 12 faithfulness rows over 3 seeds × 2 explainers × 2 cells collapse to 2 matched
  rows where 6 are expected.
- The averaging tables, the geometry ladder included, pool per-image values across seeds into a
  within-seed bootstrap. Seeds are a variance component, not more images, so that reports a between-seed
  effect inside a within-seed interval and returns a half-width that is too small — on the gate's own
  statistic, which *is* a half-width.

Aggregating across seeds correctly is §6's mixed-effects model, and it has not been implemented for
faithfulness or exercised on any real data. The consequence:

| Needs | Blocked by |
|---|---|
| H1a, H1b, H2a on more than one seed | §6 mixed-effects aggregation not implemented |
| H3a's `(1 \| seed)` term | the same; §6's estimator ladder has run on synthetic data only |
| H3b's uncensored-event count across seeds | the same |
| P3b.4 — seeds 1 and 2 | the same |

Every hypothesis in §2 is stated over (model, seed) and every one of them is currently measurable on
exactly one (model, seed). This is the largest single gap between what §2 registers and what the
instrument can do.

**Also blocked, and now blocking a primary prediction: `vit_tiny`.** Both claims were verified in the
code for this amendment rather than carried from A8:

| Claim | Where | Consequence |
|---|---|---|
| `gradcam_target_layer` raises for any architecture without `layer4` | `models/vision.py`: returns `model.layer4[-1]` behind `hasattr(model, "layer4")`, otherwise raises `ValueError` | `grad_cam` and `grad_cam_rolled` cannot run on a ViT |
| `to_normalised_tensor` emits 32×32 while `vit_tiny` is built at `img_size=112` | `data/cifar.py` converts `(N, 32, 32, 3)` uint8 to `(N, 3, 32, 32)` and never resizes; `models/vision.py` builds `timm`'s `vit_tiny_patch16_224` with `img_size=112` | **no** stage runs on `vit_tiny`, not merely the Grad-CAM arms |

`configs/version_a.yaml` is 3 models × 5 seeds × (1 clean + 5 families × 3 severities) = 240 cells, so
`vit_tiny` is 80 of them and none can run. H1b is the sharpest falsifiable prediction in §2 and it is the
one hypothesis there that cannot be executed at all today. §4 row 2 is the decision that unblocks it.

**Left standing, and listed rather than fixed: §3, §6 and §8 still describe the retired outcome.** The
author approved a change to §2 and §4. The following inconsistencies are created or exposed by it and are
recorded now so that none of them is discovered during the Version A analysis:

| Where | What it says | Why it is now wrong |
|---|---|---|
| §3 S1 | "After regressing out accuracy change, faithfulness change and calibration change remain correlated" | Inherits §2's outcome, which is no longer `relative_faithfulness`. Under the matched metric the regressand is a difference of two readings and inherits H3a's new normalisation refusal |
| §3 S2 | "Attribution stability and faithfulness do not degrade at the same rate" | Same inheritance. Stability is unaffected by A7 and A9; faithfulness is not |
| §3 S3 | Temperature scaling and AugMix "change faithfulness or stability" | Same inheritance, plus a separate defect: `train_model` took `augmix: bool = False` and no caller ever set it, so `resnet18_augmix` trained **identically** to `resnet18` and the two differed only in their cache key. S3 would have compared a model against itself at roughly 5.5 GPU-h. Fixed at commit 4348f0c — AugMix is now derived from the model id via `models.vision.uses_augmix`, read in `train.py` — but no `resnet18_augmix` checkpoint has ever been trained with AugMix, and `configs/version_a.yaml`'s budget block still states the pre-fix situation |
| §6, validity control 1 | "Faithfulness is **always** reported relative to a random-attribution control... `relative_faithfulness()` raises if the control is absent" | Still satisfied in letter — `matched_faithfulness` is a difference of two `relative_faithfulness` values and the raw control cancels exactly — but the words name the control the pilot disqualified. The fourth control H2b requires, asymmetry under the metric's own ordering, is missing |
| §8, falsifier | "if `relative_faithfulness` is indistinguishable from zero at *every* severity" | Names the retired quantity. On the matched metric the premise is not in doubt on seed 0: explainers beat their geometry-matched controls in 11 of 12 comparisons at 7.3 to 10.4 half-widths, the one exception being `black`/IG at 0.6, reported as unresolved |
| §7 | 240 primary cells | 80 of them cannot run |

None of these is amended here. Each is a change that was not approved, and inventing approval for it
inside the same amendment that insists on recording an approved change *as* approved would be
self-defeating.

**Nothing in A8 is retracted.** A9 narrows one claim in A8's recommendation (above) and executes the
decision A8 referred to the author. A8's reinstatements, withdrawals and the `black`/IG unresolved row
all stand as written. A7's retraction of A2 and A3 stands. A4, A5 and A6 stand.

**Not frozen. No `prereg-frozen` tag is cut by this amendment.** `git tag -l` remains empty. What stands
between here and a tag:

| Before a tag can be cut | Status |
|---|---|
| §4 row 2 decided — `vit_tiny` fixed or dropped | Open. Blocks H1b and 80 of 240 cells |
| §4 row 3 decided — the matched-metric gate run and read | Open. `gate_verdicts(value_key="matched")` exists in code and has never been run |
| §6 multi-seed aggregation implemented | Open. Blocks every §2 hypothesis beyond one seed, and P3b.4 |
| §6 validity controls amended to require an asymmetric control | Open. Recommended by H2b, not made here |
| §3 and §8 restated on the matched outcome | Open. See the table above |
| Version A's imputation axis settled | Open. One scheme reduces H1a's per-scheme count to a count out of 1, and cannot exhibit the across-scheme raw sign flip at all. Adding `blur` costs about +3.39 GPU-h on this grid |
| The geometry-matched control's non-trivial admissibility check | Open. Never run against an independently seeded draw at the same grid |
| Seeds 1 and 2 (P3b.4) | Open, and blocked by the multi-seed row above |

**Single-seed caveat, unchanged.** Everything §2 quotes from the pilot is `resnet18`, seed 0, 1000
images, three shift families: one architecture, one seed, one dataset. The geometry account still rests
on a single model's 4×4 Grad-CAM resolution. H1b exists precisely because that is a weakness, and H1b is
the hypothesis that cannot currently be run.

---

| Date | Section | Change | Reason | Data seen at time of amendment |
|---|---|---|---|---|
| 2026-10-15 | §4, row 3 (attribution sample size) | **Decided: 1000.** The matched-metric gate was run and fails at 0 of 29. The rule's escalation to 2000 is permitted and does not help: 0 of 29 at 2000 either | A10 | Session 1 rerun under `curves-v3` at `34e351a`: 510 units, 0 failed, 8713.3s, with `P3 GATE (GEOMETRY-MATCHED)` printed for the first time |
| 2026-10-15 | §8, row 4 | Fires again, on the corrected metric | A10 | As above |
| 2026-10-15 | §2, H3a / H3b | Recorded as **unresolved at this scale**, with point estimates, not as null results | A10 | As above |

**A10 — §4 row 3 is decided, and the matched gate decides it against H3 rather than for it.**
A9 left row 3 open because the matched-metric gate existed in code and had never been run. It has now
been run. Log: `docs/runs/2026-10-15b_curves-v3_with_matched_gate.log`.

**The run reproduces the previous one bit-for-bit.** The FAITHFULNESS, GEOMETRY LADDER and
GEOMETRY-MATCHED blocks are byte-identical between this log and
`docs/runs/2026-10-15_curves-v3_pilot_validity.log`, and the training losses match to the printed digit
at every reported epoch. That is not a formality here: `train-v2` put `epochs`, `batch_size` and `lr`
into the training cache key, so the checkpoint was not found and was retrained from scratch, and the
retrained weights reproduced every downstream number. Nothing in §2 or A8 moves.

**The matched gate fails completely: 0 of 29 resolvable pairs, against 14 of 85 on the raw metric.**
The six closest, with the sample size each would need, since the half-width falls as 1/√n:

| imputation | pair | family | ratio | n needed |
|---|---|---|---|---|
| `black` | grad_cam vs random_lowres_4 | defocus_blur | 0.143 | 2,045 |
| `uniform_noise` | grad_cam vs random_lowres_4 | gaussian_noise | 0.152 | 2,310 |
| `black` | grad_cam vs random_lowres_4 | fog | 0.154 | 2,372 |
| `uniform_noise` | grad_cam vs grad_cam_rolled | gaussian_noise | 0.161 | 2,592 |
| `black` | grad_cam vs grad_cam_rolled | fog | 0.164 | 2,690 |
| `black` | grad_cam vs grad_cam_rolled | defocus_blur | 0.172 | 2,958 |

Scaled across the whole set: **0 of 29 at n=2000**, 6 at n=3000, 7 at n=5000, 19 at n=10000 — and the
CIFAR-10 test set holds 10,000 images, so the last of those is the entire test set.

**How narrowly 2000 fails, stated so "0 of 29" is not read as a wide margin.** The best pair reaches
0.10112 at n=2000 against a threshold of 0.10 — it misses by 0.00112, and clears at n=2045. The second
and third miss by 0.0075 and 0.0089. So the escalation does not fail decisively; it fails by about one
percent of the threshold on one pair and by more on everything else. That is a reason to report the
escalation as insufficient rather than as refuted, and it is also why n=3000 buys six pairs where
n=2000 buys none. Nothing in §4 permits 3000, and this amendment does not invent it.

**So row 3 is decided at 1000, and decided by the rule rather than by default.** §4's rule escalates to
2000 on a failing gate. The escalation is permitted and is not taken, for the reason A4 already gave
and which the matched metric now reproduces exactly: 2000 moves nothing. A4 stands, and its arithmetic
now applies to the metric this document actually uses.

**What the gate does and does not bear on, which must not be blurred.** The gate measures the *change*
in matched faithfulness from clean to severity 5. It does not measure its *level*. The level is
resolved at 7.3 to 10.4 half-widths in 11 of 12 comparisons. So the corrected instrument can state that
an explainer beats a control of its own mask geometry, and cannot state whether that advantage degrades
under shift. H1 and H2 rest on the level and are unaffected by this amendment. H3 rests on the change
and is the thing the gate has just failed.

**H3a and H3b are recorded as unresolved at this scale, not as null.** The degradation is not absent.
All twelve matched pairs, change from clean to severity 5 as a fraction of the pair's own level,
against accuracy's **−36.3%** (0.8610 → 0.5487):

| imputation | pair | mean change | level | relative | vs accuracy |
|---|---|---|---|---|---|
| `black` | grad_cam vs random_lowres_4 | −0.0566 | +0.09194 | −61.6% | faster |
| `black` | grad_cam vs grad_cam_rolled | −0.0584 | +0.10068 | −58.0% | faster |
| `uniform_noise` | IG vs IG_rolled | −0.0265 | +0.04855 | −54.6% | faster |
| `mean` | IG vs IG_rolled | −0.0314 | +0.07389 | −42.5% | faster |
| `mean` | grad_cam vs grad_cam_rolled | −0.0454 | +0.11523 | −39.4% | faster |
| `uniform_noise` | grad_cam vs random_lowres_4 | −0.0430 | +0.11050 | −38.9% | faster |
| `mean` | grad_cam vs random_lowres_4 | −0.0384 | +0.10078 | −38.1% | faster |
| `uniform_noise` | grad_cam vs grad_cam_rolled | −0.0447 | +0.12083 | −37.0% | faster |
| `blur` | grad_cam vs grad_cam_rolled | −0.0121 | +0.09648 | −12.5% | **slower** |
| `blur` | grad_cam vs random_lowres_4 | −0.0081 | +0.09209 | −8.8% | **slower** |
| `blur` | IG vs IG_rolled | +0.0073 | +0.07100 | **+10.2%** | **rises** |
| `black` | IG vs IG_rolled | +0.0038 | −0.00398 | n/a | level unresolved |

H3a predicts faithfulness degrades *faster* than accuracy. **Eight of the eleven interpretable pairs
run in that direction and past it**, at 37.0% to 61.6% against accuracy's 36.3%. The three that do not
are all `blur`, where two degrade more slowly than accuracy and IG's advantage *increases* with
severity. The twelfth, `black`/IG, has a level of −0.00398 — unresolved at 0.6 half-widths — so a
ratio against it is meaningless and is not computed.

`blur` splitting from the other three schemes is the same split H1a shows, where `blur` is the one
scheme whose artifact does not exceed the largest matched effect. It is also the only scheme in which
all ten arms read positive on the raw metric. Whether one mechanism produces all three is not
established here and is not claimed.

None of this is resolved: the gate fails on every one of these pairs, which is exactly the statement
that their intervals are too wide to separate these rates from accuracy's. Reported as an unresolved
comparison with informative point estimates. Calling it a null would overclaim in the opposite
direction from H3a; calling it support for H3a would ignore the gate this amendment exists to record.

**§8 row 4 — "faithfulness intervals exceed the effect" — fires again, and this time on the metric the
document endorses.** A2 recorded it firing on the raw metric under the inverted mask; A8 recorded the
raw gate at 14 of 85 on corrected data. It now fires on the geometry-matched metric, which is the
instrument §4 row 1 selected. That is a stronger statement than either earlier one: the failure is not
an artifact of the broken control and not an artifact of the broken mask. The removal-curve family,
measured correctly against a correct control, does not resolve a degradation effect of this size on
CIFAR-10 at any sample size the dataset admits.

**What this does to Version A.** The 240-cell grid in `configs/version_a.yaml` was sized for H3's
mixed-effects fit. The gate is a per-(scheme, explainer, family) resolution check and the mixed-effects
model pools across 15 (model, seed) combinations, so the grid has more power than the gate and could
plausibly resolve the H3a interaction where the gate cannot. That is a real difference and this
amendment does not claim H3 is dead. But §4's rule offers 1000 or 2000 and offers nothing else;
"resolve it by pooling the full grid" is not a rescue the protocol pre-registered, and adopting it
after seeing the gate fail would be the same post-hoc move A9 refused to disguise. Whether to run
Version A for H3 is therefore an explicit decision for the author, on the record, and is **not taken
here**. H1 and H2 need no such grid: they are resolved, and what they need is seeds and `vit_tiny`.

**Row 3's entry in the freeze table above is discharged.** The others stand unchanged.

**Operational note, recorded because it is the second time.** CIFAR-10 was downloaded inside the
session again, 51:15 against the previous run's 22:42, so roughly 51 minutes of the 2.42-hour session
was a download that mounting a dataset removes entirely.
