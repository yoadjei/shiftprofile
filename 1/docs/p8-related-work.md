# P8 Related Work — Measurement Validity of Removal-Based Faithfulness

**Date:** 2026-10-18
**Scope:** related work for the measurement-validity paper (claims 1–4 below), which the P1 novelty
gate never covered. P1 (`docs/p1-novelty-gate.md`) was run for the joint
calibration-and-faithfulness-under-shift framing and concluded the validity fallback was not needed; the
validity literature was swept at abstract level and dropped. This document is the sweep for the paper
actually being written.

**Verification standard:** inherited from P1 §Method — an abstract is not sufficient grounds. Every paper
marked close below was read in full text (PDF downloaded and extracted, or HTML fetched) and the deciding
sentence is quoted with its location. Papers whose full text could not be reached are in §7 and are **not**
used to support any verdict.

**Claims under test** (from `prereg/preregistration.md` §2 H1a/H1b/H2a/H2b and `docs/results.md` Part 3;
Part 4 did not exist in `results.md` at the time of this sweep):

1. **Geometry artifact.** A model-free arm — noise on a k×k lattice, ReLU'd and bilinearly upsampled
   through Grad-CAM's exact pipeline, reading nothing from model, image or label — reaches −0.197 on
   32×32 CIFAR-10 against the conventional per-pixel-i.i.d. random-attribution control, monotone in blob
   size across a 2→32 ladder spanning 0.20, as large as or larger than any real explainer effect measured.
2. **The standard control is blind to ranking inversion.** For a per-pixel-i.i.d. attribution the bottom
   *k* is as random as the top *k*, so the control's reading is invariant to inverting the ranking;
   demonstrated by an entire pilot run with an inverted mask passing all three pre-registered controls.
3. **The fix.** Two independently constructed geometry-matched controls — the explainer's own map
   displaced by a seeded per-image offset (`_rolled`), and model-free noise at the explainer's native map
   resolution. Explainers beat them in 11 of 12 (scheme, pair) cells.
4. **Power.** With the corrected control the pre-registered half-width gate passes 0 of ~29 pairs in all
   three seeds, and the needed n exceeds the CIFAR-10 test set.

---

## 1. Verdict

**Claim 3 is novel as stated and is the paper's real contribution. Claim 4 is novel. Claims 1 and 2 need
narrowing, and claim 1 needs narrowing hard — the headline is not the discovery the project thinks it
is.** The proposition that a trivial or model-free attribution map matches or beats real explainers on a
standard removal-based faithfulness metric is published at least four separate times: ROAR's Sobel control
(2019), Poppi et al.'s Fake-CAM (2021), Gupta et al.'s Random and Centered-Gaussian baselines (2022), and
Nieradzik et al.'s flat statement that "Uniform is better than GradCAM" on Deletion (2024). Every one of
those papers diagnosed a *different* cause — the need to retrain, metric asymmetry, dataset centrality
bias, out-of-distribution shift — and **none** diagnosed mask geometry, none varied blob scale, and none
built a control that holds geometry fixed. So claim 1 survives as a *mechanism-and-magnitude* claim (the
cause is the mask's spatial scale; here is the ladder; here is the size in the same units as real effects)
and dies as a *discovery* claim ("removal metrics admit an artifact"). Claim 2 must narrow similarly:
ROAR Fig. 2 already showed in 2019 that a non-retrained removal metric scores the inverted ground-truth
ranking as informative, so the novel residue is the specific invariance *proof* about the per-pixel-i.i.d.
control plus the costed demonstration that a three-control pre-registered suite passed under inversion.
Claim 3 is where nothing was found: three searches with different vocabularies returned no XAI work using
a displaced copy of the explainer's own map as a control, and the only prior art is in a different field
(neuroimaging's spatial-autocorrelation-preserving null models), which the paper should cite as precedent
rather than wait for a reviewer to produce. Claim 4 has no competition at all — no published power or
sample-size analysis of removal-based faithfulness metrics was found; Tomsett et al. (2020) is the
qualitative version of it. **The framing must change** (see §6): from "we found an artifact" to "the
artifact has been found four times and misdiagnosed four times; here is its actual cause, a control that
removes it, and the cost of that control." One blocking item: claim 1's mechanism is currently
*contestable on already-published evidence*, because Gupta et al. attributed their model-free blob's
success to dataset centrality bias and tested that attribution. The paper must rule centrality out (§3.1).

---

## 2. Positioning table

One row per genuinely close paper. "What it does not do that we do" is the gap sentence; each is backed by
a quote in §3.

| Paper | Year | What it actually does | What it does *not* do that we do |
|---|---|---|---|
| [ROAD / Rong et al.](https://arxiv.org/abs/2202.00449) | 2022 | Information-theoretic analysis of pixel-perturbation evaluation. Decomposes the metric into Feature Info + **Mask Info** − Mitigator, where Mask Info is `I(C;M)`, the mutual information between the *mask* and the *class*. Fixes it with Noisy Linear Imputation | The leakage term is `I(C;M)`, which requires the mask to carry class information. A model-free lattice blob has `I(C;M)=0` by construction and still reads −0.197, so ROAD's mechanism cannot account for it. ROAD has **no random-attribution baseline at all** (verified by grep over the full text), varies no mask scale, and proposes no geometry-matched control |
| [ROAR / Hooker et al.](https://arxiv.org/abs/1806.10758) | 2019 | Retrains after removal. Two model-independent controls: random ranking and a **Sobel edge filter**; base estimators "perform on par or worse than SOBEL". Toy experiment shows the non-retrained metric misreads the *inverted* ranking | Sobel reads the image; our arm reads nothing — not model, not image, not label. The controls are a pass/fail floor, not a quantified artifact, and are not varied in scale. The inversion result is on 16-dimensional synthetic data with a least-squares model, concerns the raw curve rather than the control, and carries no claim that the control is *invariant* to inversion |
| [Gupta et al. (NeurIPS)](https://arxiv.org/abs/2211.02912) | 2022 | Completeness/soundness metrics. Includes a per-pixel **Random** Gaussian map and a model-free **Centered Gaussian** blob as baselines; Random scores mid-pack on Deletion, Centered Gaussian near-top on Insertion and Saliency Metric; builds Imagenette-Corner to show the blob's success is **dataset centrality bias** | One blob, one scale, fixed at the image centre, diagnosed as an *alignment* effect. Ours is randomly positioned, monotone in scale, and the geometry account is a claim about spatial autocorrelation rather than registration. They never compare the blob's magnitude to real explainer effects in common units, and never match geometry between arms |
| [Nieradzik et al.](https://arxiv.org/abs/2411.14946) | 2024 | Benchmarks 16 attribution maps with Uniform and Canny as sanity-check baselines; reports that Deletion fails the baseline test — "Uniform is better than GradCAM". Attributes it to **distribution shift** and replaces pixel modification with adversarial perturbation | Their "Uniform" *is* the per-pixel-i.i.d. control we argue is the wrong instrument, compared on absolute Del score rather than used as a reference. No blob, no scale ladder, no geometry-matched control, and the diagnosis is OOD shift, which a geometry account predicts would not be monotone in grid size |
| [Rao et al. (CVPR)](https://arxiv.org/abs/2205.10435) | 2022 | Argues methods can only be compared fairly at the same layer (ML-Att) because layer choice — hence **native map resolution** — "skews any comparison"; apparent differences "effectively vanish" when compared fairly; Gaussian smoothing of IntGrad/IxG "significantly improves their performance" | The metric throughout is **GridPG localization**, not a removal curve. Resolution is treated as a confound between *methods*, never as a quantity that generates a score from nothing. No model-free arm, no removal AUC, no control |
| [Poppi et al. (CVPRW)](https://arxiv.org/abs/2104.10252) | 2021 | **Fake-CAM**: a map that is 1 everywhere except a single zeroed corner pixel. Beats SOTA on Average Drop and Average Increase on the full ImageNet validation set. Proposes ADCC | Wins by masking almost nothing — it exploits the asymmetry and area-insensitivity of AD/AI. Not a removal-curve AUC against a control, and the mechanism is metric asymmetry, not mask geometry |
| [Binder et al.](https://arxiv.org/abs/2211.12486) | 2022 | Proves that "uninformative attribution maps created with zero pixel-wise covariance easily achieve high scores" on model-randomization sanity checks, and concludes those checks are inadequate for ranking methods | The object is Adebayo-style randomization checks scored by SSIM/Spearman, not a removal-metric control arm. Closest in *spirit* to claim 2 — an i.i.d. map games the check — but about a different check, and about score-maximisation rather than invariance to ranking inversion |
| [Adebayo et al.](https://arxiv.org/abs/1810.03292) | 2018 | Model-parameter and data randomization tests. Uses an **edge detector**, which "does not depend on model or training data", to argue visual inspection is a poor guide | Both tests compare *maps to each other* under randomization of the model or labels (Spearman, SSIM, HOG). Neither is a control arm inside a removal metric, and the edge-detector point is about visual similarity, not about a metric producing a score from mask shape |
| [Domeniconi et al.](https://arxiv.org/abs/2605.14641) | 2026 | Shows AD and ADCC are fooled by trivial CAMs (RandomCAM, HalfCAM, All1sCAM) but that **ROAD and their ARCC correctly assign low scores** to them | Their trivial maps are per-pixel uniform or near-constant. A coarse upsampled blob is exactly the trivial map they do not test, so their ROAD-is-robust result does not cover us — but it is the sentence a reviewer will use, and §3.1 says what to do about it |
| [Burt et al. (bioRxiv)](https://www.biorxiv.org/content/10.1101/2020.02.18.955054v1.full) | 2020 | Neuroimaging. Surrogate maps that preserve spatial autocorrelation because "randomly shuffling a brain map necessarily destroys its SA structure"; describes the spin test as "randomizing the anatomical alignment ... through spherical rotation by a random angle" | Not XAI and not a competitor — the methodological **precedent** for `_rolled`. Our control is a translation-based spatial permutation of an attribution map, i.e. the image-domain analogue of a spin test. Cite it; it converts the control from an ad hoc trick into an imported standard |

**No paper found attributes a removal-metric artifact to mask geometry or scale, and no paper found uses a
displaced copy of the explainer's own map as a control.**

---

## 3. Per-claim verdicts

### 3.1 Claim 1 — the geometry artifact. NARROW IT, and add one control.

**Closest work, in descending order of danger: Gupta et al. 2022, Nieradzik et al. 2024, ROAD 2022.**

**Gupta et al., NeurIPS 2022** ([arXiv 2211.02912](https://arxiv.org/abs/2211.02912)). §6.2, describing
baselines: *"We also include simple baselines like 'Random' map (each entry is a random Gaussian), and
'Centered Gaussian' map (2-dimensional isotropic Gaussian distribution placed at the image center)."* On
the deletion metric, §6.2: *"Although we note that 'Random' baseline performs quite well on the Deletion
metric, which suggests that deletion metric does not provide much signal on these datasets."* And on
CIFAR-10, Appendix D Table 5 caption: *"The deletion and saliency metrics are uninformative in this case,
since all methods are as good (or worse) compared to a random mask."* Their Table 1 (Imagenette, Deletion
lower-is-better) gives Centered Gaussian 0.66, Random 0.45, Gradient⊙Input 0.42, iGOS++ 0.37, Fong &
Vedaldi 0.58; Insertion-gray higher-is-better gives Centered Gaussian 0.74 against a best real method of
0.75. The Table 1 caption states: *"Strong performance of [Centered Gaussian] is likely due to centrality
bias (cf. Appendix C)."* Appendix C item 3: *"The results show that the good performance of Centered
Gaussian on Imagenette is likely due to the bias of datasets (that objects are centered with high
probability)."*

This is the most dangerous paper in the sweep and it is dangerous twice over. First, it already has a
model-free blob that scores at the top of a removal-family metric. Second, and worse, it has a *published
alternative explanation for our result* — centrality bias — together with the experiment that tests it
(Imagenette-Corner, where the blob's numbers drop). CIFAR-10 is a centred-object dataset, and coarser
grids on a 2→32 ladder put proportionally more mask mass near the centre. **A reviewer holding Gupta et
al. can claim the entire 2→32 ladder is a centrality-bias gradient and not a geometry effect, and the
project currently has no evidence against that.** This is not a positioning problem, it is a missing
control: the paper needs an off-centre replication (random-crop or corner-crop CIFAR, the Imagenette-Corner
analogue), or the artifact reported conditioned on object eccentricity, or an explicit statement that the
model-free arm's blob positions are uniform over the image and that the ladder's monotonicity holds within
position strata. Until one of those exists, claim 1's mechanism is contestable on evidence that is already
in print.

**Nieradzik et al. 2024** ([arXiv 2411.14946](https://arxiv.org/abs/2411.14946)). Table 1 caption: *"The
table shows that Del and Ins do not pass the baseline test, as Canny and Uniform should have the worst
values."* And in the evaluation discussion: *"We have already established comparatively in table 1 that
this is indeed the case for the Deletion function (Uniform is better than GradCAM)."* Their Uniform is
defined in §3, "Baseline AMs and ground truth": *"The first of these baselines is called uniform, which
emulates an equal importance of all AM pixels by sampling from the uniform distribution."* The abstract
attributes the failure to shift: *"We show that the widely used insertion/deletion metrics are susceptible
to distribution shifts that affect the reliability of the ranking."*

So the sentence "a model-free map scores as well as Grad-CAM on a removal metric" is in print, flatly, in
2024. Our distinction is real but it is a distinction, not a discovery: their Uniform is per-pixel i.i.d.
— the very control we argue is the wrong instrument — compared on absolute Del score rather than used as
a reference level, and their diagnosis is distribution shift, which does not predict monotonicity in grid
size. The grid ladder is the discriminating evidence between the two accounts and should be presented as
such.

**ROAD** ([arXiv 2202.00449](https://arxiv.org/abs/2202.00449)) is the paper a reviewer will reach for
first, because its abstract contains the word "shape": *"Our findings reveal that the results are strongly
affected by information leakage through the shape of the removed pixels as opposed to their actual
values."* §4.2 is titled "Class Information Leakage through Masking" and opens: *"We demonstrate that it
is easily possible to leak class information only through the mask's shape and to harshly manipulate the
evaluation score."* The decomposition that follows §4.2 is the identity
`I(x_l;C) = I(C;x_l|M) + I(C;M) − I(C;M|x_l)`, labelled in the paper as Eval. Outcome = Feature Info +
**Mask Info** − Mitigator. Their worked example (Appendix A) is explicitly class-dependent: *"Suppose an
explanation map that starts masking out pixels at the top for class zero and at the bottom for class one.
Thus, a retrained model will be able to infer the category just from the shape of the masked pixels and
obtain the best possible accuracy and thus score in the LeRF setting."*

**ROAD's "shape" is not our shape, and the difference is stateable in one line:** ROAD's leakage term is
`I(C;M)`, which is identically zero for a mask drawn from lattice noise that never sees the label. Put that
sentence in the paper. A grep of ROAD's full text for a random-attribution baseline returns nothing — ROAD
has no random-attribution arm anywhere, so it cannot have measured what we measure.

**The ROAD question the paper must answer anyway.** Domeniconi et al. 2026
([arXiv 2605.14641](https://arxiv.org/abs/2605.14641)) §3.3 tests three trivial maps — RandomCAM (per-pixel
uniform), HalfCAM, All1sCAM — and reports: *"We observe that HalfCAM fools both Average Drop and ADCC,
producing values comparable with non-trivial CAMs. In contrast, ARCC and ROAD correctly assign low scores
to these trivial CAMs, validating their theoretical robustness and reliability towards trivial attribution
maps."* That is a published claim that ROAD resists trivial attributions. It does not cover a coarse
upsampled blob, which they never test, but the paper's four imputation schemes (`black`, `mean`, `blur`,
`uniform_noise`) do not include ROAD's Noisy Linear Imputation either. **Either run one model-free ladder
under ROAD's imputation, or state in scope that the artifact is demonstrated for the four schemes run and
that its survival under ROAD imputation is untested.** The second is acceptable; silence is not.

**Secondary, and useful rather than threatening.** The founding paper of the convention,
Samek et al. ([arXiv 1509.06321](https://arxiv.org/abs/1509.06321)), established exactly the comparison we
critique — §IV-B: *"As a baseline we also compute the AOPC curves for random heatmaps (i.e., random
ordering O). Figure 4 displays the AOPC values as function of the perturbation steps (i.e., L) relative to
the random baseline."* Two details are worth the paper's attention. Their random baseline is a random
ordering of **9×9 non-overlapping regions** (§IV-A: *"Perturbation is applied to 9×9 non-overlapping
regions, each covering 0.157% of the image"*), not per-pixel i.i.d. — so the per-pixel form of the control
is a later simplification, and the original convention was partially geometry-aware. And they already
observed the smoothness–score association, reading it as joint evidence of quality: §II states
*"heatmaps should have low 'complexity', i.e., be as sparse and non-random as possible"* and §IV-C reports
*"LRP has best performance on all three data sets and its heatmaps have smallest file size (which means
that they are well compressible, i.e., have low complexity)."* The correlation between blobbiness and AOPC
has been visible since 2017 and was interpreted in the opposite direction. That is a strong, citable
opening for the paper and it costs nothing.

**Rao et al., CVPR 2022** ([arXiv 2205.10435](https://arxiv.org/abs/2205.10435)) is the nearest thing to
H1b. Abstract: *"To address fairness, we note that different methods are applied at different layers,
which skews any comparison, and so evaluate all methods on the same layers (ML-Att) and discuss how this
impacts their performance on quantitative metrics."* Contribution 2: *"We show that, when compared fairly,
apparent performance differences between some methods effectively vanish."* And Fig. 5 caption:
*"Smoothing the attributions for IntGrad and IxG significantly improves their performance at the input
image and middle layer."* Their metric is GridPG localization throughout, not a removal curve, and
resolution is a confound *between methods* rather than a quantity that manufactures a score from nothing.
H1b's cross-model prediction (4×4 ResNet-18 versus 7×7 ViT-Tiny) is not pre-empted — but note that
prereg §2 records H1b as currently unrunnable, so the paper cannot lean on it.

**Gap sentence.** No published work attributes a removal-based faithfulness artifact to the spatial scale
of the mask, quantifies it as a monotone function of that scale, or measures its magnitude against real
explainer effects in the same units — but that a trivial or model-free map can match real explainers on
such a metric is published repeatedly, so claim 1 must be stated as a mechanism-and-magnitude result and
must rule out centrality bias to hold its mechanism.

---

### 3.2 Claim 2 — the control is blind to ranking inversion. NARROW IT.

**Closest work: ROAR 2019, then Binder et al. 2022.**

**ROAR** ([arXiv 1806.10758](https://arxiv.org/abs/1806.10758)) §3, toy experiment: *"We compare three
rankings: the ground truth importance ranking, random ranking and the inverted ground truth ranking (the
worst possible estimate of importance). In the left plot of Fig. 2 we can observe that without re-training
the worst case estimator is shown to degrade performance relatively quickly."* Fig. 2 caption is blunter:
*"In the case where the model is not retrained, test-set accuracy quickly erodes despite the worst case
ranking of redundant features as most important. This incorrectly evaluates a completely incorrect feature
ranking as being informative."*

That is the general content of claim 2, in print since 2019. It is on 16-dimensional synthetic data with a
least-squares model, it concerns the raw removal curve rather than a control arm, and ROAR's prescription
is retraining. What remains ours, and should be the whole of what claim 2 asserts:

- the **invariance argument**, which is a property of the control and not an empirical finding: for a
  per-pixel-i.i.d. attribution map the bottom *k* pixels are distributionally identical to the top *k*, so
  the control's reading is *exactly* invariant under inversion of the ranking — not merely insensitive;
- the **costed demonstration** that three independently pre-registered validity controls all passed
  through an entire pilot run with the mask selecting lowest-magnitude pixels, reading ≈0 before the fix
  and ≈0 after (prereg §2 H2b; `metric_provenance.md` retraction; A7, L5);
- the **requirement** that follows — at least one control in every faithfulness comparison must be
  asymmetric under the metric's own ordering.

ROAR should be cited as the prior demonstration that a non-retrained removal metric misreads an inverted
ranking, and the paper should say plainly that it is not claiming priority on that.

**Binder et al. 2022** ([arXiv 2211.12486](https://arxiv.org/abs/2211.12486)) is the closest statement of
the same *kind* of failure. Abstract: *"Firstly, we show that uninformative attribution maps created with
zero pixel-wise covariance easily achieve high scores in this type of checks"*, and *"these results
demonstrate the inadequacy of model-randomization-based sanity checks as a criterion to rank attribution
methods."* Their Theorem 1 proves *"the expected SSIM absolute value is minimized by a random variable
with zero covariance."* The object is Adebayo's randomization tests scored by SSIM/Spearman, not a control
arm inside a removal metric, and the result is that an i.i.d. map *maximises* the check's score rather than
that the check is *invariant* to a defect. Different proposition, same family; cite as the nearest
precedent for "the control itself fails."

**Adebayo et al. 2018** ([arXiv 1810.03292](https://arxiv.org/abs/1810.03292)) is adjacent and is often
assumed to cover this. It does not. Both of its tests compare *maps to each other* under randomization of
model parameters (§3: *"we compare the resulting explanation from a network with random weights to the one
obtained with the model's original weights"*) or of labels (§4), scored by Spearman rank correlation, SSIM
and HOG correlation (§2, "Similarity Metrics"). Neither randomizes the attribution's *ordering*, and
neither is a control arm inside a removal metric. The one-sentence distinction for the paper: Adebayo's
tests ask whether the explainer responds to the model; claim 2 asks whether the metric's control responds
to the metric's own implementation.

**Gap sentence.** That a removal metric can score an inverted ranking as informative is ROAR's 2019 result;
what is unpublished is that the per-pixel-i.i.d. control is *provably* invariant to that inversion, and the
demonstration that a full pre-registered control suite therefore passed through a real inverted-mask bug.

---

### 3.3 Claim 3 — geometry-matched controls, `_rolled` in particular. NOVEL.

**No XAI prior art found.** Three searches with distinct vocabularies — "shifted/translated saliency map
control", "spatially shuffled / location-permuted attribution baseline", "sanity check translation
invariance attribution" — returned nothing using a displaced copy of the explainer's own map as a control.
The negative is corroborated from the other direction by what the closest papers *do* use as controls:

- ROAD: no random-attribution baseline at all (grep over full text).
- ROAR: random ranking, Sobel edge filter (§1, §4 "Control Variants").
- Nieradzik et al.: Uniform, Canny (§3, "Baseline AMs and ground truth").
- Gupta et al.: Random Gaussian, Centered Gaussian (§6.2).
- Poppi et al.: Fake-CAM (§3).
- AIM, Hsieh et al. 2026 ([arXiv 2605.16905](https://arxiv.org/abs/2605.16905)): random feature orderings
  only, as a "random-attribution bias" diagnostic. Their confound is the masking operator — *"this
  degradation can be confounded by the masking operator"*, with *"zero masking may create
  out-of-distribution artifacts"* and interpolation that *"may leak information through the shape,
  location, or reconstructed content of the masked region."* Shape appears as one leakage channel for
  interpolation, not as the confound, and their fix is PGD-adversarial replacement. No geometry-matched
  control, no translated map.

Every control in the literature either destroys the value distribution, destroys the spatial structure, or
both. `_rolled` destroys neither: identical value multiset, identical spatial autocorrelation, only
registration to the image broken. That is the paper's actual contribution and it should lead.

**Precedent to cite, from another field.** This construction is standard practice in neuroimaging, where
testing correspondence between two spatially autocorrelated maps requires a null that preserves
autocorrelation while randomizing alignment. Burt et al. 2020
([bioRxiv 10.1101/2020.02.18.955054](https://www.biorxiv.org/content/10.1101/2020.02.18.955054v1.full))
state the problem — *"randomly shuffling a brain map necessarily destroys its SA structure"* — and describe
the spin test as a procedure that *"involves randomizing the anatomical alignment between two cortical
surface maps through spherical rotation by a random angle."* A seeded per-image translation of an
attribution map is the image-domain analogue: autocorrelation-preserving, alignment-destroying. Citing this
does three things — it establishes the control is principled rather than invented, it pre-empts a reviewer
from that background, and it gives the paper a vocabulary ("SA-preserving null", "geometry-matched
control") that is already load-bearing elsewhere. The counterpart to flag honestly: these nulls are known
to be imperfectly calibrated at high autocorrelation, so the paper should not claim the matched control is
assumption-free.

One internal caution, not a literature finding: `results.md` Part 3 records that `grad_cam_rolled` lands
on the model-free line while `integrated_gradients_rolled` reads +0.04985 above it under `mean`. The
"identical geometry, aim destroyed" argument is therefore clean for Grad-CAM and partial for IG, and the
paper should say so where it makes the claim rather than in a limitations paragraph.

**Gap sentence.** No work in the XAI evaluation literature uses a displaced copy of the explainer's own
attribution map — same values, same spatial autocorrelation, broken registration — as a control for a
removal-based faithfulness metric; the construction's only precedent is the
spatial-autocorrelation-preserving null-model literature in neuroimaging, which should be cited as such.

---

### 3.4 Claim 4 — the power result. NOVEL.

No published power analysis, sample-size calculation or minimum-detectable-effect analysis for
removal-based faithfulness metrics was found. The closest is **Tomsett et al., AAAI 2020**
([arXiv 1912.01451](https://arxiv.org/abs/1912.01451)), which is the qualitative statement of the same
problem. Abstract: *"Our results show that saliency metrics can be statistically unreliable and
inconsistent, indicating that comparative rankings between saliency methods generated using such metrics
can be untrustworthy."* Introduction: *"we show that current metrics used to estimate saliency method
fidelity can have high variance and are sensitive to implementation details. We also show that these
metrics can be statistically unreliable when considered over individual saliency maps, and that the
metrics we examined were not measuring the same underlying notion of fidelity."* They apply psychometric
reliability measures; they do not compute what n a given comparison requires.

Claim 4 is the quantitative instantiation: a pre-registered gate, a stated half-width criterion, 0 of ~29
pairs passing, and an n requirement that exceeds the available test set. Nothing in the sweep competes
with it, and it is the most defensible of the four claims. It is also the one that makes the paper
*useful* rather than merely critical: it converts "this metric is unreliable" into "at the dataset sizes
this field uses, this comparison cannot be made."

**Gap sentence.** The variance and inter-metric inconsistency of saliency fidelity metrics is documented
(Tomsett et al. 2020); the sample size required for a geometry-matched removal-based comparison to resolve
a degradation effect has never been computed, and the finding that it exceeds the CIFAR-10 test set is new.

---

## 4. Cite these anyway

Not novelty threats. Required for the paper to be credible.

| Work | Cited for |
|---|---|
| [Samek et al. 2017, TNNLS](https://arxiv.org/abs/1509.06321) | Origin of region-perturbation / AOPC and of the "relative to the random baseline" convention the paper critiques. Also the 2017 observation that the best-scoring heatmaps are the most compressible — the smoothness–score association, read in the opposite direction |
| [Petsiuk et al. 2018, BMVC (RISE)](https://arxiv.org/abs/1806.07421) | Origin of the deletion/insertion **AUC** form of the metric. §3: *"The deletion metric measures the drop in the probability of a class as important pixels (given by the saliency map) are gradually removed from the image"* |
| [Hooker et al. 2019, NeurIPS (ROAR)](https://arxiv.org/abs/1806.10758) | Retraining as the remedy for removal-induced shift; the model-independent control precedent (random + Sobel); the inverted-ranking demonstration behind claim 2 |
| [Rong et al. 2022, ICML (ROAD)](https://arxiv.org/abs/2202.00449) | Class-information leakage through the mask, `I(C;M)`; Noisy Linear Imputation; the paper against which the geometry claim must be distinguished explicitly, in the introduction and not only in related work |
| [Adebayo et al. 2018, NeurIPS](https://arxiv.org/abs/1810.03292) | Randomization sanity checks; the edge-detector argument that model-free output can look like an explanation; the canonical statement that a control can fail |
| [Binder et al. 2022](https://arxiv.org/abs/2211.12486) | Prior proof that an uninformative i.i.d. map games a published sanity check — nearest precedent for claim 2's family |
| [Tomsett et al. 2020, AAAI](https://arxiv.org/abs/1912.01451) | Strongest early statement that faithfulness metrics are statistically unreliable and do not measure the same construct. The measurement-validity anchor and the lead-in to claim 4 |
| [Selvaraju et al., Grad-CAM](https://arxiv.org/abs/1610.02391) | The explainer; the 4×4 native map resolution on CIFAR ResNet-18 that the geometry account turns on; the exact upsampling pipeline the model-free arm imitates |
| [Sundararajan et al. 2017, ICML (IG)](https://arxiv.org/abs/1703.01365) | The second explainer; pixel-level attribution, the contrast case for H1b |
| [Hendrycks & Dietterich 2019, ICLR](https://arxiv.org/abs/1903.12261) | CIFAR-10-C; the severity ladder and corruption families |
| [Sturmfels, Lundberg & Lee 2020, Distill](https://distill.pub/2020/attribution-baselines/) (DOI 10.23915/distill.00022) | Baseline/reference-value choice for IG — black/zero, maximum-distance, blurred, uniform, Gaussian, expected gradients. Cite **with the distinction stated**: this is the attribution's reference point, whereas `black`/`mean`/`blur`/`uniform_noise` is the *imputation* inside the removal metric. The axes overlap in substance and are routinely conflated |
| [Haug et al. 2021](https://arxiv.org/abs/2101.00905) | "On Baselines for Local Feature Attributions" — the tabular counterpart to Sturmfels; baseline choice materially changes attributions |
| [Morasso et al. 2025](https://arxiv.org/abs/2503.19813) | Recent guidelines for baseline choice; shows the baseline question is live, not settled |
| [Gomez et al. 2022](https://arxiv.org/abs/2201.13291) | Two critiques of DAUC/IAUC the paper should adopt rather than rediscover: *"the actual saliency score values given by the saliency map are ignored as only the ranking of the scores is taken into account"* (§2.2) and *"DAUC and IAUC generate out of distribution (OOD) images"* (§2.2). The first matters directly — if only the ranking matters, `_rolled`'s preservation of the value multiset is a property of the construction, not of the metric's reading |
| [Poppi et al. 2021, CVPRW](https://arxiv.org/abs/2104.10252) | Fake-CAM; ADCC. The canonical "a trivial map wins a popular faithfulness metric" citation |
| [Zhang et al. 2024, CVIU (Opti-CAM)](https://arxiv.org/abs/2301.07002) | §1: *"there exists a trivial attribution method called Fake-CAM (39) that outperforms the state of the art in both metrics."* Independent confirmation of Poppi, and the metric-asymmetry diagnosis |
| [Gupta et al. 2022, NeurIPS](https://arxiv.org/abs/2211.02912) | Random and Centered-Gaussian baselines; *"deletion metric does not provide much signal on these datasets"*; the centrality-bias diagnosis and Imagenette-Corner, which the paper must answer |
| [Nieradzik et al. 2024](https://arxiv.org/abs/2411.14946) | *"Uniform is better than GradCAM"* on Deletion; the OOD-shift diagnosis |
| [Rao et al. 2022, CVPR](https://arxiv.org/abs/2205.10435) | Layer/resolution skews method comparison; smoothing improves metric scores. Nearest prior art for H1b |
| [Barr et al. 2023](https://arxiv.org/abs/2311.07763) | §5: *"Across the experiments, the ranked correlations reveal little consensus on the notion of faithfulness in the explanations."* Faithfulness metrics disagree with each other |
| [Hedström et al. 2023, TMLR (MetaQuantus)](https://arxiv.org/abs/2302.07265) | Meta-evaluation — evaluating the evaluator. The methodological frame this paper sits inside |
| [Hedström et al. 2024](https://arxiv.org/abs/2405.02383) | "A Fresh Look at Sanity Checks for Saliency Maps" — sMPRT/eMPRT. Precedent for repairing rather than discarding a broken check, which is what claim 3 does |
| [Zullich et al. 2026](https://arxiv.org/abs/2610.12493) | The current position paper on faithfulness-metric validity. Its five shortcomings are *"(1) disproportionate focus on classification ... (2) conflation of faithfulness with other XAI evaluation facets, (3) necessity of an arbitrary baseline for simulating feature removal, (4) emergence of OoD data ... and (5) misalignment between the main metrics and the proper computation of faithfulness."* **Mask geometry is not among them** — cite it as the state of the art the paper adds a sixth item to. Note also its Proposal (I), "contemplate model calibration when FI needs formal evaluation", which touches this project's calibration track |
| [Domeniconi et al. 2026](https://arxiv.org/abs/2605.14641) | The claim that ROAD resists trivial attribution maps — the reviewer's strongest ROAD defence, and the reason §3.1's ROAD-imputation item is not optional |
| [Hsieh et al. 2026 (AIM)](https://arxiv.org/abs/2605.16905) | Current statement of masking-operator confounding: *"a faithfulness score may reflect masking-induced bias rather than true model reliance."* Closest contemporary framing, different mechanism |
| [Miró-Nicolau et al. 2024](https://arxiv.org/abs/2311.01961) | Ground-truth-explanation datasets for fidelity assessment; the synthetic-ground-truth alternative to control-based validation |
| [Burt et al. 2020](https://www.biorxiv.org/content/10.1101/2020.02.18.955054v1.full) | Spatial-autocorrelation-preserving surrogate maps; the cross-field precedent for `_rolled` (see §3.3) |

---

## 5. What a reviewer will attack, in order

1. **"ROAD already covers this — it says 'shape' in the abstract."** Answer in one line in the
   introduction: ROAD's shape term is `I(C;M)`, mutual information between mask and class, and is zero by
   construction for a label-blind lattice blob. Do not bury this in related work.
2. **"Gupta et al. showed this is centrality bias."** Currently unanswerable. See §3.1 — needs an
   off-centre replication or a position-stratified ladder.
3. **"Nieradzik et al. already published 'Uniform beats Grad-CAM'."** Answer: their Uniform is the control
   we indict, compared as a method; the grid ladder is what separates a geometry account from their
   OOD account.
4. **"Would the artifact survive ROAD's imputation?"** Untested. Run one ladder or scope it out explicitly.
5. **"Poppi/Opti-CAM already showed trivial maps win."** Answer: on AD/AI, by masking nothing, via metric
   asymmetry. Different metric, different mechanism, and we cite them.
6. **"Resolution confounds are known (Rao et al.)."** Answer: on localization metrics, between methods.
   H1b is the removal-metric version and is currently unrunnable, so do not over-promise it.

---

## 6. Framing

**It needs to change.** The current framing — four claims led by a discovery ("removal-based faithfulness
has a geometry artifact and it is as large as any real effect") — is the one framing the literature can
embarrass. The artifact's *existence* has been found four times since 2019 and the paper will read as a
fifth rediscovery to anyone who knows Poppi, Gupta, Nieradzik or ROAR's Sobel arm.

The framing the sweep supports, in the order the evidence is strong:

> Trivial and model-free attribution maps have been beating real explainers on removal-based faithfulness
> metrics for seven years, and each report diagnosed a different cause: the need to retrain (ROAR 2019),
> metric asymmetry (Poppi 2021, Opti-CAM 2024), class-information leakage through the mask (ROAD 2022),
> dataset centrality bias (Gupta 2022), out-of-distribution shift (Nieradzik 2024). None diagnosed the
> mask's spatial scale, and none built a control that holds mask geometry fixed. We show the artifact is
> monotone in mask scale, construct two independent geometry-matched controls — including a displaced copy
> of the explainer's own map, which has no precedent in this literature and a direct one in
> autocorrelation-preserving spatial null models — and show that with the corrected instrument the field's
> standard dataset is too small to resolve degradation under shift at all.

That puts claim 3 and claim 4 in front, where the novelty is, and demotes claim 1 from discovery to
mechanism. It also makes the paper harder to dismiss: the ladder becomes the evidence that discriminates
between six competing published diagnoses, which is a stronger and more interesting role than being a
headline number.

Claim 2 should be a section, not a headline. The honest version is short and good: a known failure of
removal metrics has a provable cause in how the control is built, and this project paid for the
demonstration.

---

## 7. Not verified

Listed so nobody mistakes a partial check for a complete one. Nothing here supports a verdict above.

- **Full text not reached — paywalled (HTTP 403):**
  - *FRIES: Framework for inconsistency estimation of saliency metrics*, Pattern Recognition, 2025
    (`sciencedirect.com/science/article/pii/S0031320325007964`). Surfaced in search as reporting high
    variance in fidelity ranks for the same map and model under different perturbations — relevant to
    claim 4 and to the imputation axis. Metadata only; do not cite until read.
  - Burt et al. 2020, *Generative modeling of brain maps with spatial autocorrelation*, NeuroImage
    (`pii/S1053811920305243`). The **bioRxiv preprint of the same work was fetched and is what §3.3 quotes**;
    the journal version was not reached, so cite the journal version only after checking the quoted
    sentences survive into it.
  - Poppi et al. 2021, CVF Open Access PDF (403). **The arXiv version (2104.10252) was downloaded and read
    in full**, and all Fake-CAM quotes come from it.
- **Metadata verified via search only, full text not reached:**
  - Alexander-Bloch et al. 2018, *On testing for spatial correspondence between maps of human brain
    structure and function*, NeuroImage 178:640–651, DOI 10.1016/j.neuroimage.2018.05.070 — the original
    spin test. Recommended as a citation in §3.3 but **not read**; the autocorrelation-preserving-null
    argument in §3.3 rests on Burt et al., which was read.
  - Miró-Nicolau et al., *A comprehensive study on fidelity metrics for XAI*, Information Processing &
    Management 62(1):103900, 2025. Only the related arXiv 2311.01961 was read.
  - Haufe et al. 2026 — cited inside Zullich et al. 2026 as a broad XAI-shortcomings perspective. Not
    located independently; no identifier confirmed. **Do not cite.**
  - Hedström et al., *Quantus*, JMLR 2023 — referenced but not fetched. MetaQuantus (2302.07265) was
    fetched.
  - Krishna et al. 2022, *The disagreement problem in explainable machine learning* (arXiv 2202.01602) —
    seen only as a reference inside Barr et al. Not fetched.
- **Partially read:**
  - Hsieh et al. 2026, AIM (arXiv 2605.16905). Read via HTML fetch covering the first 100,000 of 116,403
    characters; abstract, confound definition and control set were read, the final ~16k (late appendix) was
    not.
- **Coverage assumptions, not findings:**
  - Non-English venues, theses, and medical-imaging reliability literature were not swept. Judged low risk
    for claims 3 and 4; **moderate risk for claim 1**, since the medical-imaging community uses
    low-resolution CAMs on centred anatomy and may have reported the scale effect as a localization
    caveat.
  - Forward-citation lag. Papers posted in the last few weeks may not be indexed. Re-run the claim-1 and
    claim-3 searches immediately before submission, not only before the freeze — this is P1 §5 residual
    risk 2 and it applies here with more force, because two of the closest papers found
    (arXiv 2605.14641, 2605.16905, 2610.12493) are from the last six months.
