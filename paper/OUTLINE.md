# Four Artifacts in Causal-Effect Measurement of SAE Features

Working outline. Numbers are from `sae_rare.csv`, `controls2.py`, `unembed_align.py`,
`joint_control.py`, `checks3.py`, `conditioning.py`, `geometry.py`.
Pending: soft-frozen eval (`eval_saes.csv`), nine-dictionary sweep (`multi_dict.csv`).

---

## Claim

Causal-effect measurements on SAE latents are inflated by four distinct artifacts.
Each has a mechanism, each admits a control, and none is controlled in current
practice. After controlling all four, the trained-versus-random gap on Gemma Scope
is **1.4-1.9x**, not the 2.1x an uncontrolled measurement reports -- and the heavy
tail that an uncontrolled measurement would headline is **one latent in 240**.

This explains mechanistically why sanity checks in this literature keep returning
ambiguous verdicts: the confounds all push the same way, and the statistic
everyone reports is a mean over a distribution whose shape is set by them.

## SCOPE -- what the four artifacts do and do not apply to

**They apply to ablation-based causal-effect metrics**: zero or subtract a latent's
contribution, read a KL or logit difference. That family is ubiquitous in
feature-importance and circuit work, which is why the catalogue matters.

**They do NOT transfer directly to RAVEL.** RAVEL (2402.17700) uses interchange
interventions -- swapping activations to induce counterfactual states -- and scores
via cause/isolation ratios of behavioural outcomes, not KL magnitudes. Perturbation
-norm normalisation has no direct analogue there. So the paper must NOT claim to
overturn the Sanity Checks RAVEL result (0.72 vs 0.73); it explains inflation in a
different metric family.

**And RAVEL appears to be among the more reliable SAEBench metrics.** "Are Sparse
Autoencoder Benchmarks Reliable?" (2605.18229) finds only sae-probes,
reconstruction metrics and RAVEL-disentangle discriminate meaningfully once
training-seed noise is accounted for. So their near-tie cannot be dismissed as
metric noise, and this paper should not try.

Correct claim boundary: *ablation-based causal-effect measurement of SAE latents is
confounded in four ways, all pushing the same direction.* Narrower than "SAE
evaluation is instrumented wrong", and defensible.

### Do the artifacts transfer to interchange interventions? Mostly yes.

Do NOT write "use interchange instead" -- the argument fails analytically for
single-latent interventions. For latent f with decoder direction d_f:

| intervention | perturbation applied to the residual stream |
|---|---|
| ablation | `-a_f * d_f` |
| interchange (swap a_f for a_f' from a source prompt) | `(a_f' - a_f) * d_f` |
| clamping to c | `(c - a_f) * d_f` |

**All three perturb along the same direction d_f.** Direct-path alignment is a
property of d_f -- `std_v(W_U . d_f)` -- so it scales the output response of all
three identically, differing only in the scalar. Artifact 4 therefore transfers to
interchange **by construction**, not as a possibility to check.

Artifact 3 transfers for the same reason: a near-duplicate latent g leaves the
concept partly represented whichever way f is intervened on. Artifact 1 transfers
to anything pooling over positions. Artifact 2 transfers to any magnitude readout
and is *attenuated* -- not removed -- for behavioural rate scores like
cause/isolation, since larger perturbations still flip the target more often.

**Qualification.** RAVEL applied to SAEs may intervene on many latents at once, and
for multi-latent interventions the aggregate perturbation is a combination in which
any single latent's alignment matters less. The transfer argument is strongest for
the per-latent measurements that feature-importance work uses -- which is the
target literature anyway.

### MAGNITUDE vs RATE readouts -- this is the axis that matters, not ablation vs interchange

The transfer argument above holds for MAGNITUDE readouts (KL, logit difference,
probability change), where the response scales with alignment. It does NOT hold
fully for RATE readouts (attribute flipped, argmax changed, accuracy delta), because
a rate is a threshold crossing and thresholds **saturate**: once the intervention
flips the answer, additional alignment buys nothing. Alignment then acts only
through *which* latents cross, not *how far past* they land. The confound is
compressed, not eliminated.

**RAVEL is a rate readout -- verified, not inferred.** The paper defines
`Disentangle(A,F,M,D) = 1/2 [ Cause(A,F,M,D) + Iso(A,F,M,D) ]`, discusses
`Cause = 1` and `Iso = 1` as the attainable ideal, and the appendix states outputs
are scored by comparing "model outputs against the ground truth attribute values"
with relaxed match rules (+-2 for latitude/longitude, transcription variants for
pronunciation). These are match-rate accuracies.

That is consistent with 2605.18229 finding RAVEL-disentangle among the few
SAEBench metrics that discriminate meaningfully once seed noise is accounted for:
**rate readouts may survive reliability auditing precisely because saturation
compresses the confounds catalogued here.**

### THE RECOMMENDATION, narrowed and supported

> Prefer rate-based readouts for causal claims about SAE latents. Magnitude
> readouts are dominated by direct-path unembedding alignment.

Not "use interchange" -- that was wrong, since interchange perturbs along the same
d_f. The axis is magnitude vs rate.

**The test is one extra column, already added to `multi_dict.py`:** `flip_t{0,1,3}`
(argmax changed) and `dtop_t{0,1,3}` (probability drop of the original top token)
computed from the same forward passes as `kl_t{0,1,3}`. Then compare
rho(alignment, KL) against rho(alignment, flip) on the same latents.
- rho_KL ~ 0.375 and rho_flip ~ 0.1 -> recommendation holds, quantified.
- both ~ 0.375 -> recommendation dies and the catalogue is the whole paper.

### THE HOOK (replacing the Sanity Checks framing, which is now out of scope)

The paper opened by promising to explain why sanity checks come back ambiguous --
but that ambiguous result is RAVEL, now out of scope. The replacement is to name
the literature that actually uses ablation-based per-latent effect:
feature-importance ranking, circuit discovery, attribution-patching validation,
ablation-based faithfulness checks.

> The standard way of measuring what an SAE latent does is confounded in four ways.

Better than the benchmark-number framing because it targets a method used daily
rather than one reported figure. **Requires naming three or four specific papers
whose conclusions would move** -- the same hour of reading as the audit table, and
the claim should not appear in prose until those papers are named.

---

## THE PROTOCOL (the part that gets reused)

The catalogue is the finding; this is the takeaway. Boxed, five lines:

1. **Drop special-token positions** before any pooled statistic. Including position
   0 moved explained variance from 0.863 to -3.5.
2. **Report effect per unit perturbation norm**, not raw. Magnitude varies 4x
   across frequency deciles and flips the sign of the frequency conclusion.
3. **Report the per-latent max|cos| distribution**, and adjust or match on it.
   Note that matching is infeasible against a random control -- no common support.
4. **Report unembedding alignment** and give both estimates: aligned (the
   "do latents matter" question) and alignment-controlled (the "for non-trivial
   reasons" question).
5. **Report the jackknife curve**, not the mean or the variance ratio. One latent
   in 240 moved our variance ratio from 42x to 7x.

## Setup

Gemma-2-2B, residual stream layer 12, Gemma Scope JumpReLU 16k / L0 82 (the same
model and layer as Sanity Checks, 2602.14111). Causal effect = KL(clean ||
ablated) on the next-token distribution, ablating one latent's contribution at its
own firing position. Random control: tied, untrained, unit-norm directions, full
16384 width, top-k gated at matched L0. **Harsher than the paper's baselines** --
their frozen-decoder arm still trains the encoder, and their best causal-editing
arm (soft-frozen, tau=0.8) trains the encoder and lets the decoder move within
cosine 0.8 of init. State this asymmetry: trained beating my control is a weaker
result than beating theirs; trained failing to beat mine would be stronger.

---

## Artifact 1 -- BOS reconstruction

**Mechanism.** Position 0 carries a massive activation the SAE does not
reconstruct: cosine 0.44 there against ~0.93 at every other position. Its residual
norm dominates the pooled variance.

**Effect.** Including position 0 gives explained variance **-3.5**. Excluding it
gives **0.863** at mean L0 85.4, matching the released spec.

**Control.** Drop special-token positions before any pooled statistic. Trivial,
and it silently invalidates everything downstream if missed.

## Artifact 2 -- perturbation magnitude

**Mechanism.** Ablating a latent perturbs the residual stream by `a_f * d_f`.
Activation magnitude varies systematically with firing frequency, so raw effect
sizes compare interventions of unequal size.

**Effect.** Median perturbation norm runs **3.60** in the rarest frequency decile
to **14.76** in the most frequent. Uncontrolled, the rarest decile appears *worse*
than random (ratio 0.54, p=0.039). Per unit norm it is *better* (1.65, p=0.011).
The sign of the conclusion flips.

**Control.** KL per unit perturbation norm throughout.

**Corollary worth reporting.** Firing frequency predicts causal mass in neither
arm (trained rho=-0.047 p=0.46; random +0.041 p=0.57). The apparent
frequency-dependence in the literature's framing is magnitude, not frequency.

## Artifact 3 -- near-duplicate structure

**Mechanism.** Feature splitting produces decoder rows at high mutual cosine.
Split latents share a concept, so ablating one perturbs a direction the model
uses redundantly.

**Effect.** Per-latent max|cos| predicts causal mass in the trained arm
(rho=+0.220, p=0.0006) and not in the random arm (+0.016, p=0.82). Adjusting moves
2.14x -> **1.79x**.

**Control.** One-sided adjustment, justified by the asymmetry. Note explicitly:
symmetric residualisation is *wrong* here -- fitting the random arm gives slope
+2.506 on a predictor with p=0.82, i.e. residualising against noise. Also note
that joint matching on max|cos| is **impossible**: a random dictionary structurally
cannot have near-duplicate rows, so the supports do not overlap (trained median
0.272 vs random ceiling ~0.10).

## Artifact 4 -- direct-path unembedding alignment

**Mechanism.** A trained decoder direction is aligned with directions the model
writes to the residual stream, and those feed the unembedding. A random direction
is not. Next-token KL therefore rewards latents that short-circuit the network,
independent of computational role.

**Measure.** `align(d) = std_v(W_U[v] . d)` -- the spread, not the norm, since a
uniform logit shift leaves the softmax unchanged.

**Effect.** Strongest covariate found. rho=+0.375 (p<1e-4) trained, -0.063
(p=0.38) random. Adjusting moves 2.14x -> **1.42x**. Robust to functional form:
degree 1/2/3 give 1.42 / 1.38 / 1.41 while the condition number rises from 2.2e2
to 4.4e6, and R2=0.138.

**Exhibit -- feature 14119.** The single most causally powerful latent in the
sample. Causal mass 0.0807 against 0.0066 for the runner-up: **12x**. Logit-lens
top tokens: `'k' 'ak' 'nk' 'ik' 'zk' 'ink' 'kk' 'bk' 'Ck'`. Not the sink latent
(cosine 0.017 to the BOS embedding, -0.004 to b_dec). Alignment at the 87th
percentile, max|cos| 0.335.

> A latent whose entire causal footprint is writing the letter *k* to the logits
> is not what anyone means by an interpretable feature carrying computation.
> Controlling for unembedding alignment is controlling for triviality, not for
> mechanism.

**Confound or mechanism -- argue it, do not assume it.** If alignment is what a
good feature *has*, controlling for it removes the channel by which trained
dictionaries are better and 1.4x understates them. Two claims, two numbers:
- "SAE latents carry more causal effect than random directions" -> **2.14x**
- "...for reasons beyond pointing at the output" -> **1.4-1.9x**
Feature 14119 is the argument that the second is the interesting question.

---

## The tail is one latent

| drop top-k | sd ratio (raw) | sd ratio (adjusted) | top-5% mass (adj) | median ratio (adj) |
|---|---|---|---|---|
| 0 | 42.1x | 16.4x | 50.3% | 1.56 |
| **1** | **7.2x** | **3.6x** | 28.7% | 1.55 |
| 3 | 5.8x | 2.5x | 23.2% | 1.54 |
| 10 | 2.6x | 1.7x | 18.2% | 1.50 |

Removing one latent of 240 takes the variance ratio from 42x to 7x. A bootstrap
resamples it in and out but still treats it as a population draw; the jackknife
says the statistic is not estimating anything population-level. The median claim
is untouched throughout (1.56 -> 1.50).

**Report the bootstrap instability as part of the argument.** At n=198 the random
arm's p99 has a 95% CI spanning **11.3x** (raw) and the headline
"fraction above p99" runs [4.2%, 39.6%]. Published aggregate comparisons on a few
hundred latents cannot distinguish anything.

## Endpoint

| estimate | design | value |
|---|---|---|
| **primary** | matched-support, no model | **1.93x [1.50, 2.29]**, n=180 |
| sensitivity | alignment-adjusted | 1.42x [1.19, 1.67] |
| sensitivity | joint (max\|cos\| + alignment) | 1.56x [1.31, 1.87] |
| uncontrolled | raw | 2.14x [1.72, 2.45] |

**Estimand, stated precisely.** 1.93x is *the trained-versus-random causal-effect
ratio among latents whose unembedding alignment lies within the random arm's
support* -- not the unconditional ratio. Two properties to state rather than bury:

1. The matched subset **excludes feature 14119 by construction**, because its
   alignment (0.0431) exceeds the random arm's maximum (0.0418). That is a
   fortunate property of the design, not a choice. The estimate is tail-free
   without any pruning, and barely moves on further pruning (1.88 dropping top-1,
   1.84 dropping top-3).
2. Matching **under-controls** alignment: trained median inside the window is
   0.0350 against the random median 0.0338. So matched-support is biased *upward*
   relative to full adjustment. This is why the honest answer is a bracket
   1.42-1.93 rather than a point -- the two designs differ in how completely they
   control the confound, and they bound it from either side.

## Joint fitting caveat

Under joint fitting, max|cos| flips sign: +0.241 alone, **-0.291** with alignment
in the model. Spearman(max|cos|, alignment) = +0.488, p<1e-4. So "adjust for both"
is not additive control -- it is a fit with collinear covariates. Report the slopes
so the shift is visible.

## Decoder geometry -- as a control, appearing once

Mean-over-*pairs* |cos| does not separate trained from random (0.0193 vs 0.0166,
chance 0.0166), and arguably cannot: 16384 unit vectors in 2304 dimensions are
near-orthogonal on average whether trained or not, which is what an overcomplete
dictionary that spans the space should look like. Mean-over-*features*
nearest-neighbour |cos| does separate: **0.3095 vs 0.0860**, and it is also a mean.

The lesson is not that means are uninformative but that **the informative quantity
is per-latent and standard evaluation aggregates before looking at it**.

Geometry is *the same quantity* as Artifact 3's max|cos|, so it appears as the
control that rules out the splitting explanation -- never as independent
corroboration for the tail it is controlling.

**Do not over-claim spectral concentration.** Stable rank 0.069 (~159 of 2304
dimensions) is a sigma_1-sensitive statistic and sigma_1/sigma_2 = 1.07, so no
single direction dominates. Participation ratio is the right instrument:
**0.510 vs 0.877**, i.e. **1.7x** concentration, not 8x. Survives centering
(0.5116) despite a shared direction 3.4x the random level, so it is not a
mean-offset artifact.

---

## Pending, and what each decides

**Soft-frozen comparison** (matched budget, 12M tokens, sequential arms, TopK
k=82, decoder projected to cosine 0.8 of init after every step). This is the
comparison that speaks to the published result; the 1.93x above is against an
untrained control and says nothing about it. Both outcomes pre-registered:

- *soft-frozen ~ trained*: "four artifacts inflate SAE causal-effect measurement;
  after controlling all four, trained SAEs beat an untrained dictionary by
  1.4-1.9x but do not beat a soft-frozen baseline." Sharpens Sanity Checks --
  their near-tie is real, and the small margins in the literature are further
  inflated by artifacts they did not control.
- *trained > soft-frozen after controls*: "the near-tie is an artifact of
  aggregate statistics; the advantage survives at Xx once magnitude, duplication
  and alignment are controlled." Stronger, and needs the sweep to support it.

The artifact catalogue is the contribution either way, which is why the paper does
not depend on which side this lands.

**Nine-dictionary sweep** -- layers 12/19, widths 16k/65k/131k, L0 22-445, 300
latents each. Turns "in this dictionary" into "in nine." Two questions: is a
dominant latent generic (max/median within a fixed n=300 sample, comparable across
dictionaries), and does the tail survive the **direct-path decay control** (KL read
at t, t+1, t+3 from the same logits; at t+3 the ablated token's own identity
contributes nothing directly). Report absolute magnitudes at each offset -- if both
arms collapse toward the numerical floor at t+3, that arm is underpowered rather
than informative.

## Candidate artifact 5 -- position in sequence

**Mechanism.** Effects are measured at each latent's top-activating positions.
Late positions carry more context and a sharper next-token distribution, so KL is
not comparable across them. If trained latents systematically fire later than
random directions do, that is a fifth confound in the same direction.

**Status: NOT YET TESTED.** `sae_rare.csv` does not record position -- an omission,
so the check cannot be run retrospectively on the primary sample. `multi_dict.py`
now records `pos`, `rel_pos` and `act`, so the nine-dictionary sweep answers it.
One Spearman per arm plus the position distributions by arm. If null, one sentence
saying it was checked; if not, it is artifact five.

## LITERATURE AUDIT -- is each artifact really uncontrolled?

The sentence "none is controlled in current practice" is an assertion about the
literature, not a measurement, and it is load-bearing. One paper that normalises by
intervention magnitude discounts the framing. Table to complete from methods
sections, not abstracts:

| paper | drops pos 0? | normalises by intervention magnitude? | controls duplication? | accounts for direct path? |
|---|---|---|---|---|
| Sanity Checks (2602.14111) | excludes dead latents (freq<1e-6); position handling TBC | TBC | no (it is their subject, not a control) | TBC |
| SAEBench (2503.09532) | TBC | metrics are behavioural ratios, likely N/A | TBC | TBC |
| RAVEL (2402.17700) | N/A -- interchange, not ablation | N/A | TBC | TBC |
| Are SAE Benchmarks Reliable? (2605.18229) | not addressed | not addressed | not addressed | not addressed |
| 2-3 recent causal-editing papers | TBC | TBC | TBC | TBC |

If one artifact turns out to be controlled somewhere, the claim becomes "three of
four" and the paper survives -- but the table has to exist before the claim does.

## STRONGEST AVAILABLE VERSION -- apply the protocol to someone else's data

Applying the corrected protocol to a released per-latent effect dataset would beat
applying it to our own measurements. Check whether SAEBench, RAVEL, or any recent
causal-editing paper releases **per-latent** effect measurements rather than
aggregates. SAEBench open-sources 200+ SAEs but that is weights, not per-latent
effects.

If one does and the four controls change its reported conclusion, that is a
headline rather than a catalogue. If none does, say so: per-feature effect data not
being released is part of why these artifacts persist.

## Related work

- **Sanity Checks for SAEs (2602.14111)** -- the result being explained. Their
  baselines, coverage, and the synthetic long-tail finding they state they cannot
  explain. Also: lazy training dynamics are *theirs*, reported as decoder cosines
  concentrated near 0.8 after 5-10% of training. Do not reclaim it.
- **Dissecting Chronos (2603.10071)** -- heavy-tailed SAE feature importance
  (max/median up to 30.5x) in time-series foundation models. The heavy tail as a
  phenomenon is not ours; the claim here is about what it does to *evaluation*.
- **Toward Identifiable SAEs (2605.31245)** -- Theorem 3.2 is an *existence*
  result about codes given a dictionary, paired with a positive result (3.6) under
  approximate RIP. Not an impossibility about atomicity.
- **Open Problems in Mech Interp (2501.16496)** -- "sparsity isn't a reliable
  proxy for interpretability"; "SDL ignores feature geometry"; "unclear whether the
  model doesn't use the concept or SAE training is inadequate." The last is the
  question this paper answers with "neither -- the dictionary contains both and the
  measurement conflates them."

## Limitations

One model, one layer for the primary analysis (the sweep addresses this). One
causal metric -- next-token KL under single-latent ablation, not RAVEL, so no
direct comparison to their 0.72/0.73. n=240/198 for the primary; the bootstrap
intervals are wide and reported. Matched-support under-controls alignment; the
regression over-controls if alignment is mechanism. The random control is untrained
and therefore weaker than the paper's.
