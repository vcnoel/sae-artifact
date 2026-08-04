# Pre-registration log

Predictions and decision rules, written before the corresponding run wherever
possible. Entries marked **[RECONSTRUCTED]** were agreed in conversation before
the run but written into this file afterwards, on 2026-08-03, when the project
was moved out of a temp directory. They are marked because a log whose entries
might postdate their results is worth nothing; only entries marked **[LIVE]**
were written to disk before the data existed.

Many readings reversed during this session. Every reversal is recorded here
rather than silently corrected, including three where the first reading was mine
and wrong.

---

## E1 — Does the trained/random SAE gap open on rare features? **[RECONSTRUCTED]**

**Motivation.** Sanity Checks (2602.14111) reports trained SAEs barely beating
frozen baselines on aggregate causal editing (RAVEL 0.72 vs 0.73), and in
synthetic settings reports SAEs "capture almost exclusively the highest-frequency
features, leaving over 90% of the ground-truth dictionary ... unmatched". They did
not stratify the real-model comparison by frequency.

**Prediction.** The trained-vs-random advantage grows as feature frequency falls;
the aggregate null is dominated by frequent features.

**Outcome: FALSIFIED, twice over.**
- Raw KL suggested the *opposite* (rarest decile ratio 0.54, p=0.039), but that
  was perturbation magnitude: median perturbation norm runs 3.60 in the rarest
  decile to 14.76 in the most frequent. Per unit norm the rarest decile favours
  trained (1.65, p=0.011).
- Frequency predicts causal mass in neither arm (trained rho=-0.047 p=0.46;
  random +0.041 p=0.57). There was never a frequency effect to find.

**Consequence.** The frequency stratification became vestigial, and later turned
out to be the direct cause of a bracket in the endpoint (see E6).

---

## E2 — Is the heavy tail a population phenomenon? **[RECONSTRUCTED]**

**Kill rule, set in advance.** The variance claim requires surviving removal of
the top 1% of latents. If dropping a handful collapses it, the statistic is not
estimating anything population-level.

**Outcome: KILLED.** Dropping ONE latent of 240 took the sd ratio from 42.1x to
7.2x (16.4x to 3.6x after adjustment). Feature 14119 carried 12x the causal mass
of the runner-up. Top-5% mass 65.2% -> 28.7%. The median ratio was untouched
throughout (2.14 -> 2.06).

Also recorded: at n=198 the random arm's p99 has a 95% bootstrap CI spanning
11.3x, and the headline "fraction above p99" ran [4.2%, 39.6%]. The claim did not
survive its own error bar.

---

## E3 — Lazy training dynamics **[RECONSTRUCTED]**

**Initial reading.** At 0.8M tokens the unconstrained decoder sat at mean cosine
0.910 to init vs 0.912 for the constrained arm, suggesting the constraint never
binds.

**Outcome: RETRACTED, for two independent reasons.**
1. It is not novel — Sanity Checks tests the hypothesis directly and reports
   cosines "concentrated near 0.8" after 5-10% of training. The soft-frozen
   baseline exists *because* of it. I misread their hedge ("whether lazy dynamics
   *fully* explain") as an open question.
2. At 12M tokens my decoder reached median cosine 0.231 to init, contradicting
   the phenomenon. But their measurements sit at ~25M and ~50M tokens (5-10% of
   500M), i.e. MORE tokens with LESS movement. Less training producing more
   movement is backwards, so this is a setup difference, not a discrepancy with
   their result. Likely causes: isotropic init (theirs also tests `cov` init
   estimated from real activations, which starts inside the activation subspace)
   and my unit-norm renormalisation after every step, which projects out the
   radial gradient and forces all update signal into rotation.

**Not reportable as a discrepancy.** Recorded so it is not re-derived.

---

## E4 — Soft-frozen comparison: two outcomes fixed in advance **[RECONSTRUCTED]**

Both framings were fixed before `eval_saes.py` ran, because the artifact
catalogue is the contribution either way:

- *soft-frozen ~ trained*: four artifacts inflate ablation-based causal-effect
  measurement; after controlling all four, trained beats an untrained dictionary
  but not a soft-frozen baseline. Sharpens Sanity Checks.
- *trained > soft-frozen after controls*: the near-tie is an artifact of aggregate
  statistics; the advantage survives at Xx.

**Outcome: NEITHER, cleanly.** Hodges-Lehmann 1.209, 95% CI [1.032, 1.424],
MW p=0.0216 — a small but reliable advantage. My first reading ("CI includes 1, no
reliable advantage") used a ratio-of-medians CI against a Mann-Whitney test, i.e.
an interval from one framework and a test from another. HL is that test's own
point estimate and its interval excludes 1.

Reconstruction: EV 0.830 vs 0.788 at eval, 0.755 vs 0.680 at end of training.
The EV gap is **widening** (OLS slope +0.0064 per M tokens, p<1e-4), so the
undertraining objection to any decoder claim stands and such claims must be scoped
to 12M tokens.

---

## E5 — Is unembedding alignment a confound or the mechanism? **[RECONSTRUCTED]**

The bad-control question. Alignment predicts causal mass at rho=+0.375 (Gemma
Scope) and +0.264 (my trained arm), with the control arm null in both cases
(-0.063, +0.004).

**Reading 1 (confound).** The metric rewards output-adjacent directions; trained
decoders drift toward them. Exhibit: Gemma Scope 14119, whose entire causal
footprint is writing the letter *k* (`'k' 'ak' 'nk' 'ik' 'zk' 'ink'`), 12x the
runner-up's causal mass, 90th alignment percentile, NOT the sink latent (cos 0.017
to the BOS embedding).

**Reading 2 (mechanism).** Finding directions the model uses to write output is
what decoder training is for; adjusting removes the treatment.

**Outcome: BOTH, depending on the sample — and that is the finding.** Wordlike
fraction (alphabetic, length>=3) of the top-20 latents by alignment:

| dictionary | population | wordlike median |
|---|---|---|
| gemma_scope | all 16384 | **0.10** |
| gemma_scope | frequency-stratified 240 | **0.90** |
| mine_trained | all 16384 | 0.70 |
| mine_trained | frequency-stratified 240 | 1.00 |

Same dictionary, same covariate, opposite character. The stratified 240 sit at
median alignment percentile ~49-54%, so they miss the extreme-alignment latents
entirely. **Both** of my alignment-adjusted numbers (Gemma Scope 2.14->1.42 and
frozen 1.209->0.992) were fit on that population and inherit the ambiguity.

Also retracted here: my claim that the trained arm's high-alignment latents are
semantic. That was the top-20 of the stratified sample, not of the dictionary.
Dictionary-wide the top-20 is mixed (wordlike median 0.70), not the
governor/agreement/achievement set I reported.

---

## E6 — Uniform re-sample **[LIVE — written before the run]**

**Why.** The frequency stratification is vestigial (E1) and is the direct cause of
the endpoint bracket (E5). A uniform random sample over live latents gives one
population and one answer, and repairs both adjustments rather than only one.

**Prediction.** With a uniform sample, the top-alignment latents will be junk —
wordlike fraction ~0.1-0.3, closer to the dictionary-wide figure than to the
stratified 0.90-1.00. Alignment adjustment is then licensed as a triviality
control, and **both adjusted numbers should come out LOWER than their stratified
versions** (Gemma Scope below 1.42; frozen below 0.992).

**Decision rule.** If the uniform sample's high-alignment latents are also
wordlike (median > 0.6), the triviality reading is wrong across the board.
Artifact 4 then reduces to "alignment predicts causal mass" with no triviality
interpretation available, and the alignment-adjusted numbers must be dropped in
favour of raw ones.

**Kill.** If the uniform HL for trained-vs-frozen includes 1 both raw and
adjusted, there is no soft-frozen advantage to report in either direction, and E4
resolves to the first of its two framings.

**Secondary, same run.** The missing cell — my trained arm vs an untrained tied
random dictionary on the same uniform feature list — converts the decomposition
(2.14 / 1.42 / 1.21) from three numbers across two experiments into one
dictionary measured three ways.

### E6 OUTCOME (run 2026-08-03 08:59-09:20)

**The prediction FAILED and the decision rule fired.**

Predicted: uniform sampling would put the top-alignment latents at wordlike
0.1-0.3. Observed: **0.90** in the uniform Gemma Scope sample and **0.90** in my
trained arm — identical to the stratified sample, not the dictionary-wide 0.10.

**Why the prediction was wrong.** The problem was never stratification, it was
sample SIZE. The junk latents are roughly the top 20 of 16384 (top 0.12%). The
top 20 of a 240-sample is the top 8% of that sample, which lands near the 92nd
dictionary percentile — nowhere near the extreme tail. No sampling *scheme* at
n=240 can reach those latents. E5's population-dependence result stands, but the
fix I proposed does not address it.

**Decision rule, as pre-registered:** wordlike median > 0.6 means the triviality
reading is unavailable, so the alignment-adjusted numbers are dropped in favour of
the raw ones. Honoured below.

**Endpoints (raw HL, primary; adjusted shown for completeness):**

| comparison | raw HL | 95% CI | adjusted HL |
|---|---|---|---|
| Gemma Scope vs untrained tied (uniform) | **2.269** | [1.934, 2.638] | 1.795 |
| my trained vs untrained tied (same list) | **2.104** | [1.830, 2.415] | 1.981 |
| my trained vs soft-frozen (same list) | **1.192** | [1.032, 1.399] | 1.132 (incl. 1) |

The decomposition is now one dictionary, one feature list, three controls:
**2.104x over an untrained dictionary, 1.192x over soft-frozen.** Most of the
advantage is the encoder; decoder freedom buys 1.19x.

**Unpredicted and worth reporting.** Gemma Scope (JumpReLU, >=500M tokens) and my
arm (TopK, 12M tokens) give near-identical advantages over an untrained control,
2.269x vs 2.104x. Roughly 40x more training data buys almost nothing on this
metric, which sits alongside E4's finding that the reconstruction gap widens
monotonically while causal effect does not.

**Alignment replicates in the uniform sample:** rho=+0.341 (Gemma Scope),
+0.262 (mine), against +0.375 and +0.264 stratified.

**Artifact 5 is not a threat to the endpoint:** rho(kpn, rel_pos) = -0.084,
p=0.196 on the endpoint sample (it was pooled-significant only across 2700
sweep latents).

---

## E7 — Nine-dictionary sweep **[RECONSTRUCTED, run in progress at time of writing]**

Layers 12/19, widths 16k/65k/131k, L0 22-445, 300 latents each.

**Tail generality.** Criterion: max/median within a fixed n=300 sample,
comparable across dictionaries. "Every SAE shows ~10x" is a phenomenon; it does
not require identifying the true argmax.

**Direct-path decay.** KL read at t, t+1, t+3 from the same logits. A tail present
at t and absent at t+3 is a direct-path artifact. Report absolute magnitudes — if
both arms collapse toward the numerical floor at t+3, that arm is underpowered
rather than informative.

**Saturation / rate-vs-magnitude.** Prediction: alignment predicts KL at
rho ~ 0.375 and the argmax-flip rate at rho ~ 0.1, because a rate is a threshold
crossing and saturates. If both are ~0.375 the recommendation dies.
**Required control:** threshold KL at whatever cutoff reproduces flip's own
positive rate, so both readouts are binary with identical marginals. Otherwise any
difference is information loss, not saturation.

**Position (candidate artifact 5).** Untestable retrospectively — `sae_rare.csv`
omits position, my error. `multi_dict.py` records `pos`, `rel_pos`, `act`. One
Spearman per arm plus position distributions. If it matters, the endpoint
inherits an uncontrolled confound.

**Orthography generality.** Logit-lens each dictionary's top latent. If all nine
are orthographic/token-identity, the 14119 exhibit generalises. If heterogeneous,
it is an anecdote and the direct-path argument rests on the correlation alone.

---

## E8 — Artifact 4 reframed as a SENSITIVITY, not an artifact **[post-E6]**

E6's failure has a consequence. The junk latents are roughly the top 20 of 16384,
i.e. the 99.88th percentile. **No sample a paper would realistically draw reaches
them** — the top-20-by-alignment of a 240-sample lands near the 92nd dictionary
percentile and is 90% wordlike. So the triviality confound does not exist in
practice: at any realistic sample size, high-alignment latents are semantic.

Artifact 4 is therefore NOT "measurements are inflated by trivial features". It is:

> Ablation-based causal-effect measurements are sensitive to unembedding
> alignment at rho ~ 0.3, and within realistic samples that alignment reflects
> semantic content rather than triviality.

A property to report, not a confound to adjust away — which is what dropping the
adjustment already implies. **The catalogue is four artifacts plus one
sensitivity.** Weaker and correct.

## E9 — The layer-19 tension, and what falsified my resolution **[tested]**

Two results sit in apparent contradiction:
- t+3 decay is SHARPER at layer 19 (ratio 0.037-0.114) than layer 12
  (0.197-0.523): more of the measured effect sits at the firing position deeper in.
- Alignment predicts causal mass at layer 12 (mean rho +0.328) and NOT at layer 19
  (mean rho -0.045).

If layer 19 were more direct-path dominated, alignment should matter *more* there.

**Hypothesis tested and FALSIFIED.** I proposed a ceiling effect: at layer 19 nearly
every latent has a strong direct path, so alignment stops discriminating and the
correlation compresses. The variance data rules this out — mean CV(alignment) is
0.1175 at layer 12 vs 0.1091 at layer 19 (7% lower, trivial), and sd(log kpn) is
*higher* at layer 19 (0.660 vs 0.588). Ample variance in both variables; no
correlation appears.

**The propagation reading explains ONE HALF only.** Fewer remaining attention
operations at layer 19 means less opportunity for the perturbation to propagate
across positions, which accounts for the t+3 gradient without implying direct-path
dominance. But that reading makes the effect at t *more* locally determined at
layer 19 (89-96% of it is gone by t+3), so alignment should predict causal mass
MORE strongly there, not less. The alignment null is left exactly where it was.

**Correcting my own entry:** I first wrote this up as a post-hoc resolution. It is
not one. The honest disclosure:

> The t+3 gradient is consistent with fewer remaining attention operations rather
> than with greater direct-path dominance, and therefore does not support
> artifact 4. Why alignment predicts causal mass at layer 12 (rho = +0.33) and not
> at layer 19 (rho = -0.05) is **unexplained**; the compressed-variance hypothesis
> is falsified by equal alignment spread (CV 0.1175 vs 0.1091) and greater
> causal-mass spread (sd 0.588 vs 0.660) at layer 19.

An unexplained layer dependence is a fine thing to report. A wrongly-closed one is
not. The t+3 gradient must not be presented as supporting artifact 4 either way.

## E10 — Four-depth test **[LIVE — written before the run]**

Two depths cannot discriminate; Gemma Scope covers all 26 layers of gemma-2-2b, so
four can. Adding layers 5 and 24 to the existing 12 and 19.

**Prediction 1 (propagation), fixed in advance.** If propagation limits drive the
t+3/t0 ratio, it scales with REMAINING layer count: 21 remaining at layer 5, 14 at
layer 12, 7 at layer 19, 2 at layer 24. Observed so far: ~0.38 at layer 12 and
~0.10 at layer 19. Predicted: **layer 5 above 0.5, layer 24 below 0.02**, and the
four points monotone decreasing in depth.

**The L0 confound is BOUNDED, not merely flagged.** The sweep gives the
within-layer L0 dependence of the exact quantity E10 measures, at matched 16k
width:

| layer | L0 | t+3/t0 |
|---|---|---|
| 12 | 22 | 0.204 |
| 12 | 82 | 0.298 |
| 12 | 445 | 0.468 |
| 19 | 23 | 0.029 |
| 19 | 73 | 0.059 |
| 19 | 279 | 0.129 |

Monotone increasing in L0 at both depths. Power-law exponent: +0.276 at layer 12,
+0.594 at layer 19, **+0.406 pooled with a layer fixed effect**. The E10 grid spans
L0 68-82, a ratio of 1.206, so expected L0-induced movement is
1.206^0.44 = **8.5%** — against a predicted depth effect spanning 0.5 to 0.02, a
factor of 25 (2400%). **The confound is ~283x smaller than the effect under test.**

(Convention note: these ratios are ratio-of-medians. An earlier report used
median-of-per-feature-ratios and gave 0.360 rather than 0.298 at layer 12 / L0 82.
Same conclusions; one convention must be used throughout.)

**Prediction 2 (alignment) — there IS a naive prediction, and the data already
violates it.** The alignment measure is `sqrt(d^T C d)` with `C` the embedding
covariance, so it is **layer-independent by construction**. What it estimates is
"how much this direction would move the logits if it reached the output
unchanged". At layer 5 a direction passes through 21 more blocks before the
unembedding, so embedding-space alignment should be a POOR predictor of its real
logit effect; at layer 24 only two blocks intervene, so it should be a GOOD one.

So the naive expectation is **rho(kpn, alignment) increasing monotonically with
depth**. Observed: +0.33 at layer 12 and -0.05 at layer 19 — not merely absent
deeper in, but moving the *wrong way against the measure's own construction*.
That makes this arm sharper than "threshold or gradient":

- If layer 24 is strongly positive, layer 19 is a local anomaly and the measure
  behaves as constructed.
- If layer 24 is also null or negative, the measure is **not doing what its
  construction implies**, and that is a finding about the measure rather than about
  the model — and it would undercut every alignment number in this project.

**Falsification.** If layer 5 comes in below layer 12, or layer 24 above layer 19,
the propagation reading is wrong and the t+3 gradient needs a different account
entirely.

**Kill.** If t+3/t0 is flat across all four depths, both the propagation reading and
the direct-path reading of the gradient are wrong, and the layer-12-vs-19
difference observed so far is noise across dictionary configurations rather than a
depth effect.

## E10 OUTCOME (run 12:08-12:33)

**Convention audit: no mismatch.** `multi_dict.py` reports **feature-level**
(per-feature median, then ratio of medians): layer 12 / L0 82 = 0.360 in both the
sweep and E10, which is the convention the thresholds were calibrated on. The
0.298 figure came only from the L0-fit script, which used observation-level. The
L0 exponent +0.406 was therefore fitted on the *other* convention; the two differ
by a non-constant factor 0.97-1.35, so that exponent carries an unquantified
convention error. Feature-level is the convention for everything reported.

**Prediction 1 (monotonicity): CONFIRMED.**

| layer | remaining blocks | L0 | t+3/t0 | t+1/t0 |
|---|---|---|---|---|
| 5 | 20 | 68 | 0.384 | 0.521 |
| 12 | 13 | 82 | 0.360 | 0.531 |
| 19 | 6 | 73 | 0.078 | 0.120 |
| 24 | 1 | 73 | 0.021 | 0.037 |

Monotone decreasing across all four. Threshold misses recorded: layer 5 predicted
>0.5, observed 0.384; layer 24 predicted <0.02, observed 0.021. Direction and order
right, magnitudes wrong at both ends.

**POWER LAW: FALSIFIED. [POST-HOC — E10 had already run when this was formulated,
so it is not a pre-registered test.]** Exponent fitted from layers 12 and 19 alone
(L0-corrected) is 1.91, giving out-of-sample predictions 0.760 at layer 5 and 0.003
at layer 24. Observed 0.384 (ratio 0.51) and 0.021 (ratio 8.38) — both outside a
1.5x tolerance, in **opposite** directions. The observed curve is flatter than a
power law at both ends: flat from 20 to 13 remaining blocks, a knee between 13 and
6, flat again from 6 to 1.

**Consequence.** The propagation account survives only ORDINALLY. "Effect decays
with remaining attention operations" is supported; "decays proportionally to
remaining depth" is refuted. An earlier four-point log-log fit (+1.015, R2=0.949)
was in-sample across a wide range and concealed the out-of-sample failure; calling
it stronger than monotonicity was wrong.

**Prediction 2 (alignment): naive expectation VIOLATED, measure not invalidated.**

| layer | rho(kpn, alignment) | p |
|---|---|---|
| 5 | +0.358 | <1e-4 |
| 12 | +0.339 | <1e-4 |
| 19 | +0.031 | 0.59 |
| 24 | +0.145 | 0.012 |

Slope vs layer -0.0154 (p=0.19). The measure `sqrt(d^T C d)` is layer-independent
by construction and should predict best where fewest blocks intervene, i.e. deepest.
It predicts best at layers 5 and 12, where 20 and 13 blocks intervene. Layer 24 is
positive and significant, so the measure is not dead and the catalogue's alignment
numbers stand — but layer 19 is not a local anomaly: the shape is non-monotone
(+0.36, +0.34, +0.03, +0.15) with a mid-network dip. **Unexplained, now on four
points rather than two.**

## Standing scope constraint

The four artifacts apply to **ablation-based** causal-effect metrics (zero or
subtract a latent's contribution, read a KL or logit difference). They do not
transfer directly to RAVEL, which uses interchange interventions scored by
match-rate accuracies (`Disentangle = 1/2[Cause + Iso]`, outputs compared against
ground-truth attribute values with relaxed matching). The paper must not claim to
overturn the 0.72-vs-0.73 result.

Note also that the analytic transfer argument holds for *magnitude* readouts:
ablation, interchange and clamping all perturb along the same direction d_f, so
alignment scales all three identically. "Use interchange instead" is therefore
wrong; the axis is magnitude versus rate.

**Attribution not established:** TPP and SCR are the two SAEBench metrics that
fail reliability auditing (2605.18229), and they are the only ablation-based
metrics in that suite — but they are also the only *set*-ablation metrics, and no
metric there ablates a single latent. The two properties are perfectly confounded,
so their failure cannot be attributed to the construction these confounds apply
to. The hook still needs the audit table.

## E11 — Six arms, one seed, real degrees of freedom **[RECONSTRUCTED — process
lapse noted below]**

**Process note, logged rather than hidden.** This experiment was launched
directly from the user's specification (decoder free / tau=0.80 / tau=0.90 /
lr=1e-4 / k=41 / order=1, all seed 0) without a prediction written to this file
first. The two numbers that function as predictions here — the 2-arm
extrapolation's Erho2=0.455 and "5 arms needed for Erho2=0.8" — were already on
record from the prior (E-unlabeled) 2-arm decomposition before this run started,
so they are reconstructed rather than fabricated after the fact, but the
discipline was not followed and that is a failure to note, not omit.

**Motivation.** The 2-arm crossed decomposition had 1 degree of freedom on the
arm factor; a G-theory reviewer discounts that on sight. Six arms varying real
fitting choices (decoder constraint, LR, k, data order) give the arm and
latent x arm components real replication.

**Prediction (reconstructed from the 2-arm run).** Erho2 near 0.455, ~5 arms
needed for Erho2=0.8, and the latent x arm : arm ratio should shrink from 23.6x
now that "arm" has real degrees of freedom instead of one difference.

**Outcome (run complete, `results/eval_arms.csv`, 240 shared latents x 6 arms x
up to 6 positions, 8492 rows).**

| component | 2-arm | 6-arm (direct) |
|---|---|---|
| latent | (not separated) | 16.9% |
| arm | 0.4% | 3.2% |
| latent x arm | 9.7% | 12.5% |
| residual (position) | — | 67.4% |
| latent x arm : arm ratio | 23.6x | 3.8x |
| Erho2 (one-arm design) | 0.455 | 0.415 |
| arms needed for Erho2=0.8 | 5 (extrapolated) | 6 (direct) |

**Confirmed:** the 2-arm extrapolation and the direct 6-arm estimate corroborate
each other closely (Erho2 0.455 vs 0.415; 5 vs 6 arms needed) — the headline is
not an artifact of having only one degree of freedom on "arm". Latent identity
explains only 16.9% of variance; arm, latent x arm, and position jointly explain
83.1%. That is the load-bearing number for "causal importance is not a property
of the latent."

**Falsified/revised:** the 23.6x latent-x-arm:arm ratio does NOT survive — it
falls to 3.8x. With only one degree of freedom, the 2-arm "arm" component was
almost certainly an underestimate (a single contrast can land anywhere by
chance); six real fitting choices give "arm" itself a much larger, and probably
more honest, share (3.2% vs 0.4%). The ratio was never the claim to lead with;
Erho2 and the "arms needed" figure are, and those held up. Report the ratio
change explicitly rather than quietly dropping the old number.

Pairwise Spearman across all C(6,2)=15 arm pairs: median +0.376, range
[+0.225, +0.574] — every pair positive, none above 0.6. Rank does transfer
partially, never fully; that is consistent with latent x arm being the largest
non-residual component.

**k41-sensitivity check.** `k41` alone restricts the shared live-latent
intersection to 12,080/16,384 (the other five arms each retain >16,000). Dropping
it and rerunning on the remaining 5 arms (223 latents x 5 arms x 6 positions):
latent 16.8%, arm 2.0%, latent x arm 12.6%, residual 68.6%, ratio 6.2x, Erho2
0.412, arms needed 6. Every figure matches the 6-arm run to within 1.2 points.
**The decomposition is not driven by the k41 arm's narrower dictionary.**

## D1/D3 — Literature audit and the same-data sign flip

**D3 (audit table).** Confirmed via primary-source PDF extraction (not
secondary summaries) that five published papers zero-ablate a single SAE
latent and read a magnitude: Bricken et al. 2023, Marks et al. 2024 (Sparse
Feature Circuits, KL divergence readout -- the closest methodological match to
ours), Templeton et al. 2024 (Scaling Monosemanticity), Gao et al. 2024
(Scaling and evaluating SAEs -- their "ablation sparsity" metric is a
different quantity from a magnitude-normalised effect and is labelled as such,
not folded into the same column), and Cho et al. 2025 (2607.20596, this
project's D0 target). None of the five report dropping special-token
positions, normalising by perturbation magnitude, or reporting near-duplicate
decoder structure in a way tied to the causal-effect measurement itself.

**Four candidates checked and excluded, with reasons on record (not silently
dropped):**
- Makelov, Lange, Nanda 2024 (2405.08366) -- read the full PDF directly. Edits
  2/4/6 features simultaneously on the IOI task; logit-difference readout but
  never single-latent. This is a DIFFERENT paper from the "Sanity Checks"
  paper `train_saes.py`'s docstring is built against -- an error caught before
  it reached the paper.
- Korznikov, Galichin et al. 2026 (2602.14111, the actual "Sanity Checks"
  paper) -- read the full PDF directly. Their "causal editing" metric (0.73
  frozen vs 0.72 trained) is the RAVEL framework's interchange-intervention
  match-rate accuracy, NOT zero-ablation with a magnitude readout. This
  paper's soft-frozen baseline construction ($\tau=0.8$) matches their
  Figure 1 baseline exactly; the causal-editing NUMBER is theirs and out of
  scope, consistent with the pre-existing Standing Scope Constraint re RAVEL.
- SAEBench TPP/SCR (audited by Chanin et al. 2605.18229) -- set-ablation (a
  small group of latents per concept), not single-latent; already on record.
- Rajamanoharan et al. 2024 (JumpReLU) -- "loss recovered" ablates the WHOLE
  residual stream and substitutes the SAE reconstruction; whole-dictionary,
  not single-latent.

**The same-data sign flip (feeds \S4 "One comparison, two conclusions").**
Rarest activation-frequency decile (bin 0) vs a frequency-matched decile of
the untrained tied-random control, identical latents, identical control, two
readouts:

| readout | HL | 95% CI | Mann-Whitney p |
|---|---|---|---|
| raw KL | 0.683 | [0.446, 0.962] | 0.039 |
| KL / unit perturbation norm | 1.687 | [1.159, 2.459] | 0.011 |

Both intervals exclude 1 and point opposite directions. This is a real,
significant, same-data reversal -- stronger evidence for Artifact 2 than the
qualitative "sign flips" claim already in the paper, now with its own CI on
both sides.

**Position-per-latent finding.** No separate literature sweep was run; the
finding is a direct corollary of reading the five D3 papers in full: none
report ablating a given latent at more than one sequence position and
decomposing the resulting variance. Templeton et al. ablate at one curated
position per case study; our own prior convention (before E11) measured
top-activating positions without checking whether they agreed. The six-arm
decomposition's residual term (position within latent) is 67.4% of variance
(68.6% with k41 dropped) -- larger than latent, arm, and their interaction
combined, and no audited paper checks this.

## Environment change logged

Upgraded `transformers` in the Python311 global environment (not the
`gemma_spectral` conda env, which was already at 5.5.0 and needed no change)
to add Gemma-3-1B config support; smoke-tested both Gemma-2-2B and Gemma-3-1B
forward passes before and after. `gemma_spectral`'s own transformers (5.5.0)
already supported `gemma3_text` architecture -- confirmed by a direct load and
forward pass through `google/gemma-3-1b-pt` in that env before launching any
real training. `train_arms.py`/`eval_arms.py` generalised to a `--base`
flag backed by `src/model_configs.py` (gemma2-2b: layer 12, d_model 2304;
gemma3-1b: layer 12 [same relative depth, both models have 26 layers],
d_model 1152; width/k held at 16384/82 for both). Backward compatibility
verified: a 4-sequence smoke run of the refactored `eval_arms.py` against the
existing six gemma2-2b arms reproduced the same live-latent counts and arm/k
mapping as the original run. Gemma-3-1B six-arm training + chained eval/decomp
launched in the background (`src/chain6_g3.sh`, `src/eval_decomp_g3.sh`);
outcome not yet known at time of writing.
