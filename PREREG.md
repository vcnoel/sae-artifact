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
| latent | (not separated) | 16.9% <!--{SixArmPctLatent}--> |
| arm | 0.4% <!--{TwoArmPctArm}--> | 3.2% <!--{SixArmPctArm}--> |
| latent x arm | 9.7% <!--{TwoArmPctInter}--> | 12.5% <!--{SixArmPctInter}--> |
| residual (position) | — | 67.4% <!--{SixArmPctResid}--> |
| latent x arm : arm ratio | 23.6x <!--{TwoArmRatio}--> | 3.8x <!--{SixArmRatio}--> |
| Erho2 (one-arm design) | 0.455 <!--{TwoArmErhoTwo}--> | 0.415 <!--{SixArmErhoTwo}--> |
| arms needed for Erho2=0.8 | 5 <!--{TwoArmArmsNeeded}--> (extrapolated) | 6 <!--{SixArmArmsNeeded}--> (direct) |

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
| raw KL | 0.683 <!--{RawFlipHL}--> | [0.459 <!--{RawFlipLo}-->, 0.961 <!--{RawFlipHi}-->] | 0.039 <!--{RawFlipP}--> |
| KL / unit perturbation norm | 1.687 <!--{NormFlipHL}--> | [1.171 <!--{NormFlipLo}-->, 2.498 <!--{NormFlipHi}-->] | 0.011 <!--{NormFlipP}--> |

Both intervals exclude 1 and point opposite directions.

**CI correction, logged.** An earlier version of this table recorded
[0.446, 0.962] and [1.159, 2.459]. `make_macros.py`'s `hl_ci` drew from a
single module-level RNG, so every bootstrap interval depended on how many
draws had been consumed by macros computed *earlier in the file* -- inserting
an unrelated macro upstream silently moved published intervals. `hl_ci` now
seeds a fresh generator from its own inputs, so each interval is a pure
function of its own data and is stable under reordering. Point estimates were
never affected (Hodges-Lehmann is deterministic), no interval moved by more
than 0.04, and no conclusion changes: both intervals still exclude 1 in
opposite directions. Logged rather than silently corrected because the whole
point of the generated-macro pipeline is that numbers cannot drift unnoticed,
and this was a channel by which they could. This is a real,
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
project conda env, which was already at 5.5.0 and needed no change)
to add Gemma-3-1B config support; smoke-tested both Gemma-2-2B and Gemma-3-1B
forward passes before and after. The conda env's own transformers (5.5.0)
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

**CORRECTION (2026-08-04).** That launch never ran. `chain6_g3.sh` pointed at
`anaconda3`, which no longer exists (the env moved to `miniconda3`), and `cd`'d
to `sae-artifacts` rather than `sae-artifact`; with no `set -e` the script
exited without producing a log, and the paper carried "replication in progress"
in two places for a run that did not exist. Both scripts are fixed
(`set -euo pipefail`, `cd "$(dirname "$0")/.."`, interpreter existence check),
`src/run_all.sh` now sequences the whole chain, and the paper's two claims were
restated to "implemented but not yet run" pending results. Also logged: the
original six-arm checkpoints were deleted before being archived, so the arms
are being retrained; `run_all.sh` archives both checkpoint sets outside the
repo. `transformers` is now 5.8.0, not the 5.5.0 recorded above, which
CONFOUNDS the retrain's determinism check (see E13).

## E12 — Is the six-arm design crossed on POSITION? **[post-hoc, no compute]**

The decomposition labels its residual "position", which is an interpretation.
`eval_arms.py` selects each latent's top-n positions using *that arm's own*
activations, so position is NESTED within (latent, arm), not crossed.

**Measured** (`src/position_structure.py`, from the existing CSV):

- mean Jaccard overlap of the measured position sets across the 15 arm pairs =
  **0.176**; only **1.7%** of latents have an identical position set in a pair
- arm pairs agreeing on the single top-activating position: **17.3%** of
  latents (range 7.1-31.7%)
- only **49** of 5,436 (latent, position) cells appear in all six arms, so the
  question cannot be answered by subsetting the existing data
- residual regressed on `rel_pos`, `log act` and within-cell activation rank:
  **R^2 = 0.005**. The residual is position, but not *systematic* in depth or
  strength -- it is *which* position, not how late or how strong.

**Consequences, stated in the paper rather than inferred:** the latent x arm
interaction mixes "the latent behaves differently under this fit" with "the
arms chose different tokens", and the two are not separable after the fact.
Note the overlap figure is computed on `pos` alone, ignoring which sequence,
so true overlap is **≤** 0.176 and the conclusion is conservative.

**Fix, and it needs the retrained arms:** `eval_arms.py --pos_mode shared`
picks one arm-symmetric position set per latent (candidates ranked by the
MINIMUM gated activation across arms, so the latent fires in every arm at every
selected position and no arm is the reference) and measures all arms there.
That design is fully crossed and its interaction term is selection-free. The
contrast between the two modes IS the estimate of how much of the interaction
was position selection. Pre-registered: if the shared-position interaction is
much smaller, part of what E11 reported as latent x arm was selection, and §5
is restated accordingly; if it is similar, E11's reading stands as written.
Either way the position VARIANCE result (67.4%) is unaffected, since that term
is within-cell in both designs.

## E13 — Does a seed-0 retrain reproduce E11? **[LIVE — written before the run]**

`src/repro_check.py`, three levels: identical position sets, identical KL
values, identical variance components. **The test is confounded**: the original
ran under transformers 5.5.0 and the rerun does not. A non-reproduction is
therefore evidence about pipeline-plus-environment, NOT about seed
determinism, and must not be reported as the latter. Only a clean reproduction
is unambiguous. The claim the paper makes is about the variance COMPONENTS, so
that is the line that matters; bitwise equality is a bonus.

## E14 — Rate vs magnitude readout **[pre-registered in OUTLINE, run 2026-08-04]**

`multi_dict.csv` has carried `flip_t`/`dtop_t` columns since the sweep; the
comparison against alignment was specified and never run. Alignment was never
computed for those nine dictionaries, so `src/rate_vs_magnitude.py` computes it
(same measure as `unembed_align.py`) and joins.

**Prediction (from OUTLINE):** rho(align, KL) >> rho(align, flip), with
rho(align, dtop) in between, because dtop is a bounded magnitude.

**Outcome — the binary contrast held, the ordering did not.** Restricted to
layer 12, where the alignment sensitivity actually exists (n=1500):
rho(align, KL/norm) = **+0.328** (p=5e-39), rho(align, flip) = **+0.100**
(p=1e-4), ratio **3.27**. But rho(align, dtop) = **+0.022**, *below* flip, not
between. So the magnitude-vs-rate recommendation survives and the graded
saturation story that motivated it does not; the paper claims only the former.

Two honest caveats, both in the paper: pooling over layer 19 (where alignment
predicts nothing for either readout) inflates the ratio to 5.08 by diluting
only the numerator, so the layer-12 figure is the one reported; and measured
flip rates are 0.017-0.110, i.e. a rare event, so part of the smaller
correlation is lower power rather than saturation. A design with flip rates
near 0.5 would separate those and is not available here.

## E15 — The endpoint as a distribution **[post-hoc, no compute]**

The paper's endpoint (1.19x [1.03, 1.38]) is one trained-vs-soft-frozen
contrast whose interval barely clears 1 -- the same one-degree-of-freedom
fragility the six-arm design fixed for Erho^2. Among the six arms, four leave
the decoder free and two soft-freeze it, giving 8 defensible contrasts on the
same latents (`src/arm_contrasts.py`, paired Hodges-Lehmann since the arms are
crossed).

**Outcome.** Paired HL spans **1.01x to 2.30x**; 6 of 8 intervals exclude 1.
Excluding `k41` (which also changes sparsity, and is the most favourable arm):
**1.01x to 1.62x**, median 1.26x, 4 of 6 excluding 1. One contrast
(`lr1e4` vs `tau080`) is a flat null at 1.01x [0.92, 1.18], p=0.71.

The endpoint is a range, not a point, and one defensible pair shows no
advantage at all. Reported as such. This is the paper's own thesis applied to
its own headline number, and it is the honest version.

## Bootstrap-CI stability defect, found and fixed 2026-08-04

`make_macros.py`'s `hl_ci` drew from a single module-level RNG, so every
bootstrap interval depended on how many draws earlier macros had consumed:
inserting an unrelated macro upstream silently moved published intervals. This
is how the D3 table came to record [1.159, 2.459] for an interval the source
now yields as [1.171, 2.498]. `hl_ci` now seeds a fresh generator from its own
inputs. Point estimates were never affected (Hodges-Lehmann is deterministic),
no interval moved by more than 0.04, and no conclusion changed.

`check_numbers.py` now also checks literals in `PREREG.md`, `OUTLINE.md` and
`README.md` that are bound to a macro by following the number with an inline
HTML comment naming that macro — see the D3 and E11 tables above for the exact
form (22 currently bound), so prose numbers can no longer drift from the
CSVs unnoticed. The first version of that check silently matched only 12 of 22
because its regex did not skip the unit between the number and the comment
("16.9%"); a guard now fails the run when fewer bindings parse than are
written.

## E23 — n_seq 384 **[LIVE — written before any s384 analysis was run]**

Raising the evaluation corpus from 96 to 384 sequences raises retention
mechanically, with no retraining. Measured retention (the only s384 quantity
looked at before writing this):

| model | retention at n_seq 96 | at n_seq 384 |
|---|---|---|
| gemma2-2b | 43% (103/240) | **54.6% (131/240)** |
| gemma3-1b | 24% (56/238) | **35.4% (85/240)** |

**Predictions, fixed now.**

1. *If the paired gain is driven by retention* (the account E20.1 failed to
   establish and R3 retracted), retention rose on BOTH models, so BOTH paired
   gains should RISE relative to their n_seq 96 values (+0.239 gemma2,
   +0.050 gemma3).
2. *If the gain is driven by how much latent x arm there was to remove* (E20.2,
   substantially an identity per E21a), the gain should track per-arm v_ab at
   the new retention, not retention itself.
3. *If Gemma-3's null was power*, its paired gain at 85 retained latents should
   become distinguishable from zero. If it was a model difference, it should
   stay near zero with a tighter interval.

Predictions 1 and 3 are separable: retention-driven predicts a rise on gemma2
too, power-only predicts gemma2 roughly unchanged while gemma3 sharpens.

**Decision rule.** If gemma2's gain does NOT rise while gemma3's does, the
retention account is dead (already retracted in R3) and Gemma-3's earlier null
was power. If both rise, retention is back in play and R3 should be revisited.
If neither moves, the model difference is real and belongs in the paper as one.

**Branch 4, added before running the s384 analysis.** Predictions 1 and 3 are
not fully separable as first written. Gemma-2's gain could rise from the extra
28 latents alone: more latents means a better-conditioned msa estimate, and E24c
showed that estimate is outlier-sensitive (5 of 70 latents carry 58% of
delta msa). So:

4. *If Gemma-2's gain rises but its JACKKNIFED delta msa does not*, the rise is
   estimator conditioning from the larger latent count, not retention and not a
   stronger effect. The jackknifed value must therefore be recomputed at
   n_seq 384 and compared against +0.687 (drop 5) and +0.339 (drop 10), not
   only the full-sample +1.627.

---

## E25 — n_seq 1536: does the effect survive extrapolation? **[LIVE — written before the run]**

**Why this run exists.** The paper currently reports the scope condition on two
points. Quadrupling the evaluation corpus from 96 to 384 sequences raised the
uncontrolled Erho2 from 0.373 to 0.492 and cut the paired gain from +0.239 to
+0.130. Two points define a line, and a reviewer is entitled to draw it: at a
realistic evaluation budget the gain reaches zero and the paper documents an
artifact of small corpora rather than of the convention. That objection is
available from Table 1 alone, needs no outside knowledge, and is fatal if it
lands unanswered. A third point at 1536 sequences distinguishes a slope toward
zero from a curve with a floor. Forward passes only; no retraining.

**What is run.** Both base models, both position modes, n_seq 1536, everything
else at the values used for 96 and 384 (n_feat 240, n_pos 6, layer 12, seed 0,
same six arms). Quantities reported alongside the 96 and 384 values: mean
Jaccard of the measured position sets, top-position agreement, uncontrolled
Erho2, uncontrolled latent x arm percentage, like-for-like per-arm and shared
latent x arm on the balanced cube, and the paired Erho2 gain with its interval.

**Predictions, fixed now, before any 1536 output is looked at.**

1. *Disagreement continues to fall.* Jaccard and top-position agreement are both
   LOWER at 1536 than at 384 on both models (gemma2: below 0.141 and 13.9%;
   gemma3: below 0.121 and 11.3%). This follows from the mechanism already
   stated in section 5 -- more sequences offer more candidate positions for the
   arms to disagree about -- and is the prediction most exposed to falsification,
   since nothing prevents agreement from plateauing.

2. *The like-for-like interaction remains non-zero.* At least one of the two
   models has a per-arm interaction above 2 percentage points on the balanced
   cube at 1536. The convention is a property of how position is selected, not
   of how much data the selection runs on, so the component it produces should
   not vanish with corpus size even as the uncontrolled decomposition tightens.

3. *The paired gain declines again but stays positive on gemma2.* The gemma2
   gain at 1536 is below +0.130 and above zero, with a lower interval bound
   above zero. Gemma-3's interval is expected to remain wide enough that its
   sign is not established at this corpus size.

Predictions 1 and 2 are separable and pull in opposite directions on purpose: 1
says the arms agree less as data grows, 2 says the consequence does not
disappear. If both hold, the scope condition is a curve with a floor and the
extrapolation objection is answered with data rather than argument.

**Decision rule, fixed now.**

- If the like-for-like interaction reaches zero on BOTH models at 1536, the
  extrapolation objection is CORRECT and the paper must say so. The claim then
  becomes explicitly bounded: the convention distorts comparisons at evaluation
  budgets up to some corpus size between 384 and 1536, and not beyond. The
  abstract, the contributions and section 5 all change, the headline claim is
  restated as budget-dependent, and Table 1 gains a 1536 row showing the
  vanishing. This is a real possible outcome and it is written down here so it
  cannot be reported as anything else.

- If the interaction survives on one model only, the surviving model carries the
  claim and the other is reported as a null at this budget, with the difference
  attributed to nothing until it is tested.

- If the interaction survives on both, the scope condition is restated as a
  decreasing curve with a non-zero floor, and the two-point line a reviewer
  would draw through 96 and 384 is shown to be wrong.

- If Jaccard RISES at 1536 on either model, prediction 1 is falsified and the
  "disagreement grows with corpus size" sentence in section 5 must be withdrawn
  and replaced with what the three points actually show. That sentence is
  currently stated as an expectation; three points make it a measurement.

**What this run cannot settle.** It says nothing about production-scale
dictionaries, which is a separate question addressed by E26 on released Gemma
Scope dictionaries, and nothing about whether the effect is an artifact of a
12M-token training budget, which is E27.

---

## E26 — Position agreement between RELEASED dictionaries **[LIVE — written before the run]**

**What this answers, and what it does not.** The paper's premise (two
dictionaries select different measurement positions for the same latent) is
currently evidenced on six self-trained arms at 12M tokens. The objection is
that it is an artifact of undertrained toys. The premise, unlike the repair,
needs no shared initialisation, so it can be tested on dictionaries someone
else trained at production scale. This entry covers the premise only. The
repair -- the crossed decomposition and the Erho2 gain -- cannot be tested this
way, because a latent-wise decomposition requires latent i to denote the same
thing in every arm and no released suite provides a shared seed. That is a
structural property of the estimand, not something more compute would fix.

**Design.** Gemma Scope, residual stream. Pairs, fixed now: for the 2B suite at
layer 12, width_16k L0 82 against width_16k L0 22 (same width, different
sparsity), and width_16k L0 82 against width_65k L0 72 (different width,
comparable sparsity). For the 9B suite at layer 20, width_16k L0 68 against
width_16k L0 20, and width_16k L0 58 against width_65k L0 55. Every path was
checked against the repository file listing rather than assumed. Evaluation
corpus 384 sequences of WikiText-103, 6 positions per latent, matching the
paper's main corpus.

**Matching.** Released dictionaries share no index correspondence, so latents
are matched by decoder cosine, mutual nearest neighbour, at thresholds 0.5
through 0.9. Survivor counts are reported at every threshold. Up to 512 matched
pairs are carried to the position stage under seed 0.

**Predictions, fixed now.**

1. Single-top-position agreement on matched pairs is BELOW 50% for every pair
   tested. The paper's six-arm figures are 13.9% and 11.3%, but those arms share
   an initialisation and these do not, so this is deliberately a weak threshold:
   the claim is that the phenomenon exists at production scale, not that it has
   the same magnitude.
2. Agreement is LOWER for the different-width pair than for the same-width pair,
   because differing width changes the dictionary more than differing sparsity
   does.
3. The 9B suite behaves like the 2B suite: agreement below 50% on both its
   pairs. If it does not, model scale is a moderator and the paper must say so.

**Known bias, direction stated in advance.** Matched pairs are the latents whose
decoder directions agree MOST. Their positions should therefore agree more than
a random pair's would, so any agreement figure here is an UPPER bound on
agreement across the full dictionaries. The bias runs in the paper's favour and
is reported as such.

**Decision rule.** If agreement on released pairs is above 50%, the premise does
NOT generalise to production dictionaries as stated, and the paper's scope must
be narrowed to dictionaries differing only in fitting choices from a shared
initialisation -- which would make it a much smaller claim, and the abstract and
introduction would both have to say so. If agreement is below 50%, the premise
is established at production scale and the "toy artifact" objection is answered
with data. Sensitivity to the cosine threshold is reported either way; if the
conclusion flips between thresholds 0.5 and 0.9, no conclusion is drawn.

---

## E27 — Is the disagreement an undertraining artifact? **[LIVE — written before the run]**

**The steelman.** Not "your SAEs are bad" but a mechanism: an undertrained
dictionary has a flatter activation profile across positions, so its argmax is
less determined, so two such dictionaries disagree about where to measure for a
reason that would not apply at production scale. That is testable.

**Design.** For each of our six Gemma-2-2B arms and for Gemma Scope width 16k
L0 82 on the same base model and layer, over 384 sequences: per (latent,
sequence), the ratio of the largest to the second-largest activation across
positions, and the Shannon entropy of the position profile normalised by
log(number of firing positions). Reduced to a per-latent median, then reported
as a distribution over latents. Rows with fewer than 3 firing positions are
dropped, since max-over-second-max is degenerate there. Sparsity is matched:
our arms are TopK k=82, the comparator is the released L0 82 dictionary.

**Prediction, fixed now.** Our arms are NOT flatter than the released
dictionary on both statistics simultaneously. If they are flatter on both, the
undertraining mechanism has support and the limitation must say so.

**Decision rule.** If our arms show a lower peak ratio AND higher entropy than
the released comparator, the objection has a mechanism, the disagreement we
measure is partly a property of our training budget, and the limitation is
rewritten to state that. If they do not, the objection is closed with a number:
unstable argmax is a property of SAE latents on real text rather than of a 12M
token budget.

**E27b, and why it is not in the paper.** The natural companion test is whether
position agreement RISES across our own training-budget curve (3M, 6M, 9M, 12M
tokens). It is not estimable from the files on disk: results/curve_*.csv drew
each arm's latent sample independently, so the trained and soft-frozen arms
share 3 latents of 120 at 12M. src/budget_agreement.py computes the quantity and
then refuses to report a trend below 30 shared latents, because its first
version printed a slope and a verdict from two. Answering the dose-response
question requires retraining with checkpoints and re-evaluating the milestones
on one shared latent list, which is training rather than re-analysis.

---

## E28 — Is the interaction collapse partly a selection effect of the shared-position rule? **[LIVE — written before the run]**

**The problem, found by asking a constructive question rather than an
adversarial one.** Writing ESTIMAND.md forced the question of what the
shared-position rule assumes but does not justify. It ranks candidate positions
by the MINIMUM gated activation across the six arms. That is what makes it
arm-symmetric, and it is also what makes it selective: it favours tokens where
every arm fires hard, and therefore disfavours tokens where the arms most
disagree -- which are precisely the tokens this paper is about. The controlled
estimate may be unbiased BETWEEN arms while being unrepresentative OF the
latent, and some fraction of the reported interaction collapse
(7.6% to 0.0% and 11.9% to 2.4% at 384 sequences) could be that selection
rather than the position control it is attributed to.

This is not a hypothetical: Figure 1 already shows the assigned shared position
lying outside the span between the two arms' argmaxes rather than between them.

**The test.** A second arm-symmetric rule that does NOT select away from
contested tokens. `--pos_mode union` builds each latent's candidate set from
the UNION of every arm's own top-n positions, so each arm's argmax is eligible
however much the other arms dislike it, then keeps the requirement that all six
arms have nonzero activation there (without it the design stops being crossed),
then ranks the survivors by MEAN activation across arms. Mean, not max, because
max would privilege whichever arm fires hardest and reintroduce an arm as the
reference; not min, because that rebuilds the rule under test.

Run on both base models at n_seq 384, everything else identical to the runs
already reported. Reported alongside per-arm and minimum-gated shared: the
like-for-like latent x arm component on the balanced cube, E rho^2 for both
designs, the paired gain, and retention.

**Predictions, fixed now, before any union output is examined.**

1. Retention FALLS under union relative to minimum-gated shared. The union
   admits contested tokens as candidates but still requires all six arms to
   fire, and contested tokens are where some arm is likely silent. If retention
   instead rises, the two rules are not selecting differently and the test is
   uninformative -- which would itself be worth knowing.
2. The interaction still collapses under union, to below 5% on both models.
   The claim under test is that position control, not candidate selection, is
   what removes the interaction; if that is right the rule used to pick the
   common position should not matter much.
3. The union collapse is WEAKER than the minimum-gated collapse on at least one
   model -- i.e. a higher residual latent x arm -- because the union genuinely
   admits harder tokens. A small weakening is expected and is not a failure.

Predictions 2 and 3 are deliberately in tension: 2 says the repair survives, 3
says the selection is real and does something. Both can hold, and if they do the
honest statement is that the collapse is mostly control with a measurable
selection component.

**Decision rule, fixed now.**

- If the union interaction is below 5% on both models, the repair survives its
  own selection. One clause goes in section 5 saying so, and the Limitations
  sentence about the rule's selectivity stays as a stated caveat with evidence
  behind it.
- If the union interaction is between 5% and the per-arm value on either model,
  part of the collapse WAS selection. The paper must then report both rules,
  the headline number becomes the union one (it is the conservative choice),
  and section 5's claim is restated as a range across two defensible
  arm-symmetric rules rather than a point.
- If the union interaction is at or above the per-arm value on either model,
  the repair does not survive and the position-control claim as stated is
  wrong. That would be a retraction (R7), not an edit, and nothing goes in the
  paper until it is written.
- If retention under union falls below 20 latents on a model, that model's
  result is reported as not estimable rather than as a null.

**What this cannot settle.** Two arm-symmetric rules agreeing does not mean
every arm-symmetric rule agrees, and neither rule is derived from a definition
of the estimand -- which is the gap ESTIMAND.md exists to work on after the
deadline.

---

## E29 — Does the premise hold on larger released dictionaries? **[LIVE — written before the run]**

**What is being tested.** The paper's premise is that two dictionaries select
different measurement positions for the same latent. It is established on
Gemma Scope 2B: 15 pairs, agreement 26% to 41% at median decoder cosine
0.78-0.91, and 9.8%/9.1% inside the 0.50-0.60 band. If that is a fact about
sparse autoencoders it should reappear at 9B, on dictionaries Google trained at
a scale nobody can call a toy. Forward passes and decoder matching only; no SAE
training of any kind.

**Design.** Identical to E26/E26b with the base model and layer switched.
Gemma-2-9B at layer 20, which is the richest layer Google published for it (34
dictionaries across seven widths). Grid of six: width 16k at L0 20 and 68, 32k
at L0 11 and 57, 65k at L0 11 and 55, giving 15 pairs varying both width and
sparsity. Corpus 384 sequences of WikiText-103 at 511 positions, matching
threshold, sample size and seeds unchanged from the 2B run. The band sweep uses
the same five disjoint cosine bands.

**Predictions, fixed now, before any 9B output exists.**

1. Agreement in the 0.50-0.60 band stays near the 2B value: between 5% and 15%
   on both 9B pairs. This is the headline prediction and the one most exposed:
   nothing about the mechanism says the rate must be preserved across a
   4.5x larger base model.
2. The monotone gradient with decoder cosine holds: agreement rises at every
   step of the five bands, on both pairs, as it did on 2B.
3. No pair in the 15-pair matrix exceeds 55% agreement.

**Decision rule, fixed now.**

- If band agreement lands in 5-15% and the gradient is monotone, the premise
  replicates at 9B and the paper reports two model scales rather than one.
- If band agreement is MATERIALLY HIGHER at 9B (above 20% in the 0.50-0.60
  band), that is NOT a failure of the paper: it is a scale-dependence finding,
  and it must be reported as one. The claim then becomes that the convention
  distorts comparisons and that the distortion SHRINKS with base-model scale,
  with the 2B and 9B numbers given side by side and no extrapolation past 9B.
  Writing it up as "the effect is smaller at scale" is the honest reading and
  it weakens the paper's practical urgency; that is the outcome, not a reason
  to bury it.
- If band agreement is materially LOWER (below 5%), the premise strengthens
  with scale and that too is reported as found, with the same refusal to
  extrapolate.
- If the gradient is NOT monotone at 9B, the upper-bound argument that rests on
  it (Appendix A) is weakened and must be restated as a 2B-only result.

**27B is conditional and is a weaker test.** Google published only 18
dictionaries for Gemma-2-27B, at three layers, and every one is width 131k. A
27B pair can therefore differ in sparsity but never in width, so it cannot test
the case a reader comparing two published numbers is usually in. If it runs it
is reported separately and never folded into a "three model scales" claim.

**What would make this run uninterpretable.** Fewer than 50 matched pairs in a
band, or a matched-set median cosine outside the 0.5-1.0 range the bands assume.
Either is reported as not estimable rather than as a null.

---

## E30 — Does the premise hold across DEPTH? **[LIVE — written before the run]**

**The objection this answers.** Every released-dictionary number in the paper
comes from one layer per model: layer 12 of Gemma-2-2B (46% depth) and layer 20
of Gemma-2-9B (48%). Both are mid-network, where features are most polysemantic
and least token-aligned. A reviewer can say the position disagreement is a
property of mid-network representations and would look different early or late.
That is a generality objection rather than a scale one, and it is currently
unanswered on both models.

**Design.** Same band sweep as E26, three depths per model, chosen early / mid /
late rather than evenly:

  Gemma-2-2B (26 layers): layer 5 (19%), layer 12 (46%, existing), layer 20 (77%)
  Gemma-2-9B (42 layers): layer 9 (21%), layer 20 (48%, existing), layer 31 (74%)

**Pairs are same-width 16k at every depth, and this is forced, not chosen.**
Only layer 12 of 2B and layer 20 of 9B publish the 16k/32k/65k grid; every other
layer publishes 16k plus one large width. Comparing a 16k-vs-65k pair at one
depth against a 16k-vs-131k pair at another would confound depth with pair type.
So depth is compared on same-width 16k pairs alone, at matched L0 ratios (2B
3.2-3.8x, 9B 3.2-3.4x), and the existing mid-layer number is recomputed on that
same restricted pair so all three depths are like-for-like. The mid-layer
HEADLINE numbers in the paper stay as they are; this is a robustness axis.

**Predictions, fixed now.**

1. Agreement in the 0.50-0.60 band FALLS with depth on both models: highest at
   the early layer, lowest at the late one. The reasoning is that early features
   are more token-aligned, so two dictionaries have less room to disagree about
   where a latent's evidence sits.
2. The monotone gradient with decoder cosine holds at every depth. This is the
   more important prediction: if the gradient survives at all six
   (model, layer) combinations, the mechanism is general.
3. No depth shows band agreement above 25% in 0.50-0.60, i.e. the premise does
   not disappear anywhere.

**Decision rule.**

- If agreement falls with depth as predicted, that is reported as found and the
  claim gains a sentence: the convention is safest early and worst late. It also
  slightly narrows the paper, since the audited papers mostly measure
  mid-to-late.
- If agreement is FLAT across depth, the claim generalises and that is the
  stronger outcome for the paper: one sentence saying the same gradient appears
  at three depths on both models.
- If agreement RISES with depth, prediction 1 is falsified and the mechanism
  needs restating; report it and do not fold it into the existing narrative.
- If any depth exceeds 25% in the low band, that layer is reported separately
  as a case where the convention is comparatively safe.

**What this cannot settle.** Three depths on two models of one family. Depth is
reported as a robustness axis, not a curve, and nothing is extrapolated to
layers or architectures not measured.

---

## E31 — A second model family: Llama Scope **[LIVE — written before any run, and before pod access exists]**

**Why this is the most valuable remaining experiment.** Every released-dictionary
result in the paper comes from Gemma Scope on Gemma-2. That is one vendor, one
architecture, one tokenizer, one SAE training recipe, and a base model family
that is no longer current. Llama Scope changes all five at once:
fnlp/Llama3_1-8B-Base-LXR-8x and -LXR-32x ship residual-stream dictionaries at
two expansion factors over the same 32 layers of Llama-3.1-8B, which is a
same-layer, different-width pair of exactly the kind released_agreement.py
consumes.

**Pair comparability, checked before running.** The width ratio is the quantity
that must match for the pair type to be comparable, and it does, exactly:

  gemma-2-2b   d_model 2304   16k vs 65k   ratio 4.0x   expansion  7.1x / 28.4x
  gemma-2-9b   d_model 3584   16k vs 65k   ratio 4.0x   expansion  4.6x / 18.3x
  llama-3.1-8b d_model 4096   32k vs 131k  ratio 4.0x   expansion  8.0x / 32.0x

Absolute widths differ, and the expansion relative to d_model lines up well
against 2B and less well against 9B. The 2B comparison is therefore the primary
one and the 9B comparison is secondary, stated in advance so the better-matching
pair is not selected afterwards.

**Prediction.** Agreement in the 0.50-0.60 band is between 5% and 20%, and the
monotone gradient with decoder cosine holds. That interval is deliberately wide:
the mechanism predicts a gradient, not a rate, and a rate prediction across
vendors would be false precision.

**What CANNOT be attributed, fixed now.** Llama Scope differs from Gemma Scope
in vendor, SAE architecture, training recipe, training corpus, tokenizer and
base model simultaneously. If agreement comes back materially different from
Gemma Scope's, that difference is confounded across all six and NO attribution
is available. The number is reported and not explained. Writing "Llama Scope's
SAEs are trained differently, which is why agreement is higher" would be a story
about an uncontrolled comparison, and this entry exists to make that
unavailable later.

**What a result would buy.** If the gradient replicates, the mechanism is a
property of top-activating selection rather than of Gemma Scope, and the
single-family limitation in the paper can be narrowed to name the families
tested. If it does not replicate, the paper's scope contracts to Gemma Scope and
that is reported as the finding.

**Blocked on.** meta-llama/Llama-3.1-8B is gated=manual, so it needs Meta's
approval on the account, not merely a token. Request now; run in a later
session. This is worth a pod of its own.

### E31 amendment, still before any run: sparsity checked, and the axes disagree

Reading the released hyperparameters rather than assuming them changes three
things.

**Llama Scope uses JumpReLU, the same activation family as Gemma Scope**
(`act_fn: jumprelu`). Better comparability than the entry above assumed: the
architectures are not as different as "different vendor" suggested.

**Both Llama expansions target the same sparsity, top_k = 50.** So the Llama
pair varies width at essentially constant sparsity, which is the same pair type
as the Gemma "different width, comparable sparsity" contrasts (2B L0 82 vs 72,
ratio 1.14; 9B L0 58 vs 55, ratio 1.05; Llama 50 vs 50, ratio 1.00).

**But the two comparability axes point at different Gemma models, so the
"2B is primary" choice above is withdrawn.**

  expansion vs d_model: Llama 8x/32x is closest to 2B (7.1x/28.4x),
                        not 9B (4.6x/18.3x)
  absolute sparsity:    Llama k=50 is closest to 9B (55, 58),
                        not 2B (72, 82)

Neither model matches on both axes. Rather than pick the axis that favours the
answer after seeing it, BOTH Gemma comparisons are reported as co-primary and
the disagreement between axes is stated. The width ratio, which is the axis that
defines the pair type, is 4.0x in all three and is the one that must match.

**The rate prediction is weak by design and is labelled as such.** Gemma's low
band gives 9.8, 9.1, 5.5 and 5.7 across two models and two pair types, so the
5-20% interval could barely have missed. It is retained as a sanity bound, not
as a test. THE REAL PREDICTION IS THE GRADIENT: agreement rising monotonically
across all five cosine bands. That can fail, and if it fails the mechanism does
not generalise beyond Gemma Scope.

**One implementation requirement, found in the hyperparameters.** Llama Scope
normalises activations dataset-wise (`norm_activation: dataset-wise`, with a
per-layer `dataset_average_activation_norm`, 10.8125 at layer 15). Our JumpReLU
reader does not apply that scaling, so running it unchanged would compute wrong
activations and a meaningless agreement figure. The reader must apply the
released norm before the run, and the run is not valid until it does.

## E32 [RECONSTRUCTED] Why the encode correction grows with decoder cosine

NOT a blind prediction, and marked accordingly. The mechanism was proposed AFTER
seeing four of five recomputed bands, and the 60-66 interval is a projection
fitted to that pattern rather than a commitment made in ignorance of it. Only the
top band was unknown when this was written. Calling it pre-registered would
overstate it; it is recorded because a projection written down before its target
lands is still worth more than one recalled afterwards, which is what this marker
is for.

**Observation.** Correcting the encode (R7) raised band agreement by an amount
that grows monotonically with cosine, not by a constant:

    0.50-0.60   9.84 -> 11.2   (+1.4)
    0.60-0.70  13.40 -> 14.3   (+0.9)
    0.70-0.80  23.38 -> 27.4   (+4.0)
    0.80-0.90  36.27 -> 42.7   (+6.4)
    0.90-1.00  54.66 -> ?

**Mechanism.** The old reader inflated pre-activations, so every latent fired at
roughly 3.7x too many positions (median 486 against 122 at width 16k). Spurious
firings add candidate positions and can only displace a top-activating position,
never restore one, so they depress agreement everywhere. But they depress it
most where there was concordance to destroy. A near-duplicate pair has genuinely
aligned activation profiles, so noise dilutes a real signal; a low-cosine pair
had little agreement to begin with, so noise has less to remove. The damage is
therefore proportional to the true agreement, which makes the correction grow
with cosine.

**Prediction.** The top band moves most in absolute terms: strictly more than
+6.4, and 54.66 lands in the range 60-66. Directional part: the increments
remain monotone in cosine.

**Decision rule.** Both parts must hold. If the top band lands below 61 or the
increments are non-monotone, the account is wrong and is not written into the
paper; the correction is then reported as an uneven shift with no mechanism
offered. Recorded either way. This is an explanation of an artifact in our own
tooling, not a claim about SAEs, and it stays in the retraction record and out
of the paper's argument regardless of outcome.

## E33 [LIVE] Reporting rule for a 9B matrix that will not finish

Written before the 9B matrix starts. It needs ~6h at 25 min/pair and the pod
does not have that, so it will be cut off mid-run and the stopping point will be
teardown rather than anything about the data.

**The hazard.** `itertools.combinations` walks the grid in width order, so an
interrupted run holds every 16k-vs-X pair and no wide-vs-wide pair. Its
agreement range would look like the full range while being a systematic
subsample, and choosing at teardown whether to report it is how selective
reporting happens.

**Fixed now.** Pair order is shuffled with seed 20260811 (`--shuffle_pairs`), so
any prefix is an unbiased sample of the 15 pairs. The CSV records the seed and
`pairs_total`, so a partial is visibly partial.

**Rule.** Report the 9B matrix only if it reaches at least 10 of 15 pairs, and
then only as a random subsample with the number of pairs stated and the pairs
named. Below 10 pairs it is not reported at all and the 9B evidence rests on the
band sweep. The threshold is set here, before any 9B matrix pair has been
computed under the corrected reader.

### E32 OUTCOME: FAILED. Mechanism dropped.

Top band landed at **60.2** (from 54.66), an increment of **+5.5**.

    band          old     new    delta
    0.50-0.60    9.84    11.2    +1.4
    0.60-0.70   13.40    14.3    +0.9
    0.70-0.80   23.38    27.4    +4.0
    0.80-0.90   36.27    42.7    +6.4
    0.90-1.00   54.66    60.2    +5.5   <- predicted > +6.4, and largest

Fails both parts of the rule. The prediction was that the top band moves most in
absolute terms, strictly more than +6.4; it moved +5.5, LESS than the band below
it, which is the direct opposite of what the mechanism implies. And the
increments are non-monotone in two places (+1.4 -> +0.9, and +6.4 -> +5.5),
where the account requires monotonicity.

Noting the near-miss honestly, because it is the failure mode the rule exists to
prevent: 60.2 falls inside the "60-66" projection quoted in the observation, and
reporting that clause alone would read as a pass. The operative rule was "below
61 OR non-monotone", and both clauses fire. Writing an interval and a threshold
that disagreed by a point was sloppy; the threshold was the decision rule and it
governs.

**Consequence.** The proportional-dilution account is not written into the paper
and is not offered anywhere as an explanation. The encode correction is reported
as what it is: an uneven upward shift, larger at high decoder cosine than at low.

**The shape, stated neutrally.** The increment rises to +6.4 at 0.80-0.90 and
falls to +5.5 at 0.90-1.00. Calling that an unexplained anomaly would overstate
it in the other direction: the top band started from 54.7% rather than 36.3%, so
there was less headroom for an agreement-increasing correction to move it, and a
compression near the ceiling is the obvious reading available to any reader. We
neither endorse that reading nor test it here. Declining to explain the shape is
different from pretending not to see it, and the record states the shape rather
than an anomaly.

Nothing in the paper's argument depended on this. It was an attempt to explain an
artifact in our own tooling, and the cost of dropping it is this paragraph.

## OPEN ITEMS carried out of the 2026-08-11 pod session

Every number below is computed and verified. None of it is written into the
paper yet, and three items must be settled first. Recorded here so the
constraints are not rediscovered from the numbers alone.

1. **Depth is not usable as it stands.** L0 ratios are 3.78 (L5), 3.73 (L12),
   3.23 (L20). The ratio at layer 20 is 15% off the other two, and layer 20 is
   exactly where the surprising numbers are. The low band falls monotonically
   with depth (19.1 -> 11.2 -> 6.0) while the top band rises (60.4 -> 60.2 ->
   67.7), which would be a claim that agreement DIVERGES by similarity with
   depth. Do not write that. Either find a layer-20 16k pair with a ratio near
   3.7 and rerun, or report the low-band trend alone and state that the
   top-band movement is confounded with the ratio.

2. **The 9B-agrees-less-than-2B matrix result needs a population check before
   use.** Full matrices give 2B 30.0-48.2% (median 37.5) against 9B 25.3-41.4%
   (median 32.1), but the 9B pairs span median cosine 0.75-0.92 against 2B's
   0.78-0.91, so part of the gap may be 9B pairs sitting at lower similarity.
   Weight each matrix by its own cosine distribution, or compare within matched
   cosine, before quoting it. The band-matched result already shows 9B lower
   inside every band, so this is expected to hold -- but the matrix figure is
   what a reviewer quotes and it should be the defensible version.

3. **\ScopeAgreeMax moved 41 -> 48 and a sentence depends on it.** "No pair
   agrees for more than X% of matched latents" reads differently at 48 than at
   41: close enough to half that "even the best-agreeing pair" carries less
   weight. Check that sentence still lands, and check it against
   \RelSparsityTopAgree = 60.2, which now sits ABOVE it. A matrix-wide maximum
   beside a single-band near-duplicate figure is precisely the population-mixing
   pattern the guard in check_numbers.py exists to catch.

4. **One appendix line on the L0 deviation.** Measured L0 exceeds the published
   spec by a mean of 2.6% across twelve dictionaries. It does NOT scale with
   width (pooled r = 0.445, permutation p = 0.15; 2B alone 0.76, 9B alone 0.18,
   so the 2B gradient was six-point noise). It does correlate with sparsity
   target (r = 0.614, p = 0.037), which is what a corpus difference predicts and
   not what a latent-count bug would. Frame as CONSISTENT WITH a corpus
   difference, not as established: two hypotheses were tested after looking at
   the data, so the adjusted p is roughly 0.07.

## E34 [LIVE] Layer-20 depth point at a matched L0 ratio

Written before the run. No agreement number for the 139-vs-38 pair exists.

**Why.** depth_2b_L20 used 16k L0 71 vs 22, ratio 3.23, against 3.78 (L5) and
3.73 (L12). The within-pair axis differed by 15% at exactly the layer producing
the surprising numbers, so the divergence-with-depth reading was unwritable.
Layer 20 publishes L0 22/38/71/139/294 at width 16k; 139 vs 38 gives 3.66, a
closer match than layer 12's own 3.73.

**The trade, stated in advance.** Matching the ratio unmatches the LEVEL: 139 vs
38 is far denser than 82 vs 22. Firing density is the denominator of position
agreement -- more active latents means more candidate positions, which
mechanically depresses agreement. This is a swap of one confound for a smaller
one, not the elimination of both, and the paper must say so.

**Prediction.** The low band comes back at or below the 6.0 measured at ratio
3.23, because the density level pushes the same way as any depth effect. The
top band is NOT predicted: at ratio 3.23 it was 67.7, above layer 12's 60.2, and
whether that survives a matched ratio is the open question.

**Decision rule, fixed now.**
- Depth is reported as the LOW-BAND TREND (19.1 at L5, 11.2 at L12, and whatever
  this run gives at L20) only if that trend stays monotonically decreasing.
- The top-band movement, and therefore any claim that agreement DIVERGES by
  similarity with depth, is reported only if the top band at ratio 3.66 also
  exceeds layer 12's 60.2. If it does not, the divergence reading was an artifact
  of the unmatched ratio and is dropped.
- Firing density is measured for both dictionaries and reported beside the
  agreement numbers regardless of outcome. If the pair fires at roughly double
  layer 12's rate, that goes in the same sentence as the result.
- If the low-band trend does not survive, depth is reported as confounded and
  said to be so.

### Open item 4, CORRECTED once E34's two dictionaries were measured

The sparsity correlation does not survive. E34's spec check added layer-20 L0
139 -> 143.2 (+3.02%) and L0 38 -> 40.4 (+6.32%), and the low-L0 one deviates
MORE, against the pattern. Full set:

                       r(dev, log L0)      r(dev, log width)
    12 dictionaries    +0.614  p 0.037     +0.445  p 0.147
    14 dictionaries    +0.476  p 0.086     +0.146  p 0.617

Unadjusted p moves from significant to not, and two hypotheses were tested after
looking, so nothing here is established. average_l0_38 at +6.32% is the largest
deviation in the whole set and is a low-L0 dictionary.

**What the appendix line should now say.** Measured L0 exceeds the published spec
by a mean of 2.9% across fourteen dictionaries spanning two base models, three
widths and two layers, with no structure we can establish in width or sparsity.
Consistent with evaluating on WikiText-2 rather than the training distribution.
Every dictionary is within 6.4% of spec and all pass the 15% assertion.

Recorded because the twelve-point version was written up two hours earlier and
would have gone into the paper as a sparsity-scaling finding. The correction cost
nothing only because the line had not been written yet.

### E34 OUTCOME: prediction falsified; divergence survives; low-band trend weakens

    layer  depth  ratio   0.50-0.60  ...  0.90-1.00
        5    19%   3.78        19.1            60.4
       12    46%   3.73        11.2            60.2
       20    77%   3.23         6.0            67.7   <- unmatched ratio
       20    77%   3.66        10.8            67.6   <- matched ratio

**Prediction falsified.** The low band was predicted at or below 6.0, since the
density level pushes the same way as any depth effect. It came back at 10.8, and
the 6.0 at ratio 3.23 was substantially an artifact of the unmatched ratio. The
ratio confound was doing most of the work at the low band, which is exactly why
this rerun was worth 40 minutes.

**Decision rule outcomes.**
- Top band at matched ratio is 67.6, exceeding layer 12's 60.2, so by the rule
  fixed in advance the DIVERGENCE reading may be reported: agreement at low
  similarity falls with depth (19.1 -> 11.2 -> 10.8) while agreement at high
  similarity rises (60.4 -> 60.2 -> 67.6).
- The low-band trend stays monotonically decreasing, so depth is reportable --
  but 11.2 -> 10.8 is nearly flat, against 11.2 -> 6.0 at the unmatched ratio.
  The falling half of the divergence is now WEAK and must be stated as such.

**Firing density, measured as required.** The layer-20 pair fires at medians 576
(L0 139) and 173 (L0 38), against 122 and 367 at layer 12: roughly 1.5x denser
overall. Density depresses agreement mechanically, so the level difference runs
against agreement at layer 20 in BOTH bands.

CORRECTION to a first version of this entry, which claimed the density confound
"cuts in our favour": it argued density as a headwind for the top band (a rise
despite it is more credible) and simultaneously as a tailwind for the low band (a
flattening because of it is less credible). That is the same confound read two
opposite ways to favour us in each. Nothing here justifies density acting
differently across similarity levels, and no such argument is offered. Density is
an uncontrolled level difference affecting both bands, and the entry stops there.

**Therefore, for the write-up.** Depth is compared at matched ratio and unmatched
density; the level difference runs against agreement at layer 20. What survives
is that agreement at high similarity rises with depth, while agreement at low
similarity falls between layers 5 and 12 and then flattens. No claim about which
half the density helps. Report both densities in the same sentence as the
agreement numbers.

**Scope.** Three layers on one model is a robustness axis, not a curve, and the
divergence reading is a shape claim from three points. One sentence and one
figure panel, not a section.

**Still to check before this becomes prose.** 9B layer 20 of 42 is 48% depth, a
mid-layer point on a second model. Not a depth axis, but it tests whether the
mid-layer numbers behave similarly across models. Worth one clause. Needs no GPU
-- results/released_agreement_9b.csv is already local.
