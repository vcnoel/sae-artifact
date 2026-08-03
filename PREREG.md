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
