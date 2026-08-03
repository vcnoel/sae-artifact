# Release notes

Defects found during this analysis and what now prevents each from recurring.
Recorded because several were caught only by a check that could easily have been
skipped, and because at least four would have produced a wrong published number.

## Measurement defects

**BOS reconstruction inflated the pooled variance.** Including position 0 drove the
Gemma Scope SAE's explained variance to **-3.5**; excluding it gives **0.863** at
mean L0 85.4, matching the released spec. The BOS residual carries a massive
activation the SAE does not reconstruct (cos 0.44 there against ~0.93 elsewhere)
and its norm dominates. *Prevented by:* `sae_rare.py` asserts explained variance
> 0.5 and aborts otherwise. Without that assert the whole analysis would have run
on a broken hook point.

**Unnormalised perturbations reversed a conclusion.** Median perturbation norm runs
3.60 in the rarest frequency decile to 14.76 in the most frequent. On raw KL the
rarest decile looked *worse* than random (0.54, p=0.039); per unit norm it is
*better* (1.65, p=0.011). *Prevented by:* every causal number is reported both raw
and per unit perturbation norm.

**Wrong estimator paired with the test.** A ratio-of-medians bootstrap CI
([0.97, 1.42]) was reported alongside Mann-Whitney p=0.022, from two different
frameworks, and read as a contradiction. Hodges-Lehmann is that test's own point
estimate: 1.209, CI [1.032, 1.424], excludes 1. *Prevented by:* HL reported
wherever Mann-Whitney is.

**Symmetric residualisation fitted noise.** Adjusting both arms for max|cos| fitted
a slope of **+2.506** on the random arm against a predictor with rho=0.016,
p=0.82 — residualising against nothing. *Prevented by:* covariate asymmetry is
tested and printed before any adjustment; adjustment is one-sided only when the
control arm's relationship is null.

**A sign error made a control statistic constant.** Self-similarity was set to
-2.0 before taking `abs()`, so every latent's max|cos| came out as exactly 2.000.
*Prevented by:* zero the diagonal *after* `abs()`; the all-constant column is now
visible in the printed distribution.

**Sampling scheme silently determined a headline.** The frequency-stratified 240
latents sit at median alignment percentile ~49-54%, missing the extreme-alignment
latents. Wordlike fraction of the top-20 by alignment is **0.10** dictionary-wide
in Gemma Scope and **0.90** within that sample — opposite characters, same
covariate. Both alignment-adjusted numbers were fit on the odd population.
*Prevented by:* uniform re-sampling (PREREG E6), and the sampling scheme is now
reported with every adjusted number.

**Position was not recorded, so a confound is untestable retrospectively.**
`sae_rare.csv` has no position column, so candidate artifact 5 (later positions
have sharper next-token distributions) cannot be checked on the primary sample.
*Prevented by:* `multi_dict.py` records `pos`, `rel_pos`, `act`.

## Claims retracted

- *"The trained/random gap opens on rare features."* Falsified; it was perturbation
  magnitude, and frequency predicts causal mass in neither arm.
- *"The heavy tail is a population phenomenon."* One latent of 240 takes the
  variance ratio from 42.1x to 7.2x.
- *"Lazy training does not reproduce."* Their measurements sit at ~25-50M tokens
  against my 12M — more tokens, less movement — so it is a setup difference
  (isotropic init, per-step unit-norm renormalisation), not a discrepancy.
- *"The trained arm's high-alignment latents are semantic."* That was the top-20 of
  a stratified sample, not of the dictionary.
- *"TPP and SCR fail because they ablate and difference."* They are also the only
  set-ablation metrics in that suite; the properties are perfectly confounded and
  the attribution is not established.
- *"Interchange interventions avoid the alignment confound."* Ablation, interchange
  and clamping all perturb along the same direction d_f.

## Operational defects

**Two GPU collisions.** Running `geometry.py` alongside training pushed memory to
15.96/16.38 GB, which on WDDM spills to host RAM: 100% reported utilisation at
41 C with no progress for 30 minutes. Later, a stale watcher from a superseded
chain fired `eval_saes.py` at the same moment the live chain reached its own eval
step — two processes, one output file, one GPU. A third orphaned process held
2.2 GB unnoticed. *Prevented by:* one chain at a time, arms trained sequentially,
every queued step gated on the previous step's output file, and a check that the
task list is empty before leaving a run unattended.

**Progress was unobservable.** Piping training output through `grep` buffers when
stdout is not a terminal, so a stalled run looked identical to a healthy one.
*Prevented by:* no pipes in chain logging; file milestones as the progress signal.

**Everything lived in `AppData\Local\Temp`.** Windows clears that without warning.
1.7 GB of checkpoints, every CSV, and the outline were one cleanup away from gone.
*Prevented by:* this repository.
