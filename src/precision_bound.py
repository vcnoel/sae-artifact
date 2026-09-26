# -*- coding: utf-8 -*-
"""Leg 3 as a SENSITIVITY BOUND rather than a point estimate. CPU.

The target reports rho(depth, |delta logit|) = 0.81 over 42 layer means, each
estimated from nST latents (median 80, range 22-157 in their Table 21, which our
extraction validated). Their per-latent values are not released and their 9B
setting is not runnable here, so a point estimate of their within-layer variance
is unavailable.

It is not needed. Classical attenuation: if layer means are measured with error,
the observed correlation is attenuated relative to the correlation with true
layer means by sqrt(R), where

    R = sigma2_between / (sigma2_between + sigma2_within / n)

Since any true correlation is at most 1, observing rho_obs requires

    R >= rho_obs^2   =>   sigma2_within / sigma2_between <= n (1 - rho^2) / rho^2

That is a ceiling on the variance ratio their number tolerates, derived from
their published rho and nST alone. We then report the ratio measured on
Gemma-2-2B Gemma Scope as the plausible value that makes the ceiling concrete.
The claim does not depend on that measurement transferring across model scale;
the measurement only says whether the ceiling is near or far.
"""
import io
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

RHO = 0.812          # our validated re-derivation of their Table 21 statistic


def main():
    tt = pd.read_csv("results/target_tables.csv")
    t21 = tt[tt.table == 21]
    n_med = float(t21.nST.median())
    n_min, n_max = int(t21.nST.min()), int(t21.nST.max())

    print("THEIR DESIGN (Table 21, extraction validated at rho=+0.812 vs 0.81)")
    print(f"  layers = {len(t21)}   nST per layer: median {n_med:.0f}, "
          f"range {n_min}-{n_max}")

    print("\nCEILING ON THE VARIANCE RATIO THEIR NUMBER TOLERATES")
    print("  requires R >= rho^2 = %.3f, i.e. within/between <= n(1-rho^2)/rho^2"
          % RHO ** 2)
    print(f"  {'n per layer':>13} {'max within/between':>20}")
    for n in (n_min, int(n_med), n_max):
        print(f"  {n:>13} {n * (1 - RHO**2) / RHO**2:>20.1f}")

    # ---- measured ratio on our data, same dictionary family, 4 depths ----
    d = pd.read_csv("results/depth4.csv")
    f = (d.groupby(["layer", "fid"]).agg(kpn=("kpn_t0", "median"))
         .reset_index())
    f["y"] = np.log10(f.kpn.clip(lower=1e-12))
    grand = f.y.mean()
    lm = f.groupby("layer").y.agg(["mean", "var", "count"])
    # between-layer variance of the true means, and pooled within-layer variance
    s2b = float(((lm["mean"] - grand) ** 2 * lm["count"]).sum()
                / (lm["count"].sum() - 1) * len(lm) / max(len(lm) - 1, 1))
    s2b = float(np.var(lm["mean"].values, ddof=1))
    s2w = float((lm["var"] * (lm["count"] - 1)).sum() / (lm["count"] - 1).sum())
    print("\nMEASURED ON GEMMA-2-2B GEMMA SCOPE (log10 causal effect per unit norm)")
    print(f"  depths = {len(lm)}   latents per depth = {int(lm['count'].iloc[0])}")
    print(f"  between-layer variance of means  sigma2_b = {s2b:.4f}")
    print(f"  pooled within-layer variance      sigma2_w = {s2w:.4f}")
    print(f"  ratio within/between = {s2w/s2b:.1f}")

    print("\nVERDICT")
    for n in (n_min, int(n_med), n_max):
        ceil = n * (1 - RHO ** 2) / RHO ** 2
        verd = ("BOUND BITES" if s2w / s2b > ceil
                else "within tolerance")
        print(f"  at n={n:>3}: ceiling {ceil:>6.1f}, measured {s2w/s2b:>6.1f}"
              f"  -> {verd}")
    R = s2b / (s2b + s2w / n_med)
    print(f"\n  If their within/between matched ours, reliability at n={n_med:.0f}"
          f" would be R={R:.3f},")
    print(f"  attenuating any true correlation to at most sqrt(R)={R**0.5:.3f}.")
    print(f"  Their reported {RHO:.2f} would then be "
          f"{'UNATTAINABLE' if R**0.5 < RHO else 'attainable'}.")


if __name__ == "__main__":
    main()
