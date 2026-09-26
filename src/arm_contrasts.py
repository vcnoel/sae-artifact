# -*- coding: utf-8 -*-
"""E15: the trained-vs-frozen endpoint as a DISTRIBUTION, not one contrast.

The paper's endpoint (MineVsFrozen, 1.19x [1.03, 1.38]) is a single
trained-vs-soft-frozen comparison, and its interval barely clears 1. That is
exactly the fragility the six-arm design fixed for Erho^2, and the same fix
applies here: there is not one defensible trained-vs-frozen contrast, there are
several, and the honest endpoint is their spread.

Among the six arms, decoder freedom splits them:
    DECODER-FREE  : free, lr1e4, order1, k41
    SOFT-FROZEN   : tau080 (cos 0.80 to init), tau090 (cos 0.90)
so every free x frozen pair is a defensible instance of "trained beats the
lazy-training baseline". k41 also changes sparsity, so it is reported inside the
full set and again excluded, since a reviewer will ask.

Estimator is identical to make_macros.py: Hodges-Lehmann on per-latent median
KL-per-unit-norm, bootstrap percentile interval. The arms are crossed (same
latent ids), so a paired variant is also reported -- it is the more powerful
test and the one the design actually licenses.
"""
import argparse
import io
import itertools
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

B = 4000
rng = np.random.default_rng(0)

FREE = ("free", "lr1e4", "order1", "k41")
FROZEN = ("tau080", "tau090")


def hl(a, b):
    return float(10 ** np.median(np.log10(a)[:, None] - np.log10(b)[None, :]))


def hl_ci(a, b):
    boot = np.array([hl(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                     for _ in range(B)])
    return hl(a, b), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def paired_hl_ci(x, y):
    """Both arms measured on the SAME latents: the per-latent log ratio is the
    natural paired statistic, and its median is the paired HL estimate."""
    d = np.log10(x) - np.log10(y)
    boot = np.array([np.median(rng.choice(d, len(d), True)) for _ in range(B)])
    return (float(10 ** np.median(d)), float(10 ** np.percentile(boot, 2.5)),
            float(10 ** np.percentile(boot, 97.5)),
            float(stats.wilcoxon(d).pvalue))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/eval_arms.csv")
    ap.add_argument("--out", default="results/arm_contrasts.txt")
    ap.add_argument("--csv", default="results/arm_contrasts.csv")
    A = ap.parse_args()

    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    e = pd.read_csv(A.src)
    w = e.groupby(["arm", "fid"]).kl_per_norm.median().unstack(0).dropna()
    have_free = [a for a in FREE if a in w.columns]
    have_frozen = [a for a in FROZEN if a in w.columns]

    p("=" * 84)
    p("E15  TRAINED-STYLE vs FROZEN-STYLE: EVERY DEFENSIBLE CONTRAST")
    p("=" * 84)
    p(f"  n = {len(w)} latents measured in all arms (crossed, so pairing is valid)")
    p(f"  decoder-free arms : {have_free}")
    p(f"  soft-frozen arms  : {have_frozen}")
    p("")
    p(f"  {'contrast':<22}{'HL':>7}{'95% CI':>18}"
      f"{'paired HL':>11}{'paired CI':>18}{'p':>10}")
    rows = []
    for f_, z in itertools.product(have_free, have_frozen):
        a, b = w[f_].to_numpy(), w[z].to_numpy()
        h, lo, hi = hl_ci(a, b)
        ph, plo, phi, pp = paired_hl_ci(a, b)
        rows.append(dict(free=f_, frozen=z, hl=h, lo=lo, hi=hi,
                         paired_hl=ph, paired_lo=plo, paired_hi=phi, paired_p=pp))
        p(f"  {f_ + ' vs ' + z:<22}{h:>7.3f}"
          f"{'[' + f'{lo:.2f}, {hi:.2f}' + ']':>18}"
          f"{ph:>11.3f}{'[' + f'{plo:.2f}, {phi:.2f}' + ']':>18}{pp:>10.2g}")

    d = pd.DataFrame(rows)
    d.to_csv(A.csv, index=False)

    # MATCHED contrasts differ from their comparator on decoder freedom ALONE.
    # Only `free` shares lr, k and data order with the tau arms; `lr1e4` and
    # `order1` each vary a second axis and `k41` varies sparsity, which is why
    # its contrasts are the largest in the table -- a k=41 dictionary spreads
    # the same reconstruction over half as many active latents, so each
    # latent's ablation removes more. Those are a sensitivity, not the estimand.
    for label, sub in (("MATCHED (decoder freedom only: free vs tau*)",
                        d[d.free == "free"]),
                       ("all contrasts", d),
                       ("excluding k41", d[d.free != "k41"])):
        if not len(sub):
            continue
        p("")
        p(f"  {label} (n={len(sub)}):")
        p(f"    unpaired HL  range [{sub.hl.min():.3f}, {sub.hl.max():.3f}], "
          f"median {sub.hl.median():.3f}")
        p(f"    paired   HL  range [{sub.paired_hl.min():.3f}, "
          f"{sub.paired_hl.max():.3f}], median {sub.paired_hl.median():.3f}")
        n_excl = int((sub.paired_lo > 1).sum())
        p(f"    contrasts whose PAIRED 95% CI excludes 1: {n_excl}/{len(sub)}")

    p("")
    p("  READING. The endpoint is a range over defensible fitting choices, not a")
    p("  point. Report the range. A single contrast landing at 1.19x was never")
    p("  wrong, but it was one draw from this spread and its interval's distance")
    p("  from 1 was a property of which pair happened to be run.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out} and {A.csv}")


if __name__ == "__main__":
    main()
