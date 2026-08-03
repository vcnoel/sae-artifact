# -*- coding: utf-8 -*-
"""Two checks on the trained-vs-soft-frozen result. CPU only.

(1) EV GAP OVER TRAINING. The decoder claim is measured at 12M tokens, where
    both arms sit below Gemma Scope's 0.863. If the trained-minus-frozen EV gap
    is NARROWING toward convergence, more tokens would not change the verdict
    and the undertraining objection is pre-empted. If it is still widening, the
    claim must be scoped to this budget.

(2) HODGES-LEHMANN. Mann-Whitney p=0.022 beside a bootstrap CI of [0.97, 1.42]
    reads as a contradiction because they come from different frameworks. HL --
    the median of pairwise ratios -- is Mann-Whitney's own point estimate, so
    reporting it with its CI puts test and interval in the same framework.
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(0)
B = 2000


def main():
    # ---------- (1) EV trajectory ----------
    txt = io.open("chain.log", encoding="utf-8", errors="replace").read()
    rows = []
    for m in re.finditer(r"([\d.]+)M tok \| (\w+): mse=([\d.]+) ev=([\-\d.]+)",
                         txt):
        rows.append(dict(tok=float(m.group(1)), arm=m.group(2),
                         ev=float(m.group(4))))
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="tok", columns="arm", values="ev")
    piv = piv.dropna()
    piv["gap"] = piv["trained"] - piv["frozen"]
    print("EV TRAJECTORY, matched token counts")
    print(f"{'tokens(M)':>10} {'trained':>9} {'frozen':>9} {'gap':>8}")
    for t, r in piv.iterrows():
        print(f"{t:>10.1f} {r['trained']:>9.3f} {r['frozen']:>9.3f} "
              f"{r['gap']:>8.3f}")
    n = len(piv)
    first, last = piv["gap"].iloc[:max(1, n // 3)], piv["gap"].iloc[-max(1, n // 3):]
    sl = stats.linregress(piv.index.values, piv["gap"].values)
    print(f"\n  first-third mean gap = {first.mean():+.4f}")
    print(f"  last-third  mean gap = {last.mean():+.4f}")
    print(f"  OLS slope of gap vs tokens = {sl.slope:+.5f} per M tokens "
          f"(p={sl.pvalue:.4f})")
    verdict = ("NARROWING -> undertraining objection pre-empted"
               if sl.slope < 0 else
               "WIDENING or flat -> claim must be scoped to this budget")
    print(f"  VERDICT: {verdict}")
    print(f"  (eval-corpus gap for reference: 0.830 - 0.788 = 0.042)")

    # ---------- (2) Hodges-Lehmann ----------
    df = pd.read_csv("eval_saes.csv")
    f = (df.groupby(["arm", "fid"])
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    t = f[f.arm == "trained"].kpn.to_numpy()
    r = f[f.arm == "frozen"].kpn.to_numpy()

    def hl(a, b):
        return float(np.median(np.log10(a)[:, None] - np.log10(b)[None, :]))

    point = 10 ** hl(t, r)
    boot = np.array([10 ** hl(rng.choice(t, len(t), True),
                              rng.choice(r, len(r), True)) for _ in range(B)])
    mw = stats.mannwhitneyu(t, r)
    print("\n" + "=" * 66)
    print("TRAINED vs SOFT-FROZEN, norm-controlled -- test and interval aligned")
    print("=" * 66)
    print(f"  median ratio (ratio of medians) = "
          f"{np.median(t)/np.median(r):.3f}")
    print(f"  Hodges-Lehmann (median pairwise ratio) = {point:.3f}  "
          f"95% CI [{np.percentile(boot,2.5):.3f}, "
          f"{np.percentile(boot,97.5):.3f}]")
    print(f"  Mann-Whitney U p = {mw.pvalue:.4f}   "
          f"(HL is this test's point estimate)")
    frac = (t[:, None] > r[None, :]).mean()
    print(f"  P(trained > frozen) for a random pair = {frac:.3f} "
          f"(0.5 = no effect)")
    print(f"  -> {'HL CI excludes 1: consistent with the test' if np.percentile(boot,2.5) > 1 else 'HL CI includes 1: test and interval disagree, report both'}")


if __name__ == "__main__":
    main()
