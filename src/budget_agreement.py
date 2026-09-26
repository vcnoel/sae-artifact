# -*- coding: utf-8 -*-
"""E27b: does position agreement improve as training tokens increase?

The undertraining objection says our arms disagree about where to measure
because they are undertrained. E27 (peakedness.py) tests the proposed mechanism
against a released dictionary. This tests the objection's own dose-response
directly, on data already on disk: the training-budget curve was evaluated at
3M, 6M, 9M and 12M tokens with a trained and a soft-frozen arm at each budget,
and each row records the position at which the latent was measured.

If disagreement is a symptom of undertraining, agreement should RISE with
tokens. If it is flat across a fourfold budget range, extrapolating it to
production scale has no support in our own data.

CAVEAT, stated because it bounds what this can show. These runs record two
positions per (latent, arm), not six, so Jaccard here is coarser than the
six-position figure quoted in the paper and the two are not comparable in
level. Only the TREND across budgets is interpretable, which is all the
objection turns on. The comparison is also two arms rather than six.
"""
import argparse
import glob
import io
import itertools
import re
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
from scipy import stats


def agreement(d):
    """Jaccard of measured position sets and top-1 agreement, per arm pair."""
    sets = (d.groupby(["fid", "arm"])["pos"]
            .apply(lambda s: frozenset(s.tolist())).unstack("arm"))
    top1 = (d.sort_values("act", ascending=False).groupby(["fid", "arm"])
            .head(1).pivot(index="fid", columns="arm", values="pos"))
    jac, agr, n = [], [], 0
    for x, y in itertools.combinations(list(sets.columns), 2):
        for _, r in sets[[x, y]].iterrows():
            if isinstance(r[x], frozenset) and isinstance(r[y], frozenset):
                jac.append(len(r[x] & r[y]) / len(r[x] | r[y]))
                n += 1
        t = top1[[x, y]].dropna()
        agr.append(100.0 * float((t[x] == t[y]).mean()))
    return float(np.mean(jac)), float(np.mean(agr)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/budget_agreement.txt")
    a = ap.parse_args()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    files = sorted(glob.glob("results/curve_*.csv"),
                   key=lambda x: int(re.search(r"(\d+)M", x).group(1)))
    p("=" * 78)
    p("E27b  POSITION AGREEMENT ACROSS THE TRAINING-BUDGET CURVE")
    p("=" * 78)
    p("  Two arms (trained vs soft-frozen), two positions per cell.")
    p("  Levels are not comparable to the six-arm, six-position figures in the")
    p("  paper; only the trend across budgets is.")
    p("")
    p(f"{'tokens':>8}{'latents':>9}{'pairs':>8}{'Jaccard':>10}{'top-1 agree':>13}")
    rows = []
    for f in files:
        M = int(re.search(r"(\d+)M", f).group(1))
        d = pd.read_csv(f)
        j, ag, n = agreement(d)
        rows.append(dict(tokens_M=M, latents=d.fid.nunique(), pairs=n,
                         jaccard=j, top_agree=ag))
        p(f"{M:>7}M{d.fid.nunique():>9}{n:>8}{j:>10.3f}{ag:>12.1f}%")

    t = pd.DataFrame(rows)
    t.to_csv("results/budget_agreement.csv", index=False)
    p("")
    # The curve runs sampled latents INDEPENDENTLY PER ARM (eval_saes.py
    # --uniform --per_bin), so trained and frozen share almost no latents: 3 of
    # 120 at 12M. Position agreement is undefined without a shared latent list,
    # and the first version of this script reported a slope and a verdict off
    # two or three latents without saying so. Refuse instead.
    MIN_PAIRS = 30
    if t.pairs.min() < MIN_PAIRS:
        p("=" * 78)
        p(f"REFUSING TO REPORT A TREND. The smallest budget has {t.pairs.min()} "
          f"latents measured")
        p("in BOTH arms, against a floor of "
          f"{MIN_PAIRS}. results/curve_*.csv drew each arm's")
        p("latent sample independently, so the two arms overlap on 3 of 120")
        p("latents at 12M and agreement is not estimable from these files.")
        p("")
        p("To answer the dose-response question this script was written for,")
        p("retrain with --ckpt_every (src/chain4.sh) and re-evaluate the")
        p("milestones on ONE shared latent list, as src/eval_arms.py does.")
        p("That is training, not a re-analysis, and it is the reason this")
        p("check is not in the paper.")
        with io.open(a.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nwrote {a.out}")
        raise SystemExit(2)

    if len(t) >= 3:
        sj = stats.linregress(t.tokens_M, t.jaccard)
        sa = stats.linregress(t.tokens_M, t.top_agree)
        p(f"  Jaccard vs tokens:     slope {sj.slope:+.5f} per M "
          f"(p={sj.pvalue:.2f}, r={sj.rvalue:+.3f})")
        p(f"  top-1 agree vs tokens: slope {sa.slope:+.4f}%% per M "
          f"(p={sa.pvalue:.2f}, r={sa.rvalue:+.3f})")
        p("")
        rising = sj.slope > 0 and sj.pvalue < 0.05
        if rising:
            p("  Agreement RISES with training budget. The undertraining")
            p("  objection has dose-response support in our own data and the")
            p("  limitation must say so.")
        else:
            p("  Agreement does NOT rise significantly across a fourfold budget")
            p("  range. Extrapolating the disagreement away as an undertraining")
            p("  artifact has no support in our own data. This is a bound on a")
            p("  4x range, not a claim about production budgets, which E27")
            p("  addresses by comparing profile shape against a released")
            p("  dictionary.")

    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
