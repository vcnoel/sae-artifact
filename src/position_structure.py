# -*- coding: utf-8 -*-
"""E12: is the six-arm design crossed on POSITION, and what is the residual?

The crossed decomposition (variance_decomp.py) treats latent x arm as crossed --
correct, since all arms share seed 0 -- and reports everything left over as
"residual (position)". That label is an INTERPRETATION, and this script tests it.

Two questions, in order:

  (a) Are the evaluated positions the same across arms for a given latent?
      eval_arms.py picks each latent's top-n activating positions using THAT
      ARM's own activations, so a priori they need not be. If they are not,
      position is nested within (latent, arm), not crossed with latent, and the
      latent x arm interaction is partly a disagreement about WHERE to measure
      rather than about the latent.

  (b) How much of the residual is systematic in observable position covariates
      (rel_pos, act, position index) versus genuinely unexplained?

Neither question needs a GPU or the checkpoints; both run off eval_arms.csv.
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


def overlap_stats(e):
    """Jaccard overlap of the measured position sets, per latent, per arm pair."""
    sets = (e.groupby(["fid", "arm"])["pos"]
             .apply(lambda s: frozenset(s.tolist())).unstack("arm"))
    arms = list(sets.columns)
    rows = []
    for x, y in itertools.combinations(arms, 2):
        j, shared, both = [], [], 0
        for _, rr in sets.iterrows():
            a, b = rr[x], rr[y]
            if not isinstance(a, frozenset) or not isinstance(b, frozenset):
                continue
            both += 1
            inter = len(a & b)
            j.append(inter / len(a | b))
            shared.append(inter)
        rows.append(dict(pair=f"{x} vs {y}", n_latents=both,
                         mean_jaccard=float(np.mean(j)),
                         mean_shared=float(np.mean(shared)),
                         pct_identical=100.0 * float(np.mean(np.array(j) == 1.0))))
    return pd.DataFrame(rows), sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossed", default="results/eval_arms.csv")
    ap.add_argument("--out", default="results/position_structure.txt")
    A = ap.parse_args()

    e = pd.read_csv(A.crossed)
    e["y"] = np.log10(e.kl_per_norm.clip(lower=1e-12))
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    p("=" * 84)
    p("(a) IS POSITION CROSSED WITH LATENT, OR NESTED WITHIN ARM?")
    p("=" * 84)
    ov, sets = overlap_stats(e)
    p(ov.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    mj = float(ov.mean_jaccard.mean())
    pid = float(ov.pct_identical.mean())
    p("")
    p(f"  mean Jaccard overlap of measured position sets = {mj:.3f}")
    p(f"  latents whose position set is IDENTICAL across an arm pair = {pid:.1f}%")
    p("  DESIGN: " + ("CROSSED on position (arms measure the same places)"
                      if mj > 0.95 else
                      "NESTED -- position is selected within (latent, arm); "
                      "the latent x arm term therefore mixes 'the latent behaves "
                      "differently' with 'the arms disagree about where it fires'"))

    # ---- (a2) how much do arms disagree about WHERE the top activation is? ----
    top1 = (e.sort_values("act", ascending=False)
             .groupby(["fid", "arm"]).head(1)
             .pivot(index="fid", columns="arm", values="pos"))
    agree = []
    for x, y in itertools.combinations(list(top1.columns), 2):
        d = top1[[x, y]].dropna()
        agree.append(100.0 * float((d[x] == d[y]).mean()))
    p(f"  arm pairs agreeing on the SINGLE top-activating position: "
      f"{np.mean(agree):.1f}% of latents (min {min(agree):.1f}, "
      f"max {max(agree):.1f})")

    # ---- (b) decompose the within-cell residual on position covariates ----
    p("")
    p("=" * 84)
    p("(b) IS THE RESIDUAL SYSTEMATIC IN POSITION, OR UNEXPLAINED?")
    p("=" * 84)
    # Residual = deviation from the (latent, arm) cell mean: exactly the term
    # the ANOVA calls residual. Regress it on observable position covariates.
    e["cellmean"] = e.groupby(["fid", "arm"])["y"].transform("mean")
    e["resid"] = e["y"] - e["cellmean"]
    e["log_act"] = np.log10(e["act"].clip(lower=1e-12))
    # rank within cell by activation: 0 = the top-activating position
    e["act_rank"] = (e.groupby(["fid", "arm"])["act"]
                      .rank(ascending=False, method="first") - 1)

    sub = e.dropna(subset=["resid", "rel_pos", "log_act", "act_rank"])
    ss_tot = float((sub.resid ** 2).sum())
    for name, cols in [("rel_pos", ["rel_pos"]),
                       ("log_act", ["log_act"]),
                       ("act_rank", ["act_rank"]),
                       ("rel_pos + log_act + act_rank",
                        ["rel_pos", "log_act", "act_rank"])]:
        X = np.column_stack([np.ones(len(sub))] +
                            [sub[c].values for c in cols])
        beta, *_ = np.linalg.lstsq(X, sub.resid.values, rcond=None)
        pred = X @ beta
        r2 = 1.0 - float(((sub.resid.values - pred) ** 2).sum()) / ss_tot
        p(f"  R^2 of residual on {name:<32} = {r2:.4f}")

    rho_rp = stats.spearmanr(sub.rel_pos, sub.resid)
    rho_ar = stats.spearmanr(sub.act_rank, sub.resid)
    p(f"  Spearman(rel_pos, residual)  = {rho_rp.statistic:+.3f} "
      f"(p={rho_rp.pvalue:.2g})")
    p(f"  Spearman(act_rank, residual) = {rho_ar.statistic:+.3f} "
      f"(p={rho_ar.pvalue:.2g})")

    p("")
    p("  INTERPRETATION: the residual is position-within-cell BY CONSTRUCTION")
    p("  (it is the deviation from the latent x arm cell mean, and position is")
    p("  the only thing varying inside a cell). What the R^2 above shows is how")
    p("  much of it is SYSTEMATIC in observable position covariates. A low R^2")
    p("  means position matters enormously but not monotonically in depth or")
    p("  activation -- which position, not how late or how strong.")

    with open(A.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
