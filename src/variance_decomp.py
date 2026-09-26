# -*- coding: utf-8 -*-
"""D2: variance components on log causal effect. CPU.

A STRUCTURAL CONSTRAINT THAT SHAPES THE WHOLE ANALYSIS. The reliability paper's
model x wrapper is CROSSED: the same model is measured under every wrapper. Here
the analogue is not directly available, because latents are not shared across
dictionaries -- Gemma Scope 16k/L0-82 latent 500 and 16k/L0-445 latent 500 come
from different training runs and denote nothing in common. Across the nine
dictionaries the design is therefore NESTED (latents within dictionary), and a
latent x dictionary interaction is not estimable from them.

One genuinely crossed comparison does exist. Our trained and soft-frozen arms
were initialised from the same seed, so W0 is identical and latent i denotes the
same initial direction in both; the frozen arm's decoder is constrained to stay
within cosine 0.8 of it. The uniform evaluation measured the SAME latent ids in
both arms at several positions each. That gives a crossed latent x arm design
with replication, which is the direct analogue of model x wrapper -- the question
being whether a latent's causal effect survives a change in how the dictionary
was fitted.

The untrained arm is excluded from the crossed analysis: it was seeded
differently, so its latent i is unrelated.

Components by ANOVA mean squares (method of moments), which is what a balanced
design supports without an iterative fit.
"""
import io
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def crossed_two_way(df, a, b, y):
    """y ~ A + B + A:B + e, balanced-ish; method-of-moments components."""
    d = df[[a, b, y]].dropna()
    la = sorted(d[a].unique())
    lb = sorted(d[b].unique())
    # A balanced design needs the same number of observations in every cell.
    # Taking the global minimum fails whenever any one latent fired at a single
    # position in one arm, so instead: choose a target n, keep only the latents
    # that reach it in EVERY arm, and subsample those to exactly n.
    cnt = d.groupby([a, b])[y].count().unstack(b)
    for n in (6, 5, 4, 3, 2):
        keep = cnt.index[(cnt >= n).all(axis=1)]
        if len(keep) >= 30:
            break
    else:
        return None
    d = d[d[a].isin(keep)]
    parts = [g.sample(n, random_state=0) for _, g in d.groupby([a, b])]
    d = pd.concat(parts)
    la = sorted(d[a].unique())
    lb = sorted(d[b].unique())
    I, J = len(la), len(lb)
    print(f"  balanced design: {I} latents x {J} arms x {n} positions "
          f"(dropped {len(cnt) - I} latents short of {n} in some arm)")
    gm = d[y].mean()
    ma = d.groupby(a)[y].mean()
    mb = d.groupby(b)[y].mean()
    mc = d.groupby([a, b])[y].mean()
    ssa = J * n * ((ma - gm) ** 2).sum()
    ssb = I * n * ((mb - gm) ** 2).sum()
    ssab = n * sum((mc.loc[(ai, bi)] - ma[ai] - mb[bi] + gm) ** 2
                   for ai in la for bi in lb)
    sse = sum(((g[y] - mc.loc[(ai, bi)]) ** 2).sum()
              for (ai, bi), g in d.groupby([a, b]))
    msa, msb = ssa / (I - 1), ssb / (J - 1)
    msab = ssab / ((I - 1) * (J - 1))
    mse = sse / (I * J * (n - 1))
    v_e = mse
    v_ab = max((msab - mse) / n, 0)
    v_a = max((msa - msab) / (J * n), 0)
    v_b = max((msb - msab) / (I * n), 0)
    tot = v_a + v_b + v_ab + v_e
    return dict(n_per_cell=n, I=I, J=J, v_a=v_a, v_b=v_b, v_ab=v_ab, v_e=v_e,
                pct_a=100 * v_a / tot, pct_b=100 * v_b / tot,
                pct_ab=100 * v_ab / tot, pct_e=100 * v_e / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossed", default="results/eval_arms.csv")
    A = ap.parse_args()
    # ---------- (1) NESTED: nine dictionaries, latents within ----------
    s = pd.read_csv("results/multi_dict.csv")
    s["y"] = np.log10(s.kpn_t0.clip(lower=1e-12))
    print("=" * 84)
    print("(1) NESTED across nine dictionaries (latent x dictionary NOT estimable)")
    print("=" * 84)
    gm = s.y.mean()
    per_d = s.groupby("dict_tag").y.agg(["mean", "count"])
    v_dict = float(np.var(per_d["mean"].values, ddof=1))
    per_l = s.groupby(["dict_tag", "fid"]).y.agg(["mean", "var", "count"])
    v_lat = float(per_l.groupby("dict_tag")["mean"].var().mean())
    v_pos = float((per_l["var"] * (per_l["count"] - 1)).sum()
                  / (per_l["count"] - 1).sum())
    tot = v_dict + v_lat + v_pos
    print(f"  dictionary            {v_dict:8.4f}   {100*v_dict/tot:5.1f}%")
    print(f"  latent within dict    {v_lat:8.4f}   {100*v_lat/tot:5.1f}%")
    print(f"  position within latent{v_pos:8.4f}   {100*v_pos/tot:5.1f}%")
    print(f"  -> a claim ABOUT dictionaries rests on the {100*v_dict/tot:.1f}% "
          f"component")

    # ---------- (2) CROSSED: latent x arm, shared initialisation ----------
    e = pd.read_csv(A.crossed)
    # every arm shares seed 0, so all of them are crossed with latent
    e = e.copy()
    e["y"] = np.log10(e.kl_per_norm.clip(lower=1e-12))
    print()
    print("=" * 84)
    print("(2) CROSSED latent x arm  (all arms seed 0, so latent i corresponds; "
          "same shared latent ids)")
    print("=" * 84)
    r = crossed_two_way(e, "fid", "arm", "y")
    if r is None:
        print("  insufficient replication per cell")
    else:
        print(f"  {r['I']} latents x {r['J']} arms, {r['n_per_cell']} positions "
              f"per cell")
        print(f"  latent                {r['v_a']:8.4f}   {r['pct_a']:5.1f}%")
        print(f"  arm                   {r['v_b']:8.4f}   {r['pct_b']:5.1f}%")
        print(f"  latent x arm          {r['v_ab']:8.4f}   {r['pct_ab']:5.1f}%")
        print(f"  residual (position)   {r['v_e']:8.4f}   {r['pct_e']:5.1f}%")
        print(f"\n  ratio latent x arm : arm = "
              f"{r['v_ab']/max(r['v_b'],1e-9):.1f}")

    # ---------- (3) does the ranking transfer? ----------
    p = (e.groupby(["arm", "fid"]).kl_per_norm.median().unstack(0).dropna())
    
    print()
    print("=" * 84)
    print("(3) DOES A LATENT'S CAUSAL EFFECT RANK TRANSFER ACROSS ARMS?")
    print("=" * 84)
    cols = list(p.columns)
    print(f"  n = {len(p)} latents measured in all {len(cols)} arms")
    print(f"  pairwise Spearman of per-latent causal effect across arms:")
    import itertools
    rs = []
    # NB: do not bind this loop variable to `r` -- that is the variance-component
    # dict from crossed_two_way and the Erho^2 block below needs it intact.
    for x, y2 in itertools.combinations(cols, 2):
        rho_xy = stats.spearmanr(p[x], p[y2]).statistic
        rs.append(rho_xy)
        print(f"    {x:>8} vs {y2:<8} {rho_xy:+.3f}")
    print(f"  median pairwise rho = {np.median(rs):+.3f}  "
          f"range [{min(rs):+.3f}, {max(rs):+.3f}]")
    # generalisability coefficient for a single-arm design
    if r:
        Erho2 = r["v_a"] / (r["v_a"] + r["v_ab"] + r["v_e"] / r["n_per_cell"])
        print(f"\n  E rho^2 for a ONE-ARM design (how well one dictionary's "
              f"ranking\n  estimates the latent's rank in general) = {Erho2:.3f}")
        need = (r["v_ab"] + r["v_e"] / r["n_per_cell"]) * 0.8 / (0.2 * r["v_a"]) \
            if r["v_a"] > 0 else float("inf")
        print(f"  arms needed for E rho^2 = 0.8: {np.ceil(need):.0f}")


if __name__ == "__main__":
    main()
