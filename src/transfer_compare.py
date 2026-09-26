# -*- coding: utf-8 -*-
"""E16: how much of the transfer failure is position selection?

THE RISK THIS ADDRESSES. The headline transfer statistic -- pairwise Spearman of
per-latent causal effect across arms, median 0.376 -- is computed on per-latent
MEDIANS OVER POSITIONS THAT EACH ARM CHOSE FOR ITSELF. Arms agree on the single
top-activating position for only 17% of latents (E12). So correlating arm x
against arm y partly correlates latent i measured in one place against latent i
measured in another, and an unknown share of the observed instability is
position selection rather than the dictionary.

Under --pos_mode shared every arm is measured at the same arm-symmetric
positions, so that noise is gone. rho_shared SHOULD exceed rho_per_arm. This
script reports both, side by side, with the same estimator.

BOTH NUMBERS GO IN THE PAPER. rho under per-arm positions is what published
practice produces; rho under shared positions is what the latent alone
contributes. Their difference is the position finding expressed as a transfer
statistic rather than a variance share.

Decision rule is pre-registered in paper/S5_BRANCHES.md and must not be chosen
after seeing the number.
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

from variance_decomp import crossed_two_way

THRESH_RHO = 0.60
THRESH_HEDGE = 0.50


def summarise(path, label, fids=None):
    """fids restricts to a latent subset -- needed because the shared-position
    design KEEPS ONLY latents that fire in every arm at a common position, and
    those are plausibly the more positionally stable ones. Comparing all-latent
    per-arm against subset shared would confound the design with that selection,
    so the per-arm number is also computed on the shared design's own latents."""
    e = pd.read_csv(path) if isinstance(path, str) else path
    if fids is not None:
        e = e[e.fid.isin(fids)]
    e = e.copy()
    e["y"] = np.log10(e.kl_per_norm.clip(lower=1e-12))
    r = crossed_two_way(e, "fid", "arm", "y")
    pw = e.groupby(["arm", "fid"]).kl_per_norm.median().unstack(0).dropna()
    rhos = [stats.spearmanr(pw[x], pw[y]).statistic
            for x, y in itertools.combinations(pw.columns, 2)]
    erho2 = (r["v_a"] / (r["v_a"] + r["v_ab"] + r["v_e"] / r["n_per_cell"])
             if r else float("nan"))
    return dict(label=label, n_latents=len(pw), n_pairs=len(rhos),
                rho_med=float(np.median(rhos)), rho_min=float(min(rhos)),
                rho_max=float(max(rhos)), pct_latent=r["pct_a"],
                pct_arm=r["pct_b"], pct_inter=r["pct_ab"], pct_resid=r["pct_e"],
                erho2=erho2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_arm",
                    default="results/eval_arms_rerun_fixsamp.csv")
    ap.add_argument("--shared",
                    default="results/eval_arms_shared_fixsamp.csv")
    ap.add_argument("--original", default="results/eval_arms.csv")
    ap.add_argument("--out", default="results/transfer_compare.txt")
    A = ap.parse_args()

    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    rows = []
    import os
    shared_fids = None
    if os.path.exists(A.shared):
        shared_fids = set(pd.read_csv(A.shared).fid.unique())
    # --original defaults to the gemma2 first run; when it is the same file as
    # --per_arm (any model without an earlier run, e.g. gemma3) showing both
    # would print one measurement as two independent rows.
    specs = []
    if os.path.abspath(A.original) != os.path.abspath(A.per_arm):
        specs.append((A.original, "original (per-arm positions)", None))
    specs.append((A.per_arm, "per-arm positions", None))
    if shared_fids is not None:
        specs.append((A.per_arm, "per-arm, RESTRICTED to shared latents",
                      shared_fids))
    specs.append((A.shared, "SHARED positions", None))
    for path, label, fids in specs:
        if os.path.exists(path):
            rows.append(summarise(path, label, fids))
        else:
            p(f"  [missing] {path} -- run src/run_all.sh")

    if not rows:
        raise SystemExit("no inputs available")

    p("=" * 84)
    p("E16  TRANSFER UNDER PER-ARM vs SHARED POSITION SELECTION")
    p("=" * 84)
    p(f"{'design':<32}{'lat':>5}{'pairs':>7}{'rho med':>9}{'rho range':>18}"
      f"{'inter%':>8}{'Erho2':>8}")
    for r in rows:
        rng_txt = "[{:+.3f}, {:+.3f}]".format(r["rho_min"], r["rho_max"])
        p(f"{r['label']:<32}{r['n_latents']:>5}{r['n_pairs']:>7}"
          f"{r['rho_med']:>+9.3f}{rng_txt:>18}"
          f"{r['pct_inter']:>8.1f}{r['erho2']:>8.3f}")

    sh = next((r for r in rows if "SHARED" in r["label"]), None)
    base = next((r for r in rows if r["label"] == "per-arm positions"), None)
    restr = next((r for r in rows if "RESTRICTED" in r["label"]), None)
    if sh is None or base is None:
        p("\n  shared-position run not available yet; nothing to decide.")
    else:
        p("")
        p(f"  rho:          {base['rho_med']:+.3f} -> {sh['rho_med']:+.3f}  "
          f"({sh['rho_med'] - base['rho_med']:+.3f} total)")
        p(f"  latent x arm: {base['pct_inter']:.1f}% -> {sh['pct_inter']:.1f}%")
        p(f"  E rho^2:      {base['erho2']:.3f} -> {sh['erho2']:.3f}")
        if restr:
            sel = restr["rho_med"] - base["rho_med"]
            des = sh["rho_med"] - restr["rho_med"]
            p("")
            p("  DECOMPOSING THAT GAIN (the shared design keeps only latents")
            p("  firing in every arm at a common position, so part of the gain")
            p("  is selection onto positionally stable latents, not the design):")
            p(f"    selection onto those latents : {sel:+.3f}"
              f"  ({100*sel/max(sh['rho_med']-base['rho_med'],1e-9):.0f}% of gain)")
            p(f"    position crossing itself     : {des:+.3f}"
              f"  ({100*des/max(sh['rho_med']-base['rho_med'],1e-9):.0f}% of gain)")
            p(f"    like-for-like interaction    : {restr['pct_inter']:.1f}% -> "
              f"{sh['pct_inter']:.1f}% on the SAME latents")
            p(f"    like-for-like E rho^2        : {restr['erho2']:.3f} -> "
              f"{sh['erho2']:.3f}")
            p("")
            p("  The like-for-like lines are the defensible ones. Quote those.")
        p("")
        # decide on the like-for-like comparison where one is available
        cmp_base = restr or base
        if sh["rho_med"] >= THRESH_RHO and sh["pct_inter"] < cmp_base["pct_inter"]:
            p("  >>> BRANCH A (see paper/S5_BRANCHES.md): the instability is")
            p("  largely position selection. The abstract's 'attributions do")
            p("  not transfer' claim is TOO STRONG as written and needs")
            p("  rewriting, not rewording.")
        elif sh["rho_med"] <= THRESH_HEDGE:
            p("  >>> BRANCH B (see paper/S5_BRANCHES.md): the nesting was a")
            p("  nuisance and it is now controlled. Abstract stands.")
        else:
            p("  >>> BRANCH A with the hedge (rho in the 0.50-0.60 band).")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
