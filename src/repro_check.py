# -*- coding: utf-8 -*-
"""E13: does a seed-0 retrain reproduce the original six-arm measurement?

The original six arms were deleted before being archived, so this rerun is both
the restoration and a determinism test. Either outcome is reportable, but the
test is CONFOUNDED and the writeup must say so:

  the original run used transformers 5.5.0 (logged in PREREG); the rerun uses
  whatever is installed now (recorded in the archive's ENVIRONMENT.json).

So a non-reproduction is evidence of "this pipeline is not reproducible across
an environment upgrade", NOT of "seed 0 is nondeterministic". Only a clean
reproduction is unambiguous. Do not report the stronger claim.

Three levels, weakest evidence to strongest:
  1. identical position sets per (latent, arm)  -- selection determinism
  2. identical KL values                        -- numerical determinism
  3. identical variance components              -- what the paper actually claims
"""
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from variance_decomp import crossed_two_way


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="results/eval_arms.csv")
    ap.add_argument("--rerun", default="results/eval_arms_rerun.csv")
    ap.add_argument("--out", default="results/repro_report.txt")
    A = ap.parse_args()

    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    o = pd.read_csv(A.orig)
    r = pd.read_csv(A.rerun)
    p("=" * 84)
    p("REPRODUCIBILITY OF THE SIX-ARM RUN UNDER SEED 0")
    p("=" * 84)
    p("CONFOUND: the original ran under transformers 5.5.0; this rerun did not.")
    p("A mismatch below is evidence about the pipeline+environment jointly,")
    p("not about seed determinism alone.")
    p("")
    p(f"  rows: original {len(o)}, rerun {len(r)}")
    p(f"  arms: original {sorted(o.arm.unique())}")
    p(f"        rerun    {sorted(r.arm.unique())}")

    # ---- 1. same latents sampled? ----
    fo, fr = set(o.fid), set(r.fid)
    p(f"  latents: {len(fo)} vs {len(fr)}, "
      f"intersection {len(fo & fr)} ({100*len(fo & fr)/max(len(fo),1):.1f}%)")

    # ---- 2. same positions chosen? ----
    key = ["arm", "fid"]
    po = o.groupby(key)["pos"].apply(lambda s: frozenset(s))
    pr = r.groupby(key)["pos"].apply(lambda s: frozenset(s))
    both = po.index.intersection(pr.index)
    if len(both):
        ident = float(np.mean([po[i] == pr[i] for i in both]))
        p(f"  (latent, arm) cells in both: {len(both)}")
        p(f"  cells with an IDENTICAL position set: {100*ident:.1f}%")
    else:
        ident = 0.0
        p("  no overlapping (latent, arm) cells")

    # ---- 3. same measured values? ----
    m = o.merge(r, on=["arm", "fid", "pos"], suffixes=("_o", "_r"))
    if len(m):
        d = np.abs(m.kl_per_norm_o - m.kl_per_norm_r)
        rel = d / m.kl_per_norm_o.abs().clip(lower=1e-12)
        rho = stats.spearmanr(m.kl_per_norm_o, m.kl_per_norm_r).statistic
        p(f"  matched (arm, latent, position) observations: {len(m)}")
        p(f"  max |delta kl_per_norm|      = {d.max():.3e}")
        p(f"  median relative difference   = {np.median(rel):.3e}")
        p(f"  Spearman(original, rerun)    = {rho:+.4f}")
        bitwise = bool(d.max() == 0.0)
    else:
        bitwise = False
        p("  no matched observations to compare")

    # ---- 4. same variance components? ----
    p("")
    p("  variance components, original vs rerun:")
    comps = {}
    for name, df in (("original", o), ("rerun", r)):
        df = df.copy()
        df["y"] = np.log10(df.kl_per_norm.clip(lower=1e-12))
        comps[name] = crossed_two_way(df, "fid", "arm", "y")
    if all(comps.values()):
        a_, b_ = comps["original"], comps["rerun"]
        p(f"    {'component':<20}{'original':>10}{'rerun':>10}{'delta':>10}")
        for k, lab in (("pct_a", "latent"), ("pct_b", "arm"),
                       ("pct_ab", "latent x arm"), ("pct_e", "residual")):
            p(f"    {lab:<20}{a_[k]:>9.1f}%{b_[k]:>9.1f}%"
              f"{b_[k]-a_[k]:>+9.1f}")
        e_o = a_["v_a"] / (a_["v_a"] + a_["v_ab"] + a_["v_e"] / a_["n_per_cell"])
        e_r = b_["v_a"] / (b_["v_a"] + b_["v_ab"] + b_["v_e"] / b_["n_per_cell"])
        p(f"    {'E rho^2':<20}{e_o:>10.3f}{e_r:>10.3f}{e_r-e_o:>+10.3f}")
        p("")
        p(f"  VERDICT: {'bit-identical' if bitwise else 'NOT bit-identical'}; "
          f"position sets {100*ident:.1f}% identical; "
          f"E rho^2 moves {e_r - e_o:+.3f}")
        p("  The paper's claim is about the COMPONENTS, so the last line is the")
        p("  one that matters -- bitwise equality is a bonus, not the standard.")

    with open(A.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
