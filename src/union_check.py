# -*- coding: utf-8 -*-
"""E28: does the interaction collapse survive a non-selective shared rule?

Compares three position rules on the same latents at n_seq 384:

  per-arm            each arm at its own top-n positions (published practice)
  shared (min-gated) one common set per latent, ranked by the MINIMUM activation
                     across arms -- arm-symmetric, and selective toward tokens
                     where every arm fires hard
  union              candidates are the UNION of every arm's own top-n, still
                     requiring all arms nonzero, ranked by the MEAN across arms
                     -- arm-symmetric without selecting away from contested
                     tokens

Predictions and the decision rule are in PREREG.md E28, written before the
union evaluation ran. This script computes what that entry named, and prints
each prediction's outcome next to it so the result cannot be reported
selectively.
"""
import argparse
import io
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

from boundary_check import comp_cube
from moderator import build as build_cubes

B = 2000

MODELS = {
    "gemma2-2b": dict(per_arm="results/eval_arms_g2_s384.csv",
                      shared="results/eval_arms_g2_s384_shared.csv",
                      union="results/eval_arms_g2_s384_union.csv"),
    "gemma3-1b": dict(per_arm="results/eval_arms_g3_s384.csv",
                      shared="results/eval_arms_g3_s384_shared.csv",
                      union="results/eval_arms_g3_s384_union.csv"),
}
# The families added with the Qwen3.5, OLMo-2 and SmolLM3 arms, under the same
# three files per model. A model whose files are absent is reported MISSING and
# skipped, so the Gemma rows come out exactly as before on any checkout.
for _m, _s in (("qwen35-2b", "q35"), ("olmo2-1b", "o2"), ("smollm3-3b", "s3"),
               ("qwen35-4b", "q354b"), ("qwen35-9b", "q359b")):
    MODELS[_m] = dict(per_arm=f"results/eval_arms_{_s}_s384.csv",
                      shared=f"results/eval_arms_{_s}_s384_shared.csv",
                      union=f"results/eval_arms_{_s}_s384_union.csv")


def paired(ca, cs):
    seed = abs(int(np.sum(np.round(ca, 9) * 1e6))) % (2 ** 31)
    r = np.random.default_rng(seed)
    I = ca.shape[0]
    d = []
    for _ in range(B):
        p = r.integers(0, I, I)
        x, y = comp_cube(ca[p])["erho2"], comp_cube(cs[p])["erho2"]
        if np.isfinite(x) and np.isfinite(y):
            d.append(y - x)
    d = np.array(d)
    return (float(np.median(d)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/union_check.txt")
    a = ap.parse_args()
    lines, rows = [], []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    for model, f in MODELS.items():
        missing = [k for k, v in f.items() if not os.path.exists(v)]
        if missing:
            p(f"{model}: MISSING {missing}")
            continue
        p("=" * 92)
        p(f"{model}  (n_seq 384)")
        p("=" * 92)
        p(f"{'rule':<22}{'latents':>9}{'latent x arm':>14}{'Erho2':>9}"
          f"{'paired gain vs per-arm':>26}")
        res = {}
        for rule in ("shared", "union"):
            _, sh, ca, cs, fids = build_cubes(f["per_arm"], f[rule])
            A, Bc = comp_cube(ca), comp_cube(cs)
            g, lo, hi = paired(ca, cs)
            res[rule] = dict(n=len(fids), pa=A["pct_ab"], sh=Bc["pct_ab"],
                             pa_e=A["erho2"], sh_e=Bc["erho2"],
                             gain=g, lo=lo, hi=hi,
                             retained=sh.fid.nunique())
            if rule == "shared":
                p(f"{'per-arm':<22}{len(fids):>9}{A['pct_ab']:>13.1f}%"
                  f"{A['erho2']:>9.3f}{'--':>26}")
            ci = "{:+.3f} [{:+.3f}, {:+.3f}]".format(g, lo, hi)
            lab = "shared (min-gated)" if rule == "shared" else "union (mean)"
            p(f"{lab:<22}{len(fids):>9}{Bc['pct_ab']:>13.1f}%"
              f"{Bc['erho2']:>9.3f}{ci:>26}")
            rows.append(dict(model=model, rule=rule, n_paired=len(fids),
                             retained=sh.fid.nunique(),
                             pct_ab_perarm=A["pct_ab"],
                             pct_ab_shared=Bc["pct_ab"],
                             erho_perarm=A["erho2"], erho_shared=Bc["erho2"],
                             gain=g, lo=lo, hi=hi))
        p("")
        p(f"  retention: min-gated {res['shared']['retained']} latents, "
          f"union {res['union']['retained']}")
        p("")
        p("  PRE-REGISTERED PREDICTIONS (PREREG.md E28)")
        p(f"    P1 retention falls under union: "
          f"{res['union']['retained'] < res['shared']['retained']}")
        p(f"    P2 union interaction below 5%: "
          f"{res['union']['sh'] < 5.0}  ({res['union']['sh']:.1f}%)")
        p(f"    P3 union collapse weaker than min-gated: "
          f"{res['union']['sh'] > res['shared']['sh']}  "
          f"({res['union']['sh']:.1f}% vs {res['shared']['sh']:.1f}%)")
        p("")
        u, pa = res["union"]["sh"], res["union"]["pa"]
        if res["union"]["n"] < 20:
            p("    VERDICT: fewer than 20 latents; NOT ESTIMABLE on this model.")
        elif u >= pa:
            p("    VERDICT: union interaction at or above the per-arm value.")
            p("    The repair does NOT survive its own selection. This is a")
            p("    retraction, not an edit -- write R7 before touching the paper.")
        elif u < 5.0:
            p("    VERDICT: collapse survives. Position control, not candidate")
            p("    selection, is what removes the interaction.")
        else:
            p("    VERDICT: partial. Some of the collapse was selection. Report")
            p("    both rules; the union figure is the conservative headline and")
            p("    section 5 becomes a range across two arm-symmetric rules.")
        p("")

    if rows:
        pd.DataFrame(rows).to_csv("results/union_check.csv", index=False)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
