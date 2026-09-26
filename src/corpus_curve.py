# -*- coding: utf-8 -*-
"""E25: the corpus-size curve on three points, and the extrapolation test.

Two points define a line. The paper reports 96 and 384, and a reviewer reading
Table 1 can draw a line through them and conclude the effect reaches zero at a
realistic evaluation budget. This adds 1536 so the question is answered with a
third point rather than an argument.

Predictions and the decision rule are in PREREG.md E25, written before the
1536 evaluation was run. This script computes only what that entry names, in
the order it names it, so the outcome cannot be reported selectively.
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

from boundary_check import comp_cube, to_cube
from moderator import build as build_cubes
from position_structure import overlap_stats

B_PAIRED = 2000

CURVE = {
    "gemma2-2b": [
        # *_fixsamp, not the bare files: the latter predate the sampler fix and
        # share 7 of 240 latents with the larger corpora. See make_macros.py.
        (96, "results/eval_arms_rerun_fixsamp.csv",
         "results/eval_arms_shared_fixsamp.csv"),
        (384, "results/eval_arms_g2_s384.csv",
         "results/eval_arms_g2_s384_shared.csv"),
        (1536, "results/eval_arms_g2_s1536.csv",
         "results/eval_arms_g2_s1536_shared.csv"),
    ],
    "gemma3-1b": [
        (96, "results/eval_arms_g3.csv", "results/eval_arms_g3_shared.csv"),
        (384, "results/eval_arms_g3_s384.csv",
         "results/eval_arms_g3_s384_shared.csv"),
        (1536, "results/eval_arms_g3_s1536.csv",
         "results/eval_arms_g3_s1536_shared.csv"),
    ],
}


def one(pap, shp):
    pa = pd.read_csv(pap)
    pa["y"] = np.log10(pa.kl_per_norm.clip(lower=1e-12))
    cu, _ = to_cube(pa)
    u = comp_cube(cu)

    ov, _ = overlap_stats(pa)
    t1 = (pa.sort_values("act", ascending=False).groupby(["fid", "arm"]).head(1)
          .pivot(index="fid", columns="arm", values="pos"))
    ag = []
    for x, y in itertools.combinations(list(t1.columns), 2):
        d = t1[[x, y]].dropna()
        ag.append(100.0 * float((d[x] == d[y]).mean()))

    _, _, ca, cs, fids = build_cubes(pap, shp)
    a_, b_ = comp_cube(ca), comp_cube(cs)

    seed = abs(int(np.sum(np.round(ca, 9) * 1e6))) % (2 ** 31)
    r = np.random.default_rng(seed)
    I = ca.shape[0]
    diffs = []
    for _ in range(B_PAIRED):
        pick = r.integers(0, I, I)
        dr = comp_cube(ca[pick])["erho2"]
        ds = comp_cube(cs[pick])["erho2"]
        if np.isfinite(dr) and np.isfinite(ds):
            diffs.append(ds - dr)
    diffs = np.array(diffs)
    return dict(
        unc_n=cu.shape[0], unc_pct_ab=u["pct_ab"], unc_erho=u["erho2"],
        jaccard=float(ov.mean_jaccard.mean()), top_agree=float(np.mean(ag)),
        paired_n=len(fids), pa_pct_ab=a_["pct_ab"], sh_pct_ab=b_["pct_ab"],
        pa_erho=a_["erho2"], sh_erho=b_["erho2"],
        gain=float(np.median(diffs)),
        lo=float(np.percentile(diffs, 2.5)),
        hi=float(np.percentile(diffs, 97.5)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/corpus_curve.txt")
    a = ap.parse_args()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    import os
    rows = []
    for model, pts in CURVE.items():
        p("=" * 100)
        p(f"{model}")
        p("=" * 100)
        p(f"{'n_seq':>6}{'Jaccard':>9}{'top-agree':>11}"
          f"{'unc lat':>9}{'unc ab%':>9}{'unc Erho':>10}"
          f"{'pair lat':>10}{'per-arm ab%':>13}{'shared ab%':>12}"
          f"{'gain':>8}{'95% CI':>20}")
        for n, pap, shp in pts:
            if not (os.path.exists(pap) and os.path.exists(shp)):
                p(f"{n:>6}   MISSING ({pap} / {shp})")
                continue
            r = one(pap, shp)
            r.update(model=model, n_seq=n)
            rows.append(r)
            ci = "[{:+.3f}, {:+.3f}]".format(r["lo"], r["hi"])
            p(f"{n:>6}{r['jaccard']:>9.3f}{r['top_agree']:>10.1f}%"
              f"{r['unc_n']:>9}{r['unc_pct_ab']:>8.1f}%{r['unc_erho']:>10.3f}"
              f"{r['paired_n']:>10}{r['pa_pct_ab']:>12.1f}%"
              f"{r['sh_pct_ab']:>11.1f}%"
              f"{r['gain']:>+8.3f}{ci:>20}")
        p("")

    d = pd.DataFrame(rows)
    d.to_csv("results/corpus_curve.csv", index=False)

    # ---- the pre-registered checks, evaluated in PREREG E25's own order ----
    p("=" * 100)
    p("PRE-REGISTERED CHECKS (PREREG.md E25)")
    p("=" * 100)
    for model in CURVE:
        m = d[d.model == model].sort_values("n_seq")
        if len(m) < 3:
            p(f"{model}: fewer than three points; checks not evaluable")
            continue
        j = m.jaccard.to_numpy()
        t = m.top_agree.to_numpy()
        p(f"{model}")
        p(f"  P1 disagreement keeps falling: Jaccard {j[0]:.3f} -> {j[1]:.3f} "
          f"-> {j[2]:.3f}; top-agree {t[0]:.1f}% -> {t[1]:.1f}% -> {t[2]:.1f}%")
        p(f"     Jaccard fell 384->1536: {j[2] < j[1]};  "
          f"top-agree fell: {t[2] < t[1]}")
        p(f"  P2 like-for-like interaction non-zero at 1536: "
          f"per-arm {m.pa_pct_ab.iloc[2]:.1f}%, shared {m.sh_pct_ab.iloc[2]:.1f}%"
          f"  (>2pp: {m.pa_pct_ab.iloc[2] > 2.0})")
        p(f"  P3 gain declines but stays positive: "
          f"{m.gain.iloc[0]:+.3f} -> {m.gain.iloc[1]:+.3f} -> "
          f"{m.gain.iloc[2]:+.3f}, CI at 1536 "
          f"[{m.lo.iloc[2]:+.3f}, {m.hi.iloc[2]:+.3f}]"
          f"  (lower bound > 0: {m.lo.iloc[2] > 0})")
        p("")
    p("Decision rule is in PREREG.md E25 and is applied in the write-up, not")
    p("here: this script reports the quantities the entry named and nothing more.")

    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
