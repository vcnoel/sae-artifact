# -*- coding: utf-8 -*-
"""E29d: compare agreement across base-model scales, three ways.

2B and 9B do not have the same matched-pair population: 34.7% of 2B's mutual
nearest neighbours sit above cosine 0.90 against 28.7% of 9B's. Agreement rises
steeply with cosine, so that shift alone depresses 9B's population-weighted
agreement even if the convention behaves identically at both scales. Reporting
one number would let the population difference masquerade as a scale effect.

Three readings, all reported:

  (1) MATCHED ABSOLUTE COSINE -- agreement inside the same cosine band at each
      scale. Isolates the convention, because a band holds the degree of feature
      agreement fixed. This is primary if the three disagree.
  (2) MATCHED PERCENTILE -- the same quantile of each scale's OWN cosine
      distribution. Like-for-like by construction, but the two quantiles sit at
      different absolute similarities, so it mixes in the population shift.
  (3) POPULATION-WEIGHTED -- each scale's band agreements weighted by its own
      band shares. What a reader drawing a random matched pair would see, and
      the number most affected by the population difference.

Written before the 9B numbers were complete, so which of the three is primary
was not chosen to suit the answer.
"""
import io
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

BANDS = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
COL = {b: "band_" + b.replace("-", "_") for b in BANDS}


def load(suite):
    ra = f"results/released_agreement_{suite}.csv"
    cd = f"results/cosine_dist_{suite}.csv"
    if not (os.path.exists(ra) and os.path.exists(cd)):
        return None, None
    return pd.read_csv(ra), pd.read_csv(cd)


def main():
    out, rows = [], []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    data = {}
    for suite in ("2b", "9b", "27b"):
        ra, cd = load(suite)
        if ra is None:
            p(f"{suite}: MISSING, skipped")
            continue
        data[suite] = (ra, cd)

    if len(data) < 2:
        p("fewer than two scales available; nothing to compare")
        return

    # ---- (1) matched absolute cosine -------------------------------------
    p("=" * 78)
    p("(1) MATCHED ABSOLUTE COSINE  -- agreement inside the same band")
    p("=" * 78)
    p("%-12s %-10s %10s %10s %10s" % ("band", "suite", "median cos",
                                      "agree %", "n pairs"))
    for b in BANDS:
        for suite, (ra, _) in data.items():
            sub = ra[ra.matching == b]
            if not len(sub):
                continue
            p("%-12s %-10s %10.3f %10.1f %10d"
              % (b, suite, sub.median_cos.mean(), sub.top_agree.mean(),
                 len(sub)))
            rows.append(dict(reading="absolute", band=b, suite=suite,
                             median_cos=float(sub.median_cos.mean()),
                             agree=float(sub.top_agree.mean())))
        p("")

    # ---- (2) matched percentile ------------------------------------------
    p("=" * 78)
    p("(2) MATCHED PERCENTILE of each scale's own cosine distribution")
    p("=" * 78)
    p("  Band shares differ, so the same percentile lands at different absolute")
    p("  cosines. The cumulative share below each band boundary:")
    p("%-10s %s" % ("suite", "  ".join("%>%s" % b.split("-")[0] for b in BANDS)))
    for suite, (_, cd) in data.items():
        cum, acc = [], 0.0
        for b in reversed(BANDS):
            acc += float(cd[COL[b]].mean())
            cum.append(acc)
        p("%-10s %s" % (suite, "  ".join("%6.1f" % c for c in reversed(cum))))
    p("")

    # ---- (3) population-weighted -----------------------------------------
    p("=" * 78)
    p("(3) POPULATION-WEIGHTED by each scale's own band shares")
    p("=" * 78)
    for suite, (ra, cd) in data.items():
        w = np.array([float(cd[COL[b]].mean()) for b in BANDS])
        got = [ra[ra.matching == b].top_agree.mean() for b in BANDS]
        if any(pd.isna(g) for g in got):
            p("%-10s incomplete bands; not computed" % suite)
            continue
        a = np.array([float(g) for g in got])
        val = float((w * a).sum() / w.sum())
        p("%-10s weighted agreement %5.1f%%   (shares %s)"
          % (suite, val, " ".join("%.1f" % x for x in w)))
        rows.append(dict(reading="weighted", band="all", suite=suite,
                         median_cos=float("nan"), agree=val))
    p("")

    # ---- verdict ----------------------------------------------------------
    p("=" * 78)
    if "2b" in data and "9b" in data:
        ab = [r for r in rows if r["reading"] == "absolute"]
        pairs = [(b,
                  next((r["agree"] for r in ab if r["band"] == b and r["suite"] == "2b"), None),
                  next((r["agree"] for r in ab if r["band"] == b and r["suite"] == "9b"), None))
                 for b in BANDS]
        pairs = [(b, x, y) for b, x, y in pairs if x is not None and y is not None]
        lower = sum(1 for _, x, y in pairs if y < x)
        p("bands where 9B agreement is LOWER than 2B: %d of %d"
          % (lower, len(pairs)))
        wt = {r["suite"]: r["agree"] for r in rows if r["reading"] == "weighted"}
        if "2b" in wt and "9b" in wt:
            p("population-weighted: 2B %.1f%%  9B %.1f%%  (difference %+.1f)"
              % (wt["2b"], wt["9b"], wt["9b"] - wt["2b"]))
        p("")
        p("If band-matched and weighted point the same way, the scale claim is")
        p("safe. If they disagree, the band-matched reading is primary: it holds")
        p("feature agreement fixed and so isolates the convention, whereas the")
        p("weighted one moves with the matched-pair population.")

    pd.DataFrame(rows).to_csv("results/scale_compare.csv", index=False)
    with io.open("results/scale_compare.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\nwrote results/scale_compare.txt")


if __name__ == "__main__":
    main()
