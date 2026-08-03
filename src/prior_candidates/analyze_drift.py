# -*- coding: utf-8 -*-
"""Candidate C, Test 1: drift distributions, the content null, dose-response.

THE REAL DRIFT is within-pair: A and A' are the same text, so any movement is
attributable to segmentation and nothing else. Distributions are reported, not
means, because a mean over a skewed drift distribution says little.

THE NULL is between-text: pair A with a DIFFERENT base text B whose token count
sits the same distance from A's as A''s does. That is the drift one gets from
changing the content entirely while holding the size change fixed, and it is the
yardstick -- a formatting drift that approaches it means reformatting moves the
metric about as much as swapping the sentence.

SIGNED drift is reported alongside absolute, because the threat to a published
comparison is not noise but bias. A metric that moves randomly under
reformatting inflates variance; one that moves in a consistent direction will
make any condition with denser segmentation read as systematically higher or
lower.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCALE_FREE = ["hfer", "fiedler", "lambda_max", "sink_conc", "smoothness"]
NORMALISED = ["attn_entropy_norm", "attn_distance_norm", "eff_rank_norm",
              "spec_entropy_norm", "energy_norm"]
RAW = ["attn_entropy", "attn_distance", "eff_rank", "spec_entropy", "energy"]
ORDER = RAW + NORMALISED + SCALE_FREE

d = pd.read_csv("panel.csv")
ctrl = d[d.side == "identity"]
d = d[d.pair >= 0]
A = d[d.side == "a"].set_index("pair").sort_index()
B = d[d.side == "b"].set_index("pair").sort_index()
common = A.index.intersection(B.index)
A, B = A.loc[common], B.loc[common]

out = []
w = out.append

# --- numerical floor ------------------------------------------------------
if len(ctrl):
    a0 = A.loc[0]
    fl = max(abs(ctrl.iloc[0][m] - a0[m]) / max(abs(a0[m]), 1e-12)
             for m in ORDER)
    w(f"numerical floor (same text measured twice): max drift = {fl:.2e}")

# --- the content null -----------------------------------------------------
na = A["n_tok"].to_numpy()
dtok = B["n_tok"].to_numpy() - na
null_j = []
for i in range(len(A)):
    target = na[i] + dtok[i]
    cand = np.abs(na - target)
    cand[i] = 10 ** 9
    null_j.append(int(np.argmin(cand)))
null_j = np.array(null_j)
NB = A.iloc[null_j]

w(f"\npairs: {len(A)}   |token delta| median = "
  f"{np.median(np.abs(dtok / na)):.1%}")
w(f"null match quality: median |n_null - n_target| = "
  f"{np.median(np.abs(NB['n_tok'].to_numpy() - (na + dtok))):.0f} tokens")


def rel(x, ref):
    return (x - ref) / np.maximum(np.abs(ref), 1e-12)


def q(v, f):
    return float(np.nanquantile(v, f))


w("\n" + "=" * 96)
w("WITHIN-PAIR DRIFT vs CONTENT NULL   (semantically identical text, "
  "reformatted)")
w("=" * 96)
w(f"{'metric':>20} {'|drift| med':>12} {'p75':>8} {'p90':>8} "
  f"{'signed med':>11} {'NULL med':>9} {'real/null':>10}")
w("-" * 96)

rows = []
for m in ORDER:
    a, b, nb = A[m].to_numpy(), B[m].to_numpy(), NB[m].to_numpy()
    r, rn = rel(b, a), rel(nb, a)
    med, nmed = q(np.abs(r), .5), q(np.abs(rn), .5)
    rows.append(dict(metric=m, med=med, p75=q(np.abs(r), .75),
                     p90=q(np.abs(r), .90), signed=q(r, .5), null=nmed,
                     ratio=med / nmed if nmed > 0 else np.nan))
    tag = ("  [scale-free]" if m in SCALE_FREE
           else "  [normalised]" if m in NORMALISED else "  [raw]")
    w(f"{m:>20} {med:>11.2%} {q(np.abs(r), .75):>7.2%} "
      f"{q(np.abs(r), .90):>7.2%} {q(r, .5):>+10.2%} {nmed:>8.2%} "
      f"{med/nmed if nmed>0 else float('nan'):>9.2f}{tag}")

t = pd.DataFrame(rows)
t.to_csv("drift_summary.csv", index=False)
w("\n  group medians of |drift|:")
for name, grp in (("raw (ceiling-dependent)", RAW),
                  ("ceiling-normalised", NORMALISED),
                  ("scale-free", SCALE_FREE)):
    s = t[t.metric.isin(grp)]
    w(f"    {name:>24}: {s.med.median():.2%}   (null {s.null.median():.2%}, "
      f"ratio {s.ratio.median():.2f})")

# --- stratum split --------------------------------------------------------
w("\n" + "=" * 96)
w("BY STRATUM: does the result survive without the NBSP operator?")
w("=" * 96)
w(f"{'metric':>20} " + "".join(f"{s:>22}" for s in ("typographic only",
                                                    "with NBSP")))
for m in ORDER:
    cells = []
    for s in ("typographic", "nbsp"):
        sel = (A["stratum"] == s).to_numpy()
        if sel.sum() < 5:
            cells.append(f"{'-':>22}")
            continue
        r = rel(B[m].to_numpy()[sel], A[m].to_numpy()[sel])
        dd = np.abs(dtok / na)[sel]
        cells.append(f"{q(np.abs(r), .5):>13.2%} (n={sel.sum()},"
                     f"d={np.median(dd):.0%})")
    w(f"{m:>20} " + "".join(cells))

# --- dose-response --------------------------------------------------------
w("\n" + "=" * 96)
w("DOSE-RESPONSE: does drift scale with the size of the segmentation change?")
w("  Spearman rho of |drift| against |token delta|, over all pairs.")
w("=" * 96)
absd = np.abs(dtok / na)
for m in ORDER:
    r = np.abs(rel(B[m].to_numpy(), A[m].to_numpy()))
    ok = np.isfinite(r)
    rho = pd.Series(r[ok]).corr(pd.Series(absd[ok]), method="spearman")
    w(f"{m:>20}  rho = {rho:+.3f}")

open("drift_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
