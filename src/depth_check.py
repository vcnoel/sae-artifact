# -*- coding: utf-8 -*-
"""E30 analysis: the mid-layer reproducibility check, then the depth trend.

THE CHECK COMES FIRST AND CAN VETO THE REST. The depth sweep recomputes the
existing mid layers on the pair already in the paper: 2B layer 12 at 16k L0 82
vs 22, and 9B layer 20 at 16k L0 68 vs 20. Those must return what the earlier
runs returned. If they do, the depth axis and the headline sit on the same
footing and the paper can say so. If they do not, there is a remote-versus-local
reproducibility problem and NEITHER number should be used until it is explained.
Comparing explicitly is the point; assuming it lands is how the sampler defect
survived.

THE L0 RATIO IS A CONFOUND AND IS REPORTED, NOT HIDDEN. Depth is compared on
same-width pairs, so the thing varying within a pair is sparsity, and the
released L0 targets do not sit at identical ratios across layers: 2B runs
3.8x / 3.7x / 3.2x and 9B 3.2x / 3.4x / 3.2x. That spread is small but it is
not zero, and a depth trend of a few points could partly be the ratio drifting.
The ratio is printed beside every row so a reader can weigh it.
"""
import io
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import pandas as pd

BANDS = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]

# suite -> [(layer, depth_pct, hi_l0, lo_l0)], mid layer marked by being the
# one whose pair also appears in the paper
DEPTHS = {
    "2b": [(5, 19, 68, 18), (12, 46, 82, 22), (20, 77, 71, 22)],
    "9b": [(9, 21, 51, 16), (20, 48, 68, 20), (31, 74, 63, 20)],
}
MID = {"2b": 12, "9b": 20}
# what the earlier local runs reported for the mid layer, low band, same-width pair
REFERENCE = {"2b": 9.836066, "9b": None}     # 9b filled from its own CSV
TOL = 0.05                                    # percentage points


def band_row(df, band):
    sub = df[df.matching == band]
    return None if not len(sub) else float(sub.top_agree.iloc[0])


def main():
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    # ---------- the check ----------
    p("=" * 78)
    p("MID-LAYER REPRODUCIBILITY CHECK (must pass before the trend is read)")
    p("=" * 78)
    verdict_ok = True
    for suite, ref in REFERENCE.items():
        f = f"results/depth_{suite}_L{MID[suite]}.csv"
        if not os.path.exists(f):
            p(f"  {suite} L{MID[suite]}: depth run MISSING, cannot check")
            verdict_ok = False
            continue
        got = band_row(pd.read_csv(f), "0.50-0.60")
        if ref is None:
            old = f"results/released_agreement_{suite}.csv"
            if os.path.exists(old):
                d = pd.read_csv(old)
                d = d[d.pair.str.startswith("same")]
                ref = band_row(d, "0.50-0.60")
        if ref is None:
            p(f"  {suite} L{MID[suite]}: no reference value available")
            verdict_ok = False
            continue
        delta = abs(got - ref)
        ok = delta <= TOL
        verdict_ok &= ok
        p(f"  {suite} L{MID[suite]} low band: remote {got:.3f}%  reference "
          f"{ref:.3f}%  delta {delta:.3f}pp  {'MATCH' if ok else 'MISMATCH'}")
    p("")
    if not verdict_ok:
        p("  Depth trend NOT reported: the mid-layer recomputation does not")
        p("  reproduce the existing number, so the two are not on the same")
        p("  footing and the difference must be explained first.")
        _write(out)
        return

    # ---------- the trend ----------
    p("=" * 78)
    p("DEPTH TREND  (same-width 16k pairs; L0 ratio shown as the confound)")
    p("=" * 78)
    rows = []
    for suite, spec in DEPTHS.items():
        p(f"{suite}")
        p("  %-6s %-7s %-9s %s" % ("layer", "depth", "L0 ratio",
                                   "  ".join("%9s" % b for b in BANDS)))
        for layer, pct, hi, lo in spec:
            f = f"results/depth_{suite}_L{layer}.csv"
            if not os.path.exists(f):
                p("  %-6d %-7s MISSING" % (layer, f"{pct}%"))
                continue
            d = pd.read_csv(f)
            vals = [band_row(d, b) for b in BANDS]
            p("  %-6d %-7s %-9s %s"
              % (layer, f"{pct}%", f"{hi/lo:.2f}x",
                 "  ".join("%8.1f%%" % v if v is not None else "        -"
                           for v in vals)))
            rows.append(dict(suite=suite, layer=layer, depth_pct=pct,
                             l0_hi=hi, l0_lo=lo, l0_ratio=hi / lo,
                             **{b: v for b, v in zip(BANDS, vals)}))
        p("")

    d = pd.DataFrame(rows)
    if len(d):
        d.to_csv("results/depth_summary.csv", index=False)
        p("=" * 78)
        for suite in DEPTHS:
            s = d[d.suite == suite].sort_values("depth_pct")
            if len(s) < 2:
                continue
            low = s[BANDS[0]].tolist()
            p(f"  {suite} low-band agreement by depth: "
              + " -> ".join("%.1f%%" % v for v in low)
              + "   (L0 ratios "
              + ", ".join("%.2f" % r for r in s.l0_ratio) + ")")
            spread = (s.l0_ratio.max() - s.l0_ratio.min()) / s.l0_ratio.min()
            p(f"      ratio spread {100*spread:.0f}%; agreement spread "
              f"{max(low)-min(low):.1f}pp. If the second is not comfortably "
              f"larger than\n      what the first could explain, say so rather "
              f"than claiming a depth effect.")
    _write(out)


def _write(out):
    with io.open("results/depth_check.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\nwrote results/depth_check.txt")


if __name__ == "__main__":
    main()
