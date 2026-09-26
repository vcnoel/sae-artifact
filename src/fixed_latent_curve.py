# -*- coding: utf-8 -*-
"""E25b: does the paired gain rise with DATA, or with SAMPLE COMPOSITION?

E25 added a third corpus and the paired gain did not decline as pre-registered:
+0.239 -> +0.130 -> +0.186 on Gemma-2 and +0.046 -> +0.138 -> +0.362 on
Gemma-3. Read naively that says the effect strengthens with evaluation data.

It probably does not, and the reason is visible in the same table. The paired
cube requires six positions per (latent, arm), and more corpus means more
latents clear that filter: 70 -> 93 -> 123 and 32 -> 53 -> 70. The latents that
newly clear it are the ones that fire rarely, and rare latents are exactly where
position selection has the most room to disagree. So the gain could be rising
because the SAMPLE is changing, not because the effect is.

This is the confound retracted as R2 and again as the latent-selection half of
the E16 gain. It has returned through a different door, and the test is the one
already built for it: hold the latent set FIXED across corpora and recompute.

  (1) FIXED-LATENT CURVE. Restrict all three corpora to the latents balanced in
      the paired cube at 96 AND 384 AND 1536 simultaneously. Any remaining
      trend is about data, because the sample cannot change.

  (2) WHAT THE FILTER ADMITS. Compare the latents balanced at 96 against those
      that only clear at 1536, on firing density, positional consistency and
      their own per-arm interaction. If the newcomers are the less consistent
      ones, sample composition is the explanation and (1) should be flat.

Neither outcome rescues the extrapolation objection: disagreement falls
monotonically on both models either way, and the interaction survives at every
corpus size. What changes is which sentence the paper is allowed to write.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

from boundary_check import comp_cube
from moderator import build as build_cubes
from retention_curve import consistency
from corpus_curve import CURVE

B_PAIRED = 2000


def paired_gain(ca, cs, idx):
    """Paired bootstrap over latents: ONE resample applied to both designs."""
    a, b = ca[idx], cs[idx]
    seed = abs(int(np.sum(np.round(a, 9) * 1e6))) % (2 ** 31)
    r = np.random.default_rng(seed)
    I = a.shape[0]
    d = []
    for _ in range(B_PAIRED):
        pick = r.integers(0, I, I)
        x = comp_cube(a[pick])["erho2"]
        y = comp_cube(b[pick])["erho2"]
        if np.isfinite(x) and np.isfinite(y):
            d.append(y - x)
    d = np.array(d)
    return (float(np.median(d)), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/fixed_latent_curve.txt")
    a = ap.parse_args()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    rows = []
    for model, pts in CURVE.items():
        built = {}
        for n, pap, shp in pts:
            pa, sh, ca, cs, fids = build_cubes(pap, shp)
            built[n] = dict(pa=pa, ca=ca, cs=cs, fids=list(fids))

        sizes = [n for n, _, _ in pts]
        common = set(built[sizes[0]]["fids"])
        for n in sizes[1:]:
            common &= set(built[n]["fids"])
        common = sorted(common)

        p("=" * 92)
        p(f"{model}")
        p("=" * 92)
        p(f"  latents balanced in the paired cube at each corpus: "
          + ", ".join(f"{n}:{len(built[n]['fids'])}" for n in sizes))
        p(f"  balanced at ALL THREE simultaneously: {len(common)}")
        if len(common) < 12:
            p("  FEWER THAN 12 COMMON LATENTS -- the fixed-latent curve is not")
            p("  estimable and no conclusion is drawn about which explanation")
            p("  holds. Report the varying-sample curve with that caveat.")
            p("")
            continue

        p("")
        p("  (1) FIXED LATENT SET -- the sample cannot change across rows")
        p(f"{'n_seq':>7}{'lat':>6}{'per-arm ab%':>13}{'shared ab%':>12}"
          f"{'per-arm Erho':>14}{'shared Erho':>13}{'gain':>8}{'95% CI':>20}")
        fixed = []
        for n in sizes:
            fl = built[n]["fids"]
            idx = np.array([fl.index(f) for f in common])
            A = comp_cube(built[n]["ca"][idx])
            Bc = comp_cube(built[n]["cs"][idx])
            g, lo, hi = paired_gain(built[n]["ca"], built[n]["cs"], idx)
            ci = "[{:+.3f}, {:+.3f}]".format(lo, hi)
            p(f"{n:>7}{len(common):>6}{A['pct_ab']:>12.1f}%{Bc['pct_ab']:>11.1f}%"
              f"{A['erho2']:>14.3f}{Bc['erho2']:>13.3f}{g:>+8.3f}{ci:>20}")
            fixed.append(g)
            rows.append(dict(model=model, n_seq=n, latent_set="fixed",
                             n_lat=len(common), pct_ab_perarm=A["pct_ab"],
                             pct_ab_shared=Bc["pct_ab"], erho_perarm=A["erho2"],
                             erho_shared=Bc["erho2"], gain=g, lo=lo, hi=hi))
        p("")
        d_fix = fixed[-1] - fixed[0]
        p(f"  gain across corpora on the FIXED set: "
          + " -> ".join(f"{g:+.3f}" for g in fixed)
          + f"   (net {d_fix:+.3f})")
        p("  Compare against the varying-sample curve in results/corpus_curve.txt.")
        p("  If the fixed-set gain is flat or falling while the varying-sample")
        p("  gain rises, the rise is sample composition, not data.")

        # ---- (2) who newly clears the filter ---------------------------------
        p("")
        p("  (2) WHAT THE BALANCED-CUBE FILTER NEWLY ADMITS AT 1536")
        base, top = sizes[0], sizes[-1]
        early = set(built[base]["fids"])
        newc = [f for f in built[top]["fids"] if f not in early]
        p(f"  balanced at {base}: {len(early)};  new at {top}: {len(newc)}")
        if len(newc) < 12:
            p("  too few newcomers to characterise; skipped")
        else:
            pa_top = built[top]["pa"]
            cons = consistency(pa_top)
            fires = pa_top.groupby("fid").act.size()
            act = pa_top.groupby("fid").act.median()
            fl = built[top]["fids"]
            p("")
            p(f"{'group':<26}{'lat':>5}{'median act':>13}"
              f"{'positional consist.':>21}{'per-arm ab%':>13}")
            for lab, ids in (("balanced already at %d" % base,
                              [f for f in fl if f in early]),
                             ("new at %d" % top, newc)):
                idx = np.array([fl.index(f) for f in ids])
                c = comp_cube(built[top]["ca"][idx])
                cc = cons.reindex(ids).dropna()
                p(f"{lab:<26}{len(ids):>5}{act.reindex(ids).median():>13.3f}"
                  f"{cc.median():>21.3f}{c['pct_ab']:>12.1f}%")
                rows.append(dict(model=model, n_seq=top, latent_set=lab,
                                 n_lat=len(ids),
                                 median_act=float(act.reindex(ids).median()),
                                 consistency=float(cc.median()),
                                 pct_ab_perarm=c["pct_ab"]))
            p("")
            p("  Positional consistency is mean pairwise Jaccard of a latent's")
            p("  measured position sets across the 15 arm pairs, scored on the")
            p("  per-arm evaluation so it is independent of the shared design.")
        p("")

    pd.DataFrame(rows).to_csv("results/fixed_latent_curve.csv", index=False)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
