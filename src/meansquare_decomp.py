# -*- coding: utf-8 -*-
"""E22: does the v_a rise survive the msab coupling? And is the excess
size-dependent?

THE COUPLING. v_a and v_ab are not independent estimates: both are functions of
msab.

    v_ab = (msab - mse) / n
    v_a  = (msa - msab) / (J n)

The shared design lowers msab by construction. Because v_a SUBTRACTS msab, that
mechanically raises v_a. So "v_ab fell and v_a rose" is partly one event seen
twice, and E21's identity prediction held v_a fixed at its per-arm value --
exactly the quantity that cannot be held fixed when msab moves.

The correct decomposition runs in MEAN-SQUARE space:

    delta v_a  =  delta msa / (J n)   -   delta msab / (J n)
                  \_____________/        \______________/
                   genuine signal          coupling artifact

If msa is essentially unchanged and the whole v_a rise is -delta msab/(Jn), the
excess is an artifact and there is no finding left. If msa genuinely rises, the
effect is real.

Also (E22b) regresses the excess on subset latent count, because on the nested
sweep it trends down on Gemma-2 (0.102 -> 0.047 as latents fall 70 -> 21) and
UP on Gemma-3 (-0.002 -> 0.047 as latents fall 32 -> 12). Opposite directions
is not a clean size artifact, but a 4-of-6 count on disjoint partitions
spanning -0.025 to +0.216 is thin enough to check rather than assume.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
from scipy import stats

from boundary_check import comp_cube
from moderator import MODELS, build, FRACS
from retention_curve import consistency
from identity_check import point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/meansquare_decomp.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    cache = {name: build(pap, shp) for name, pap, shp in MODELS}

    p("=" * 96)
    p("E22 (a)  MEAN-SQUARE DECOMPOSITION: is the v_a rise real or msab coupling?")
    p("=" * 96)
    p(f"{'model':<11}{'design':<12}{'msa':>9}{'msab':>9}{'mse':>9}"
      f"{'v_a':>9}{'v_ab':>9}")
    rows = {}
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        idx = np.arange(ca.shape[0])
        a, b = comp_cube(ca[idx]), comp_cube(cs[idx])
        rows[name] = (a, b)
        for lab, c in (("per-arm", a), ("shared", b)):
            p(f"{name:<11}{lab:<12}{c['msa']:>9.4f}{c['msab']:>9.4f}"
              f"{c['mse']:>9.4f}{c['v_a']:>9.4f}{c['v_ab']:>9.4f}")
    p("")
    p(f"{'model':<11}{'d msa':>10}{'d msab':>10}{'d mse':>10}"
      f"{'d v_a':>10}{'signal':>10}{'coupling':>10}{'signal %':>10}")
    for name, (a, b) in rows.items():
        J, n = a["J"], a["n"]
        d_msa = b["msa"] - a["msa"]
        d_msab = b["msab"] - a["msab"]
        d_mse = b["mse"] - a["mse"]
        d_va = b["v_a"] - a["v_a"]
        sig = d_msa / (J * n)
        cpl = -d_msab / (J * n)
        share = 100 * sig / d_va if abs(d_va) > 1e-12 else float("nan")
        p(f"{name:<11}{d_msa:>+10.4f}{d_msab:>+10.4f}{d_mse:>+10.4f}"
          f"{d_va:>+10.4f}{sig:>+10.4f}{cpl:>+10.4f}{share:>9.0f}%")
    p("")
    p("  'signal' is delta msa / (J n): the part of the v_a rise that is NOT")
    p("  the mechanical consequence of msab falling. 'coupling' is the rest.")
    p("")
    for name, (a, b) in rows.items():
        J, n = a["J"], a["n"]
        sig = (b["msa"] - a["msa"]) / (J * n)
        cpl = -(b["msab"] - a["msab"]) / (J * n)
        if sig <= 0:
            p(f"  {name}: msa did NOT rise (delta = {b['msa']-a['msa']:+.4f}). "
              f"The entire v_a increase is msab coupling -- ARTIFACT.")
        elif sig > abs(cpl):
            p(f"  {name}: msa rose by {b['msa']-a['msa']:+.4f}; genuine signal "
              f"exceeds coupling. The v_a rise is REAL.")
        else:
            p(f"  {name}: msa rose by {b['msa']-a['msa']:+.4f} but coupling "
              f"({cpl:+.4f}) is larger. Mostly artifact.")

    # ---- recompute the excess with the coupling correction ----
    p("")
    p("  EXCESS AFTER THE COUPLING CORRECTION")
    p(f"{'model':<11}{'excess (E21)':>14}{'coupling in Erho2':>20}"
      f"{'corrected excess':>19}")
    for name, (a, b) in rows.items():
        J, n = a["J"], a["n"]
        cpl_va = -(b["msab"] - a["msab"]) / (J * n)
        # E rho^2 the shared design would show if v_a had risen ONLY by coupling
        va_cpl = a["v_a"] + cpl_va
        den = va_cpl + b["v_ab"] + b["v_e"] / n
        er_cpl = va_cpl / den if den > 0 else float("nan")
        exc = b["erho2"] - a["erho2"] - (
            a["v_a"] / (a["v_a"] + 0.0 + a["v_e"] / n) - a["erho2"])
        cpl_eff = er_cpl - a["v_a"] / (a["v_a"] + 0.0 + a["v_e"] / n)
        p(f"{name:<11}{exc:>+14.4f}{cpl_eff:>+20.4f}{exc - cpl_eff:>+19.4f}")
    p("")
    p("  If the corrected excess is near zero the v_a finding does not survive.")

    # ---- (b) excess vs subset size ----
    p("")
    p("=" * 96)
    p("E22 (b)  IS THE EXCESS SIZE-DEPENDENT?")
    p("=" * 96)
    ns, exs, tags = [], [], []
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        cons = consistency(pa).reindex(fids)
        order = np.argsort(-cons.to_numpy())
        for f in FRACS:
            k = max(12, int(round(len(fids) * f)))
            r = point(ca, cs, order[:k])
            ns.append(k); exs.append(r["excess"]); tags.append(name)
    rng = np.random.default_rng(0)
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        K = 4 if len(fids) >= 64 else 2
        perm = rng.permutation(len(fids))
        for idx in np.array_split(perm, K):
            if len(idx) >= 12:
                r = point(ca, cs, idx)
                ns.append(len(idx)); exs.append(r["excess"]); tags.append(name)
    ns, exs = np.array(ns, float), np.array(exs)
    b1, b0 = np.polyfit(ns, exs, 1)
    rr = stats.pearsonr(ns, exs)
    p(f"  n = {len(ns)} points (10 nested + {len(ns)-10} disjoint), both models")
    p(f"  excess = {b1:+.5f} * latents {b0:+.4f}")
    p(f"  Pearson r = {rr.statistic:+.3f} (p={rr.pvalue:.3g})")
    for name in ("gemma2-2b", "gemma3-1b"):
        m = np.array([t == name for t in tags])
        if m.sum() > 2:
            s = np.polyfit(ns[m], exs[m], 1)[0]
            p(f"    within {name:<10} slope = {s:+.5f}")
    p("")
    p(f"  excess > 0 in {int((exs>0).sum())}/{len(exs)}; "
      f"median {np.median(exs):+.4f}, range [{exs.min():+.4f}, {exs.max():+.4f}]")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
