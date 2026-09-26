# -*- coding: utf-8 -*-
"""E18: (a) paired interval on the E rho^2 gain, (b) is the gain measurement
precision or variance inflation?

(a) THE PAIRED TEST. The shared-position and per-arm-restricted designs are
computed on the SAME latents, so comparing two marginal bootstrap intervals and
noting they do not overlap is a conservative substitute for the test actually
available: resample latents ONCE per draw, apply that draw to both designs, and
take the difference. Reported as a point estimate, an interval, and the fraction
of draws below zero.

(b) THE SUBPOPULATION CLAIM. The shared design keeps 43% of latents, and those
have systematically LARGER causal effect -- which is the dependent variable.
E rho^2 = v_a / (v_a + v_ab + v_e/n) rises mechanically when between-latent true
variance v_a rises, with no improvement in measurement precision at all. So
"the protocol identifies the latents whose numbers are worth trusting" is only
supportable if the ERROR terms fall too. This prints v_a, v_ab and v_e as
absolute variances for retained and dropped latents under the SAME (per-arm)
design, where the only difference is which latents are in the set.

The like-for-like comparison in (a) is immune to this, since both designs use
the same latents; only the external subpopulation claim is at risk.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

from boundary_check import to_cube, comp_cube

B = 2000


def aligned_cubes(per_arm, shared):
    """Balanced cubes for both designs on the SAME latent set, row-aligned."""
    ca, na = to_cube(per_arm)
    cs, ns = to_cube(shared)
    fa = sorted(per_arm[per_arm.fid.isin(
        _kept(per_arm, na))].fid.unique())
    fs = sorted(shared[shared.fid.isin(_kept(shared, ns))].fid.unique())
    common = sorted(set(fa) & set(fs))
    ia = [fa.index(f) for f in common]
    isx = [fs.index(f) for f in common]
    return ca[ia], cs[isx], common


def _kept(d, n):
    cnt = d.groupby(["fid", "arm"])["y"].count().unstack("arm")
    return cnt.index[(cnt >= n).all(axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_arm",
                    default="results/eval_arms_rerun_fixsamp.csv")
    ap.add_argument("--shared",
                    default="results/eval_arms_shared_fixsamp.csv")
    ap.add_argument("--out", default="results/paired_erho2.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    pa = pd.read_csv(A.per_arm)
    sh = pd.read_csv(A.shared)
    for d in (pa, sh):
        d["y"] = np.log10(d.kl_per_norm.clip(lower=1e-12))
    sfids = set(sh.fid.unique())
    pa_r = pa[pa.fid.isin(sfids)].copy()

    # ---------------- (a) paired difference ----------------
    p("=" * 84)
    p("E18 (a)  PAIRED INTERVAL ON THE E rho^2 GAIN (common latent resample)")
    p("=" * 84)
    ca, cs, common = aligned_cubes(pa_r, sh)
    p(f"  {len(common)} latents balanced in BOTH designs; "
      f"cubes {ca.shape} (per-arm) and {cs.shape} (shared)")
    b_r, b_s = comp_cube(ca)["erho2"], comp_cube(cs)["erho2"]
    p(f"  E rho^2 per-arm restricted = {b_r:.3f}")
    p(f"  E rho^2 shared positions   = {b_s:.3f}")
    p(f"  observed difference        = {b_s - b_r:+.3f}")
    rng = np.random.default_rng(0)
    I = ca.shape[0]
    diffs = []
    for _ in range(B):
        pick = rng.integers(0, I, I)           # ONE draw, both designs
        d_r = comp_cube(ca[pick])["erho2"]
        d_s = comp_cube(cs[pick])["erho2"]
        if np.isfinite(d_r) and np.isfinite(d_s):
            diffs.append(d_s - d_r)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p("")
    p(f"  paired difference: {np.median(diffs):+.3f}  "
      f"95% CI [{lo:+.3f}, {hi:+.3f}]  ({B} draws)")
    p(f"  fraction of draws <= 0: {float((diffs <= 0).mean()):.4f}")
    p("")
    p("  This replaces the non-overlap-of-marginals argument everywhere.")

    # ---------------- (b) retained vs dropped, absolute ----------------
    p("")
    p("=" * 84)
    p("E18 (b)  IS THE RETAINED SET'S ADVANTAGE PRECISION, OR VARIANCE?")
    p("=" * 84)
    p("  Both rows use the SAME per-arm design; only the latent set differs.")
    p("  v_a is between-latent TRUE variance (the numerator); v_ab and v_e are")
    p("  error terms. A gain that is all v_a is variance inflation, not a")
    p("  better measurement.")
    p("")
    p(f"{'latent set':<16}{'n':>5}{'v_a':>10}{'v_ab':>10}{'v_e':>10}"
      f"{'Erho2':>9}")
    rows = {}
    for lab, sub in (("RETAINED", pa[pa.fid.isin(sfids)]),
                     ("DROPPED", pa[~pa.fid.isin(sfids)])):
        cube, n = to_cube(sub)
        if cube is None:
            p(f"{lab:<16}  insufficient replication")
            continue
        c = comp_cube(cube)
        rows[lab] = c
        p(f"{lab:<16}{cube.shape[0]:>5}{c['v_a']:>10.4f}{c['v_ab']:>10.4f}"
          f"{c['v_e']:>10.4f}{c['erho2']:>9.3f}")
    if len(rows) == 2:
        R, D = rows["RETAINED"], rows["DROPPED"]
        p("")
        for k, lab in (("v_a", "between-latent (numerator)"),
                       ("v_ab", "latent x arm (error)"),
                       ("v_e", "position within cell (error)")):
            ratio = R[k] / D[k] if D[k] else float("nan")
            p(f"    {lab:<32} retained/dropped = {ratio:.2f}x")
        err_R = R["v_ab"] + R["v_e"]
        err_D = D["v_ab"] + D["v_e"]
        p(f"    {'total error (v_ab + v_e)':<32} retained/dropped = "
          f"{err_R / err_D:.2f}x")
        p("")
        if R["v_a"] > D["v_a"] and err_R >= err_D:
            p("  VERDICT: the advantage is LARGER TRUE VARIANCE with no reduction")
            p("  in error -- variance inflation. The subpopulation claim must NOT")
            p("  go in the abstract; report it as an observation about which")
            p("  latents the design retains.")
        elif err_R < err_D and R["v_a"] > D["v_a"]:
            p("  VERDICT: BOTH -- larger true variance AND smaller error. The")
            p("  subpopulation claim stands and is strong.")
        else:
            p("  VERDICT: mixed; read the ratios above before claiming anything.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
