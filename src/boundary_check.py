# -*- coding: utf-8 -*-
"""E17: is the shared-design latent x arm component a clipped negative, and
how wide is E rho^2 around the 0.80 threshold it is being compared to?

crossed_two_way clips every variance component at zero
(v_ab = max((msab - mse)/n, 0)), which is standard and also silently converts
"the estimator returned a negative number" into "the component is exactly
zero". An estimate of exactly 0.0% is the signature of that, so it has to be
checked rather than quoted.

DIRECTION OF THE BIAS, since it is easy to get backwards. E rho^2 is
    v_a / (v_a + v_ab + v_e/n).
v_ab sits in the DENOMINATOR, so replacing a negative raw estimate with zero
makes the denominator LARGER and E rho^2 SMALLER. Clipping is therefore
conservative here: if the raw component is negative, the clipped 0.808 is a
lower bound on what the unclipped estimator would report, not an inflation of
it. That does not make 0.808 safe to quote -- a component pinned at a boundary
is not a measurement, and the honest statement is an interval.

Also bootstraps E rho^2 over LATENTS (the unit of generalisation) for every row
of the E16 table, because the shared row rests on 70 balanced latents against
180 for the per-arm row and a point estimate sitting on its own comparison
threshold needs a width.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

B = 1000


def to_cube(d, n_target=6):
    """Balanced (latent, arm, position) cube as a numpy array.

    The pandas path rebuilt a DataFrame per bootstrap draw and ran ~1s per
    draw; on a cube, a draw is one row-index operation and the whole
    decomposition is three means. Same estimator, ~1000x faster.
    """
    cnt = d.groupby(["fid", "arm"])["y"].count().unstack("arm")
    for n in (n_target, 5, 4, 3, 2):
        keep = cnt.index[(cnt >= n).all(axis=1)]
        if len(keep) >= 30:
            break
    else:
        return None, None
    sub = d[d.fid.isin(keep)]
    arms = sorted(sub.arm.unique())
    fids = sorted(keep)
    cube = np.empty((len(fids), len(arms), n))
    rng = np.random.default_rng(0)
    for i, f in enumerate(fids):
        for j, arm in enumerate(arms):
            v = sub[(sub.fid == f) & (sub.arm == arm)]["y"].to_numpy()
            cube[i, j] = v[rng.choice(len(v), n, replace=False)]
    return cube, n


def comp_cube(cube, clip=True):
    """Method-of-moments components on a balanced I x J x n cube."""
    I, J, n = cube.shape
    gm = cube.mean()
    ma = cube.mean(axis=(1, 2))
    mb = cube.mean(axis=(0, 2))
    mc = cube.mean(axis=2)
    ssa = J * n * ((ma - gm) ** 2).sum()
    ssb = I * n * ((mb - gm) ** 2).sum()
    ssab = n * ((mc - ma[:, None] - mb[None, :] + gm) ** 2).sum()
    sse = ((cube - mc[:, :, None]) ** 2).sum()
    msa, msb = ssa / (I - 1), ssb / (J - 1)
    msab = ssab / ((I - 1) * (J - 1))
    mse = sse / (I * J * (n - 1))
    f = (lambda v: max(v, 0.0)) if clip else (lambda v: v)
    v_e = mse
    v_ab = f((msab - mse) / n)
    v_a = f((msa - msab) / (J * n))
    v_b = f((msb - msab) / (I * n))
    tot = v_a + v_b + v_ab + v_e
    den = v_a + v_ab + v_e / n
    return dict(I=I, J=J, n=n, v_a=v_a, v_b=v_b, v_ab=v_ab, v_e=v_e,
                pct_ab=100 * v_ab / tot if tot else float("nan"),
                erho2=v_a / den if den > 0 else float("nan"),
                msa=msa, msb=msb, msab=msab, mse=mse)


def components(d, a="fid", b="arm", y="y", n_target=6, clip=True):
    """Method-of-moments components. clip=False exposes raw negatives."""
    d = d[[a, b, y]].dropna()
    cnt = d.groupby([a, b])[y].count().unstack(b)
    for n in (n_target, 5, 4, 3, 2):
        keep = cnt.index[(cnt >= n).all(axis=1)]
        if len(keep) >= 30:
            break
    else:
        return None
    d = d[d[a].isin(keep)]
    d = pd.concat([g.sample(n, random_state=0) for _, g in d.groupby([a, b])])
    la, lb = sorted(d[a].unique()), sorted(d[b].unique())
    I, J = len(la), len(lb)
    gm = d[y].mean()
    ma, mb = d.groupby(a)[y].mean(), d.groupby(b)[y].mean()
    mc = d.groupby([a, b])[y].mean()
    ssa = J * n * ((ma - gm) ** 2).sum()
    ssb = I * n * ((mb - gm) ** 2).sum()
    ssab = n * sum((mc.loc[(ai, bi)] - ma[ai] - mb[bi] + gm) ** 2
                   for ai in la for bi in lb)
    sse = sum(((g[y] - mc.loc[(ai, bi)]) ** 2).sum()
              for (ai, bi), g in d.groupby([a, b]))
    msa, msb = ssa / (I - 1), ssb / (J - 1)
    msab = ssab / ((I - 1) * (J - 1))
    mse = sse / (I * J * (n - 1))
    f = (lambda v: max(v, 0)) if clip else (lambda v: v)
    v_e = mse
    v_ab = f((msab - mse) / n)
    v_a = f((msa - msab) / (J * n))
    v_b = f((msb - msab) / (I * n))
    tot = v_a + v_b + v_ab + v_e
    den = v_a + v_ab + v_e / n
    return dict(I=I, J=J, n=n, v_a=v_a, v_b=v_b, v_ab=v_ab, v_e=v_e,
                pct_ab=100 * v_ab / tot, pct_a=100 * v_a / tot,
                erho2=v_a / den if den > 0 else float("nan"),
                msab=msab, mse=mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/boundary_check.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    shared = pd.read_csv("results/eval_arms_shared_fixsamp.csv")
    per_arm = pd.read_csv("results/eval_arms_rerun_fixsamp.csv")
    sfids = set(shared.fid.unique())
    for d in (shared, per_arm):
        d["y"] = np.log10(d.kl_per_norm.clip(lower=1e-12))

    designs = [("per-arm, all latents", per_arm),
               ("per-arm, restricted to shared latents",
                per_arm[per_arm.fid.isin(sfids)]),
               ("SHARED positions", shared)]

    p("=" * 84)
    p("E17 (1)  IS latent x arm A CLIPPED NEGATIVE?")
    p("=" * 84)
    p(f"{'design':<40}{'v_ab clipped':>14}{'v_ab RAW':>12}{'MS_ab':>10}{'MSE':>10}")
    for lab, d in designs:
        c = components(d, clip=True)
        r = components(d, clip=False)
        p(f"{lab:<40}{c['v_ab']:>14.5f}{r['v_ab']:>12.5f}"
          f"{r['msab']:>10.4f}{r['mse']:>10.4f}")
    rs = components(shared, clip=False)
    p("")
    if rs["v_ab"] < 0:
        p(f"  CONFIRMED: the shared design's raw latent x arm component is "
          f"NEGATIVE ({rs['v_ab']:+.5f}),")
        p("  i.e. MS_ab < MSE -- latents vary no more across arms than positions")
        p("  vary within a cell. Clipped to 0 for reporting.")
        p("")
        p("  Direction of the resulting bias in E rho^2, which is easy to get")
        p("  backwards: v_ab is in the DENOMINATOR, so clipping a negative up to")
        p("  zero ENLARGES the denominator and SHRINKS E rho^2.")
        cs = components(shared, clip=True)
        p(f"    E rho^2 clipped (reported) = {cs['erho2']:.3f}")
        p(f"    E rho^2 unclipped          = {rs['erho2']:.3f}")
        p("  So the clipped figure is the conservative one. The honest statement")
        p("  is still not a point estimate: a component pinned at a boundary is")
        p("  'not distinguishable from zero', and the interval below is what")
        p("  should be quoted.")
    else:
        p(f"  Raw component is positive ({rs['v_ab']:+.5f}); no clipping "
          f"occurred and the point estimate stands.")

    p("")
    p("=" * 84)
    p("E17 (2)  BOOTSTRAP OVER LATENTS")
    p("=" * 84)
    p("  Resampling LATENTS with replacement (the unit of generalisation),")
    p(f"  {B} draws, components recomputed per draw.")
    p("")
    p(f"{'design':<40}{'bal.lat':>8}{'Erho2':>8}{'95% CI':>18}"
      f"{'P(>0.8)':>9}{'v_ab<0':>8}")
    rng = np.random.default_rng(0)
    for lab, d in designs:
        cube, n = to_cube(d)
        if cube is None:
            p(f"{lab:<40}  insufficient replication")
            continue
        base = comp_cube(cube, clip=True)
        I = cube.shape[0]
        er, neg = [], 0
        for _ in range(B):
            c = comp_cube(cube[rng.integers(0, I, I)], clip=True)
            raw = comp_cube(cube[rng.integers(0, I, I)], clip=False)
            if np.isfinite(c["erho2"]):
                er.append(c["erho2"])
            neg += int(raw["v_ab"] < 0)
        er = np.array(er)
        lo, hi = np.percentile(er, [2.5, 97.5])
        p(f"{lab:<40}{I:>8}{base['erho2']:>8.3f}"
          f"{'[' + f'{lo:.3f}, {hi:.3f}' + ']':>18}"
          f"{float((er > 0.8).mean()):>9.2f}{neg / B:>8.2f}")
    p("")
    p("  'v_ab<0' is the fraction of draws whose RAW latent x arm component is")
    p("  negative. Near 0.5 means the component is sitting on the boundary and")
    p("  should be reported as indistinguishable from zero, not as a value.")

    p("")
    p("=" * 84)
    p("E17 (3)  WHAT DOES THE SHARED DESIGN KEEP?")
    p("=" * 84)
    allf = set(per_arm.fid.unique())
    kept, dropped = sfids, allf - sfids
    p(f"  retained {len(kept)} of {len(allf)} latents "
      f"({100*len(kept)/len(allf):.0f}%)")
    p("  survivors are latents live in ALL six arms that additionally share at")
    p("  least one firing position across all six.")
    p("")
    g = (per_arm.groupby("fid")
         .agg(act_med=("act", "median"), act_max=("act", "max"),
              kpn=("kl_per_norm", "median"), n_obs=("act", "size")))
    g["kept"] = g.index.isin(kept)
    p(f"{'':<14}{'median act':>12}{'max act':>10}{'median kpn':>12}{'n obs':>8}")
    for k, sub in g.groupby("kept"):
        p(f"{'RETAINED' if k else 'DROPPED':<14}{sub.act_med.median():>12.3f}"
          f"{sub.act_max.median():>10.3f}{sub.kpn.median():>12.6f}"
          f"{sub.n_obs.median():>8.0f}")
    from scipy import stats as st
    for col in ("act_med", "act_max", "kpn"):
        a_, b_ = g[g.kept][col], g[~g.kept][col]
        p(f"  Mann-Whitney {col:<9} p={st.mannwhitneyu(a_, b_).pvalue:.2g}")
    p("")
    p("  The retained set is a SUBPOPULATION, not a sample: any claim built on")
    p("  the shared design is a claim about positionally consistent latents.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
