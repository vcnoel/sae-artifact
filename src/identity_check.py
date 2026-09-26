# -*- coding: utf-8 -*-
"""E21: is gain = 6.02 x v_ab a mechanism, or the definition of E rho^2?

THE PROBLEM. E rho^2 = v_a / (v_a + v_ab + v_e/n), and the shared design drives
v_ab to zero on every subset. So removing v_ab necessarily raises E rho^2, and a
positive gain-vs-v_ab correlation is guaranteed by the algebra before any
mechanism is invoked. r = +0.917 is not evidence until it is shown to exceed
what the identity alone predicts.

THE DECOMPOSITION. For each subset, with per-arm components (v_a, v_ab, v_e):

    predicted gain  =  v_a/(v_a + v_e/n)  -  v_a/(v_a + v_ab + v_e/n)

i.e. what you get from zeroing v_ab with v_a and v_e HELD AT THEIR PER-ARM
VALUES. The observed gain also includes whatever v_a and v_e actually did:

    excess = observed - predicted

If observed tracks predicted, the v_ab relationship restates the definition and
the only real finding is the v_a rise. If observed exceeds predicted
systematically, the excess IS the v_a rise, quantified.

ALSO (E21b) the nested-subset problem. The ten points of E20(2) are five nested
subsets per model, so the effective sample is nearer two and p=0.00018 is not
quotable. Independent evidence here: disjoint partitions of the latents, plus a
bootstrap interval on the slope.

ALSO (E21c) the compression account of the reversed v_a result. Weak-rho latents
showed the LARGER v_a rise on both models. Candidate: where activation does not
predict effect, per-arm top-activation selection is close to random position
selection, which adds noise to each latent's mean and compresses between-latent
differences; where activation does predict effect, per-arm and shared selection
nearly coincide and there is little to gain. Prediction: per-arm/shared position
overlap is HIGHER for strong-rho latents.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

from boundary_check import comp_cube
from moderator import MODELS, build, FRACS, NPOS
from retention_curve import consistency


def erho2(v_a, v_ab, v_e, n):
    den = v_a + v_ab + v_e / n
    return v_a / den if den > 0 else float("nan")


def point(ca, cs, idx):
    a, b = comp_cube(ca[idx]), comp_cube(cs[idx])
    obs = b["erho2"] - a["erho2"]
    # zero v_ab, hold v_a and v_e at their per-arm values
    pred_hi = erho2(a["v_a"], 0.0, a["v_e"], a["n"])
    pred = pred_hi - a["erho2"]
    return dict(v_ab=a["v_ab"], obs=obs, pred=pred, excess=obs - pred,
                va_pa=a["v_a"], va_sh=b["v_a"], ve_pa=a["v_e"], ve_sh=b["v_e"],
                n=len(idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/identity_check.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    cache = {name: build(pap, shp) for name, pap, shp in MODELS}

    # ---------- (a) identity check ----------
    p("=" * 92)
    p("E21 (a)  IS THE v_ab RELATIONSHIP ANYTHING BEYOND THE DEFINITION?")
    p("=" * 92)
    p(f"{'model':<11}{'frac':>6}{'lat':>5}{'v_ab':>9}{'observed':>10}"
      f"{'predicted':>11}{'excess':>9}{'v_a rise':>10}")
    obs_l, pred_l, exc_l, tags = [], [], [], []
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        cons = consistency(pa).reindex(fids)
        order = np.argsort(-cons.to_numpy())
        for f in FRACS:
            k = max(12, int(round(len(fids) * f)))
            r = point(ca, cs, order[:k])
            obs_l.append(r["obs"]); pred_l.append(r["pred"])
            exc_l.append(r["excess"]); tags.append(name)
            p(f"{name:<11}{f:>6.0%}{k:>5}{r['v_ab']:>9.4f}{r['obs']:>+10.3f}"
              f"{r['pred']:>+11.3f}{r['excess']:>+9.3f}"
              f"{r['va_sh']/max(r['va_pa'],1e-9):>9.2f}x")
    obs_a, pred_a, exc_a = map(np.array, (obs_l, pred_l, exc_l))
    rr = stats.pearsonr(pred_a, obs_a)
    p("")
    p(f"  correlation(predicted, observed) = {rr.statistic:+.3f}")
    p(f"  mean observed  = {obs_a.mean():+.4f}")
    p(f"  mean predicted = {pred_a.mean():+.4f}  "
      f"({100*pred_a.mean()/max(obs_a.mean(),1e-9):.0f}% of observed)")
    p(f"  mean excess    = {exc_a.mean():+.4f}  "
      f"({100*exc_a.mean()/max(obs_a.mean(),1e-9):.0f}% of observed)")
    p(f"  excess > 0 in {int((exc_a > 0).sum())}/{len(exc_a)} subsets")
    t = stats.wilcoxon(exc_a)
    p(f"  Wilcoxon on excess: p={t.pvalue:.3g}")
    p("")
    if pred_a.mean() / max(obs_a.mean(), 1e-9) > 0.85:
        p("  VERDICT: the gain is essentially the identity. Removing v_ab")
        p("  explains almost all of it, so 'gain = 6.02 x v_ab' RESTATES the")
        p("  definition of E rho^2 and must not be reported as a mechanism.")
    else:
        p("  VERDICT: the identity explains only part of the gain; the excess")
        p("  is the v_a rise and THAT is the finding.")

    # ---------- (b) independent points ----------
    p("")
    p("=" * 92)
    p("E21 (b)  DISJOINT PARTITIONS (the E20 sweep was nested; this is not)")
    p("=" * 92)
    p(f"{'model':<11}{'part':>6}{'lat':>5}{'v_ab':>9}{'observed':>10}"
      f"{'predicted':>11}{'excess':>9}")
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        K = 4 if len(fids) >= 64 else 2
        perm = rng.permutation(len(fids))
        for q, idx in enumerate(np.array_split(perm, K)):
            if len(idx) < 12:
                continue
            r = point(ca, cs, idx)
            xs.append(r["v_ab"]); ys.append(r["obs"])
            p(f"{name:<11}{q + 1:>6}{len(idx):>5}{r['v_ab']:>9.4f}"
              f"{r['obs']:>+10.3f}{r['pred']:>+11.3f}{r['excess']:>+9.3f}")
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) >= 4:
        rr2 = stats.pearsonr(xs, ys)
        p("")
        p(f"  {len(xs)} INDEPENDENT points: Pearson r = {rr2.statistic:+.3f} "
          f"(p={rr2.pvalue:.3g})")
    # bootstrap the slope over latents
    p("")
    p("  bootstrap over latents, slope of observed gain on per-arm v_ab:")
    sl = []
    for _ in range(400):
        bx, by = [], []
        for name, _, _ in MODELS:
            pa, sh, ca, cs, fids = cache[name]
            I = ca.shape[0]
            for f in (1.00, 0.55):
                k = max(12, int(round(I * f)))
                pick = rng.integers(0, I, k)
                r = point(ca, cs, pick)
                bx.append(r["v_ab"]); by.append(r["obs"])
        if len(set(np.round(bx, 6))) > 1:
            sl.append(np.polyfit(bx, by, 1)[0])
    sl = np.array(sl)
    p(f"    slope = {np.median(sl):+.2f}  95% CI "
      f"[{np.percentile(sl,2.5):+.2f}, {np.percentile(sl,97.5):+.2f}]")
    p("    (p = 0.00018 from the nested sweep is withdrawn and not quoted.)")

    # ---------- (c) compression account ----------
    p("")
    p("=" * 92)
    p("E21 (c)  WHY DO WEAK-rho LATENTS GAIN MORE v_a? position-set overlap")
    p("=" * 92)
    p("  Prediction: per-arm and shared position sets overlap MORE for")
    p("  strong-rho latents, because where activation predicts effect the two")
    p("  selection rules nearly coincide.")
    p("")
    p(f"{'model':<11}{'group':<20}{'lat':>5}{'Jaccard(per-arm, shared)':>26}"
      f"{'p':>10}")
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        rho = {}
        for f, g in pa[pa.fid.isin(fids)].groupby("fid"):
            if g.act.nunique() > 2:
                rho[f] = stats.spearmanr(g.act, g.kl_per_norm).statistic
        rr3 = pd.Series(rho).reindex(fids)
        med = rr3.median()
        pos_pa = pa[pa.fid.isin(fids)].groupby(["fid", "arm"])["pos"].apply(
            lambda x: frozenset(x))
        pos_sh = sh[sh.fid.isin(fids)].groupby(["fid", "arm"])["pos"].apply(
            lambda x: frozenset(x))
        jac = {}
        for f in fids:
            vals = []
            for arm in sorted(pa.arm.unique()):
                a_ = pos_pa.get((f, arm))
                b_ = pos_sh.get((f, arm))
                if a_ and b_:
                    vals.append(len(a_ & b_) / len(a_ | b_))
            if vals:
                jac[f] = float(np.mean(vals))
        jj = pd.Series(jac).reindex(fids)
        grp = {}
        for lab, m in (("STRONG rho", rr3 > med), ("WEAK rho", rr3 <= med)):
            v = jj[m.reindex(jj.index).fillna(False)].dropna()
            grp[lab] = v
            p(f"{name:<11}{lab:<20}{len(v):>5}{v.mean():>26.3f}"
              f"{'':>10}")
        if len(grp) == 2 and all(len(v) > 3 for v in grp.values()):
            pv = stats.mannwhitneyu(grp["STRONG rho"], grp["WEAK rho"]).pvalue
            p(f"{'':<11}{'-> Mann-Whitney':<20}{'':>5}{'':>26}{pv:>10.3g}")
    p("")
    p("  If strong-rho overlap is higher, per-arm selection is near-random")
    p("  exactly where activation fails to predict effect, and that is why")
    p("  those latents gain the most from a fixed position.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
