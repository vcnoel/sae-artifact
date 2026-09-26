# -*- coding: utf-8 -*-
"""E19: do the two models disagree, or sit at different points on one curve?

THE HYPOTHESIS. Gemma-2 keeps 103/240 latents (43%) under the shared-position
design and shows a paired E rho^2 gain of +0.239. Gemma-3 keeps 56/238 (24%)
and shows +0.050, not distinguishable from zero. Gemma-3's filter is nearly
twice as stringent, and its retained set is correspondingly more extreme
(v_a ratio 4.92x vs 2.43x, error ratio 0.50x vs 0.65x). Its per-arm restricted
design therefore already sits at 0.781, leaving position crossing almost
nothing to recover.

If that is the explanation, the two models are not disagreeing about mechanism.
The size of the position-crossing gain is a DECREASING function of how
positionally consistent the retained set already is, and one curve predicts
both numbers.

THE TEST, run here on data already on disk: within the latents balanced in both
designs, rank by positional consistency and sweep the fraction kept. If
Gemma-2's gain falls toward Gemma-3's as its retention is tightened to match,
the models agree and retention stringency is the moderator.

Consistency is scored from the PER-ARM evaluation -- mean pairwise Jaccard of
each latent's measured position sets across the 15 arm pairs -- so the score is
independent of the shared design being tested.

Also runs (E19b) the n-ladder, since requiring 6 positions per cell discards
43% of Gemma-3's retained latents, and (E19c) the absolute components on both
paired cubes, to find where Gemma-3's interaction went when it fell to zero
without E rho^2 moving.
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

from boundary_check import comp_cube

B = 1000


def consistency(per_arm):
    """Mean pairwise Jaccard of measured position sets across arm pairs."""
    s = (per_arm.groupby(["fid", "arm"])["pos"]
         .apply(lambda x: frozenset(x)).unstack("arm"))
    out = {}
    for fid, row in s.iterrows():
        vals = [v for v in row if isinstance(v, frozenset)]
        if len(vals) < 2:
            continue
        j = [len(a & b) / len(a | b) for a, b in itertools.combinations(vals, 2)]
        out[fid] = float(np.mean(j))
    return pd.Series(out, name="consistency")


def cube_for(d, fids, n):
    """Balanced cube over exactly `fids`, n positions per cell, or None."""
    sub = d[d.fid.isin(fids)]
    cnt = sub.groupby(["fid", "arm"])["y"].count().unstack("arm")
    keep = cnt.index[(cnt >= n).all(axis=1)]
    if len(keep) < 12:
        return None, []
    arms = sorted(sub.arm.unique())
    fl = sorted(keep)
    cube = np.empty((len(fl), len(arms), n))
    rng = np.random.default_rng(0)
    for i, f in enumerate(fl):
        for j, arm in enumerate(arms):
            v = sub[(sub.fid == f) & (sub.arm == arm)]["y"].to_numpy()
            cube[i, j] = v[rng.choice(len(v), n, replace=False)]
    return cube, fl


def paired_diff(ca, cs, B=B):
    rng = np.random.default_rng(0)
    I = ca.shape[0]
    d = []
    for _ in range(B):
        pk = rng.integers(0, I, I)
        a, b = comp_cube(ca[pk])["erho2"], comp_cube(cs[pk])["erho2"]
        if np.isfinite(a) and np.isfinite(b):
            d.append(b - a)
    d = np.array(d)
    return (comp_cube(cs)["erho2"] - comp_cube(ca)["erho2"],
            np.percentile(d, 2.5), np.percentile(d, 97.5),
            float((d <= 0).mean()))


def load(pa_path, sh_path):
    pa, sh = pd.read_csv(pa_path), pd.read_csv(sh_path)
    for d in (pa, sh):
        d["y"] = np.log10(d.kl_per_norm.clip(lower=1e-12))
    return pa, sh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/retention_curve.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    MODELS = [("gemma2-2b", "results/eval_arms_rerun_fixsamp.csv",
               "results/eval_arms_shared_fixsamp.csv"),
              ("gemma3-1b", "results/eval_arms_g3.csv",
               "results/eval_arms_g3_shared.csv")]

    p("=" * 88)
    p("E19 (a)  RETENTION-MATCHED PAIRED TEST: one curve, or two models?")
    p("=" * 88)
    p("  Within the latents balanced in BOTH designs, rank by positional")
    p("  consistency (mean pairwise Jaccard, scored from the per-arm eval) and")
    p("  keep the top fraction. Retention is relative to all evaluated latents.")
    p("")
    p(f"{'model':<11}{'keep':>6}{'lat':>5}{'retention':>11}{'per-arm':>9}"
      f"{'shared':>8}{'diff':>8}{'95% CI':>20}{'frac<=0':>9}")
    for name, pap, shp in MODELS:
        pa, sh = load(pap, shp)
        cons = consistency(pa)
        _, base_f = cube_for(sh, set(sh.fid.unique()), 6)
        _, base_a = cube_for(pa, set(base_f), 6)
        both = sorted(set(base_f) & set(base_a))
        n_all = pa.fid.nunique()
        ranked = cons.reindex(both).sort_values(ascending=False).index.tolist()
        for frac in (1.00, 0.75, 0.55, 0.45, 0.30):
            k = max(12, int(round(len(ranked) * frac)))
            if k > len(ranked):
                continue
            sel = set(ranked[:k])
            ca, fa = cube_for(pa, sel, 6)
            cs, fs = cube_for(sh, sel, 6)
            if ca is None or cs is None or len(fa) != len(fs):
                continue
            e_a, e_s = comp_cube(ca)["erho2"], comp_cube(cs)["erho2"]
            dif, lo, hi, fr = paired_diff(ca, cs)
            p(f"{name:<11}{frac:>6.0%}{len(fa):>5}{len(fa)/n_all:>11.1%}"
              f"{e_a:>9.3f}{e_s:>8.3f}{dif:>+8.3f}"
              f"{'[' + f'{lo:+.3f}, {hi:+.3f}' + ']':>20}{fr:>9.3f}")
    p("")
    p("  Read down each model's block: if the gain shrinks as retention")
    p("  tightens, the two models are one curve and stringency is the moderator.")

    p("")
    p("=" * 88)
    p("E19 (b)  THE n-LADDER -- requiring 6 positions per cell costs latents")
    p("=" * 88)
    p(f"{'model':<11}{'n':>3}{'lat':>5}{'per-arm':>9}{'shared':>8}{'diff':>8}"
      f"{'95% CI':>20}{'frac<=0':>9}")
    for name, pap, shp in MODELS:
        pa, sh = load(pap, shp)
        for n in (3, 4, 5, 6):
            sfids = set(sh.fid.unique())
            ca, fa = cube_for(pa, sfids, n)
            cs, fs = cube_for(sh, sfids, n)
            if ca is None or cs is None:
                continue
            both = sorted(set(fa) & set(fs))
            ca, fa = cube_for(pa, set(both), n)
            cs, fs = cube_for(sh, set(both), n)
            e_a, e_s = comp_cube(ca)["erho2"], comp_cube(cs)["erho2"]
            dif, lo, hi, fr = paired_diff(ca, cs)
            p(f"{name:<11}{n:>3}{len(fa):>5}{e_a:>9.3f}{e_s:>8.3f}{dif:>+8.3f}"
              f"{'[' + f'{lo:+.3f}, {hi:+.3f}' + ']':>20}{fr:>9.3f}")
    p("")
    p("  Latents are the unit of generalisation, so trading within-cell")
    p("  precision for more latents is likely the right trade; the paper should")
    p("  state which n it uses and why.")

    p("")
    p("=" * 88)
    p("E19 (c)  WHERE DID THE INTERACTION GO? absolute components, paired cubes")
    p("=" * 88)
    p(f"{'model':<11}{'design':<20}{'v_a':>9}{'v_ab':>9}{'v_e':>9}"
      f"{'denom':>9}{'Erho2':>8}")
    for name, pap, shp in MODELS:
        pa, sh = load(pap, shp)
        sfids = set(sh.fid.unique())
        ca, fa = cube_for(pa, sfids, 6)
        cs, fs = cube_for(sh, sfids, 6)
        both = sorted(set(fa) & set(fs))
        ca, _ = cube_for(pa, set(both), 6)
        cs, _ = cube_for(sh, set(both), 6)
        for lab, c in (("per-arm restricted", comp_cube(ca)),
                       ("shared positions", comp_cube(cs))):
            den = c["v_a"] + c["v_ab"] + c["v_e"] / c["n"]
            p(f"{name:<11}{lab:<20}{c['v_a']:>9.4f}{c['v_ab']:>9.4f}"
              f"{c['v_e']:>9.4f}{den:>9.4f}{c['erho2']:>8.3f}")
    p("")
    p("  Forcing a common position measures some arms away from where they fire")
    p("  hardest, which should ADD within-cell noise. If v_e rises under the")
    p("  shared design that is a real cost of the fix and belongs in the paper.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
