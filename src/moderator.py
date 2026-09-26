# -*- coding: utf-8 -*-
"""E20: is positional consistency really the moderator, and what is the
mechanism behind the v_a rise?

(1) NULL CONTROL FOR THE RETENTION SWEEP. E19(a) swept the top-f fraction of
latents by positional consistency and found the paired gain declining
monotonically on Gemma-2. But those five points are NESTED SUBSETS of one
sample, so monotonicity is close to automatic if a few low-consistency latents
carry the effect. The control: rank by a RANDOM score and sweep the identical
fractions, many times. If the gain declines under random ranking too, the E19
trend is a sample-size artifact and consistency is not established as the
moderator.

(2) THE MECHANISM VARIABLE IS v_ab, NOT RETENTION. The gain should be a
function of how much latent x arm there was to remove. Gemma-2 per-arm
v_ab = 0.0363 with gain +0.245; Gemma-3 v_ab = 0.0134 with gain +0.050. If
every subset point from both models falls on one line in (v_ab, gain), that is
the unified claim, and it explains why matching retention did not match the
baseline level.

(3) WHY DOES v_a RISE AT ALL? Re-measurement does not normally increase signal.
The candidate: per-arm selection takes each latent's TOP-ACTIVATING position,
activation correlates with the outcome, so every arm measures every latent near
its own local maximum -- compressing genuine between-latent differences toward
a common ceiling. That is selection on a correlate of the dependent variable,
and it predicts v_a should rise most where the activation-to-outcome
correlation is strongest. Tested here by splitting latents on that correlation.

Cubes are built once per model and subsets taken by row indexing: every latent
in the full cube is already balanced in both designs, so any subset of its rows
is still balanced, and this makes the sweep affordable.
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
from scipy import stats

from boundary_check import comp_cube
from retention_curve import consistency, cube_for, load

FRACS = (1.00, 0.75, 0.55, 0.45, 0.30)
N_RANDOM = 20
NPOS = 6

MODELS = [("gemma2-2b", "results/eval_arms_rerun_fixsamp.csv",
           "results/eval_arms_shared_fixsamp.csv"),
          ("gemma3-1b", "results/eval_arms_g3.csv",
           "results/eval_arms_g3_shared.csv")]


def build(pap, shp):
    """Aligned per-arm and shared cubes over latents balanced in both."""
    pa, sh = load(pap, shp)
    sfids = set(sh.fid.unique())
    _, fa = cube_for(pa, sfids, NPOS)
    _, fs = cube_for(sh, sfids, NPOS)
    both = sorted(set(fa) & set(fs))
    ca, f1 = cube_for(pa, set(both), NPOS)
    cs, f2 = cube_for(sh, set(both), NPOS)
    assert f1 == f2
    return pa, sh, ca, cs, f1


def gain(ca, cs, idx):
    a = comp_cube(ca[idx])
    b = comp_cube(cs[idx])
    return b["erho2"] - a["erho2"], a, b


def slope(xs, ys):
    return float(np.polyfit(xs, ys, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/moderator.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    cache = {}
    for name, pap, shp in MODELS:
        cache[name] = build(pap, shp)

    # ---------------- (1) null control ----------------
    p("=" * 86)
    p("E20 (1)  NULL CONTROL: is the retention trend about CONSISTENCY?")
    p("=" * 86)
    p("  Sweep the top-f fraction ranked by positional consistency, then by")
    p(f"  {N_RANDOM} random rankings over the identical fractions. Slope is")
    p("  gain regressed on fraction kept (positive = gain grows with f).")
    p("")
    p(f"{'model':<11}{'consistency slope':>19}{'random slope mean':>19}"
      f"{'random 95% range':>22}{'percentile':>12}")
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        cons = consistency(pa).reindex(fids)
        order = np.argsort(-cons.to_numpy())
        gs = []
        for f in FRACS:
            k = max(12, int(round(len(fids) * f)))
            gs.append(gain(ca, cs, order[:k])[0])
        s_obs = slope(FRACS, gs)
        rng = np.random.default_rng(0)
        s_rand = []
        for _ in range(N_RANDOM):
            perm = rng.permutation(len(fids))
            g2 = []
            for f in FRACS:
                k = max(12, int(round(len(fids) * f)))
                g2.append(gain(ca, cs, perm[:k])[0])
            s_rand.append(slope(FRACS, g2))
        s_rand = np.array(s_rand)
        pct = 100.0 * float((s_rand < s_obs).mean())
        rng_txt = "[{:+.4f}, {:+.4f}]".format(np.percentile(s_rand, 2.5),
                                              np.percentile(s_rand, 97.5))
        p(f"{name:<11}{s_obs:>+19.4f}{s_rand.mean():>+19.4f}"
          f"{rng_txt:>22}{pct:>11.0f}%")
    p("")
    p("  A consistency slope inside the random range means the E19 trend is a")
    p("  subset-size artifact and consistency is NOT established as moderator.")

    # ---------------- (2) gain vs v_ab ----------------
    p("")
    p("=" * 86)
    p("E20 (2)  THE MECHANISM VARIABLE: gain against per-arm v_ab")
    p("=" * 86)
    p(f"{'model':<11}{'frac':>6}{'lat':>5}{'per-arm v_ab':>14}{'gain':>9}")
    xs, ys, tags = [], [], []
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        cons = consistency(pa).reindex(fids)
        order = np.argsort(-cons.to_numpy())
        for f in FRACS:
            k = max(12, int(round(len(fids) * f)))
            g, a, b = gain(ca, cs, order[:k])
            xs.append(a["v_ab"]); ys.append(g); tags.append(name)
            p(f"{name:<11}{f:>6.0%}{k:>5}{a['v_ab']:>14.4f}{g:>+9.3f}")
    xs, ys = np.array(xs), np.array(ys)
    r = stats.pearsonr(xs, ys)
    rs = stats.spearmanr(xs, ys)
    b1, b0 = np.polyfit(xs, ys, 1)
    p("")
    p(f"  pooled over BOTH models, n={len(xs)} subset points:")
    p(f"    Pearson r  = {r.statistic:+.3f} (p={r.pvalue:.2g})")
    p(f"    Spearman   = {rs.statistic:+.3f} (p={rs.pvalue:.2g})")
    p(f"    fit: gain = {b1:+.3f} * v_ab {b0:+.3f}")
    for name in ("gemma2-2b", "gemma3-1b"):
        m = np.array([t == name for t in tags])
        resid = ys[m] - (b1 * xs[m] + b0)
        p(f"    mean residual, {name:<10} = {resid.mean():+.4f}")
    p("")
    p("  If both models scatter about one line, the gain is a function of how")
    p("  much latent x arm there was to remove, and retention is only a proxy.")

    # ---------------- (3) selection on a correlate of the outcome ----------
    p("")
    p("=" * 86)
    p("E20 (3)  MECHANISM FOR THE v_a RISE: selection on a correlate of KL")
    p("=" * 86)
    p("  Per-latent Spearman(activation, kl_per_norm) under the per-arm design.")
    p("  If per-arm selection compresses between-latent variance by measuring")
    p("  every latent near its own activation maximum, v_a should rise more")
    p("  under position control where that correlation is stronger.")
    p("")
    p(f"{'model':<11}{'group':<22}{'lat':>5}{'rho(act,kpn)':>14}"
      f"{'v_a per-arm':>13}{'v_a shared':>12}{'rise':>8}")
    for name, _, _ in MODELS:
        pa, sh, ca, cs, fids = cache[name]
        rho = {}
        for f, g in pa[pa.fid.isin(fids)].groupby("fid"):
            if g.act.nunique() > 2:
                rho[f] = stats.spearmanr(g.act, g.kl_per_norm).statistic
        rr = pd.Series(rho).reindex(fids)
        med = rr.median()
        for lab, m in (("STRONG rho(act,kpn)", (rr > med).to_numpy()),
                       ("WEAK rho(act,kpn)", (rr <= med).to_numpy())):
            idx = np.where(m)[0]
            if len(idx) < 12:
                continue
            _, a, b = gain(ca, cs, idx)
            p(f"{name:<11}{lab:<22}{len(idx):>5}{rr[m].median():>14.3f}"
              f"{a['v_a']:>13.4f}{b['v_a']:>12.4f}"
              f"{b['v_a'] / max(a['v_a'], 1e-9):>7.2f}x")
    p("")
    p("  Prediction: the STRONG group shows the larger v_a rise. If the two")
    p("  groups rise equally, this mechanism is not what is happening and the")
    p("  v_a increase stays unexplained.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
