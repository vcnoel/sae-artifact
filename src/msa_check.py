# -*- coding: utf-8 -*-
"""E24: is the 40% msa rise real separation, or a location-scale shift?

THE THREAT. The shared design measures every arm at positions chosen to be
common across arms, which biases toward positions where activation is lower.
If the outcome scales with activation, the whole y distribution shifts and
spreads, and every mean square grows with it. A 40% jump in msa from a
location-scale change is not a finding.

THE CONTROL. Standardise y WITHIN each design to zero mean and unit variance,
then recompute. Standardisation removes any common shift and stretch, so what
survives is separation between latents relative to the spread. If delta msa
survives, latents genuinely separated. If it does not, the outcome distribution
merely stretched.

ALSO (b) msa carries I-1 = 69 df and a single ratio of mean squares has no
interval; bootstrapped over latents here.

ALSO (c) a jackknife. If the rise is carried by a handful of latents it is an
outlier story, not a design effect -- the same check that killed the tail claim
in E2.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np

from boundary_check import comp_cube
from moderator import MODELS, build

B = 1000


def zscore(cube):
    return (cube - cube.mean()) / cube.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/msa_check.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    cache = {name: build(pap, shp) for name, pap, shp in MODELS}

    # ---------- (a) location-scale control ----------
    p("=" * 92)
    p("E24 (a)  DID LATENTS SEPARATE, OR DID THE OUTCOME DISTRIBUTION STRETCH?")
    p("=" * 92)
    p(f"{'model':<11}{'design':<10}{'mean y':>9}{'sd y':>8}"
      f"{'msa raw':>10}{'msa std':>10}")
    for name, _, _ in MODELS:
        _, _, ca, cs, fids = cache[name]
        for lab, c in (("per-arm", ca), ("shared", cs)):
            p(f"{name:<11}{lab:<10}{c.mean():>9.3f}{c.std():>8.3f}"
              f"{comp_cube(c)['msa']:>10.4f}"
              f"{comp_cube(zscore(c))['msa']:>10.4f}")
    p("")
    p(f"{'model':<11}{'d mean':>9}{'sd ratio':>10}{'d msa raw':>12}"
      f"{'d msa std':>12}{'std survives?':>15}")
    for name, _, _ in MODELS:
        _, _, ca, cs, fids = cache[name]
        d_mean = cs.mean() - ca.mean()
        sd_ratio = cs.std() / ca.std()
        d_raw = comp_cube(cs)["msa"] - comp_cube(ca)["msa"]
        d_std = comp_cube(zscore(cs))["msa"] - comp_cube(zscore(ca))["msa"]
        verdict = "YES" if d_std > 0.15 * abs(d_raw) / max(ca.std() ** 2, 1e-9) \
            else ("YES" if d_std > 0 else "NO")
        p(f"{name:<11}{d_mean:>+9.3f}{sd_ratio:>10.3f}{d_raw:>+12.4f}"
          f"{d_std:>+12.4f}{verdict:>15}")
    p("")
    p("  msa on standardised y is a variance ratio: a rise there means latents")
    p("  separated RELATIVE to the spread, which a location-scale shift cannot")
    p("  produce. A rise in raw msa with none in standardised msa is a stretch.")

    # ---------- (b) bootstrap ----------
    p("")
    p("=" * 92)
    p("E24 (b)  BOOTSTRAP INTERVAL ON delta msa")
    p("=" * 92)
    p(f"{'model':<11}{'d msa':>10}{'95% CI':>22}{'rel. change':>13}"
      f"{'CI on rel.':>22}{'frac<=0':>9}")
    rng = np.random.default_rng(0)
    for name, _, _ in MODELS:
        _, _, ca, cs, fids = cache[name]
        I = ca.shape[0]
        d, rel = [], []
        for _ in range(B):
            pick = rng.integers(0, I, I)
            a = comp_cube(ca[pick])["msa"]
            b = comp_cube(cs[pick])["msa"]
            d.append(b - a)
            rel.append((b - a) / a if a > 0 else np.nan)
        d = np.array(d)
        rel = np.array([x for x in rel if np.isfinite(x)])
        base = comp_cube(cs)["msa"] - comp_cube(ca)["msa"]
        base_rel = base / comp_cube(ca)["msa"]
        ci = "[{:+.3f}, {:+.3f}]".format(*np.percentile(d, [2.5, 97.5]))
        cir = "[{:+.0%}, {:+.0%}]".format(*np.percentile(rel, [2.5, 97.5]))
        p(f"{name:<11}{base:>+10.3f}{ci:>22}{base_rel:>12.0%}"
          f"{cir:>22}{float((d <= 0).mean()):>9.3f}")

    # ---------- (c) jackknife ----------
    p("")
    p("=" * 92)
    p("E24 (c)  JACKKNIFE: is the rise broad, or a few latents?")
    p("=" * 92)
    p("  Latents ranked by change in their own cell mean between designs.")
    p("")
    p(f"{'model':<11}{'dropped':<10}{'lat':>5}{'d msa':>10}{'% of full':>11}")
    for name, _, _ in MODELS:
        _, _, ca, cs, fids = cache[name]
        # per-latent change in cell mean (averaged over arms)
        chg = np.abs(cs.mean(axis=(1, 2)) - ca.mean(axis=(1, 2)))
        order = np.argsort(-chg)
        full = comp_cube(cs)["msa"] - comp_cube(ca)["msa"]
        for k in (0, 5, 10):
            keep = order[k:]
            d = comp_cube(cs[keep])["msa"] - comp_cube(ca[keep])["msa"]
            p(f"{name:<11}{('top ' + str(k)) if k else 'none':<10}"
              f"{len(keep):>5}{d:>+10.3f}{100 * d / full:>10.0f}%")
    p("")
    p("  If dropping five latents removes most of delta msa, this is an outlier")
    p("  story and should be reported as one (cf. E2, where one latent of 240")
    p("  moved the variance ratio from 42x to 7x).")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
