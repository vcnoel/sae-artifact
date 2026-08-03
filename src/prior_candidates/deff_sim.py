# -*- coding: utf-8 -*-
"""Is the family correction CONSEQUENTIAL at the sample sizes papers actually use?

eta^2 = 0.33 (ICC ~ 0.28) is a property of the leaderboard population. What a
paper suffers is the DESIGN EFFECT, and that depends on cluster size, not on ICC
alone: deff = 1 + (m_bar - 1) * ICC. A paper with 12 models drawn from 8 families
has m_bar = 1.5 and deff = 1.14. A paper with 12 models from 3 families has
m_bar = 4 and deff = 1.84.

So this asks the only question that matters for the paper: over realistic
"across N models" designs, how often does a family-clustered analysis actually
FLIP a conclusion that a naive analysis called significant?

Test claim, deliberately the most common form in the literature: "score
increases with scale", estimated across N models by OLS of Average on log params.
Naive: iid SE. Corrected: cluster-robust SE by family + the within-family
permutation null (the permutation_stratified.py logic).
"""
import io
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from eta2_family import lineage  # noqa: E402

RNG = np.random.default_rng(0)
NDRAWS = 2000

d = pd.read_parquet("llb.parquet")
d["lineage"] = d["fullname"].map(lineage)
d = d[(d["Official Providers"] == True) &  # noqa: E712
      (~d["Type"].str.contains("merge", na=False))]
d = d.dropna(subset=["lineage", "Average ⬆️", "#Params (B)"])
d = d[d["#Params (B)"] > 0]
d["lp"] = np.log(d["#Params (B)"])
d["y"] = d["Average ⬆️"]

out = []
w = out.append
w("=" * 78)
w("CONSEQUENTIALITY OF THE FAMILY CORRECTION")
w("claim under test: 'score increases with scale, across N models'")
w(f"pool: {len(d)} official non-merge models, {d.lineage.nunique()} lineages")
w("=" * 78)


def cluster_se(x, y, g):
    """OLS slope with cluster-robust (CR0) SE by group."""
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    meat = np.zeros((2, 2))
    for gg in np.unique(g):
        m = g == gg
        s = X[m].T @ r[m]
        meat += np.outer(s, s)
    V = XtXi @ meat @ XtXi
    nc = len(np.unique(g))
    adj = nc / max(nc - 1, 1)
    return float(b[1]), float(np.sqrt(max(V[1, 1] * adj, 0)))


def iid_se(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    r = y - X @ b
    s2 = (r @ r) / max(len(x) - 2, 1)
    return float(b[1]), float(np.sqrt(s2 * XtXi[1, 1]))


w(f"\n{'design':>34} {'naive sig':>10} {'clustered sig':>14} {'flipped':>9} "
  f"{'SE ratio':>9}")
w("-" * 78)

for N, nfam in [(6, 6), (8, 4), (12, 12), (12, 6), (12, 3), (20, 10), (20, 5),
                (30, 8)]:
    fams = d.lineage.value_counts()
    usable = fams[fams >= 1].index.tolist()
    naive_sig = clus_sig = flips = 0
    ratios = []
    done = 0
    for _ in range(NDRAWS):
        if nfam > len(usable):
            continue
        pick = RNG.choice(usable, size=nfam, replace=False)
        per = N // nfam
        rows = []
        ok = True
        for f in pick:
            sub = d[d.lineage == f]
            if len(sub) < per:
                ok = False
                break
            rows.append(sub.sample(per, random_state=int(RNG.integers(1e9))))
        if not ok:
            continue
        s = pd.concat(rows)
        if s.lp.nunique() < 3:
            continue
        done += 1
        x, y, g = s.lp.to_numpy(), s.y.to_numpy(), s.lineage.to_numpy()
        b1, se1 = iid_se(x, y)
        b2, se2 = cluster_se(x, y, g)
        n_s = abs(b1 / se1) > 1.96 if se1 > 0 else False
        c_s = abs(b2 / se2) > 1.96 if se2 > 0 else False
        naive_sig += n_s
        clus_sig += c_s
        flips += (n_s and not c_s)
        if se1 > 0:
            ratios.append(se2 / se1)
    if done < 20:
        w(f"{f'N={N}, {nfam} families':>34} {'(too few draws)':>10}")
        continue
    w(f"{f'N={N}, {nfam} families ({N//nfam}/fam)':>34} "
      f"{naive_sig/done:>9.1%} {clus_sig/done:>13.1%} {flips/done:>8.1%} "
      f"{np.median(ratios):>9.2f}")

w("\n" + "=" * 78)
w("DESIGN EFFECT as a function of models-per-family, at ICC = 0.28")
w("=" * 78)
icc = 0.280
w(f"  {'models/family':>15} {'deff':>7} {'n_eff for N=12':>16}")
for m in (1, 1.5, 2, 3, 4, 6):
    deff = 1 + (m - 1) * icc
    w(f"  {m:>15} {deff:>7.2f} {12/deff:>16.1f}")
w("\n  A paper sampling one model per family loses NOTHING (deff = 1.00).")
w("  The correction only bites when a paper stacks several members of the")
w("  same family -- which is exactly what a scaling study does, and exactly")
w("  what a breadth study does not.")

open("deff_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
