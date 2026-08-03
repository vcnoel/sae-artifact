# -*- coding: utf-8 -*-
"""Candidate B's decisive test: how much of benchmark-score variance is between
model FAMILIES rather than within them?

Analogue of the multilingual paper's eta^2 = 0.701 for language family against
log Wikipedia size. Source: Open LLM Leaderboard v2 contents (4576 models,
6 benchmarks), pulled from the HF datasets parquet.

Three things the naive version of this test gets wrong, all handled here:

1. SINGLETON INFLATION. With 3291 distinct base models over 4576 rows, a family
   grouping with many size-1 groups drives eta^2 toward 1 by construction: each
   singleton "explains" its own point. Groups of size 1 are dropped and both
   eta^2 and the unbiased omega^2 are reported alongside the group count.

2. POPULATION MISMATCH -- the load-bearing one. The leaderboard is 76% community
   fine-tunes and merges. A paper that says "across 12 models" does not sample
   that population; it samples one or two members per family from official
   releases. eta^2 is therefore computed separately on a paper-like subset, and
   on a strict one-per-family draw, because that is the population whose
   effective sample size the proposed correction is about.

3. SCALE AS THE REAL CLUSTER. Family effects may be nothing but parameter-count
   effects. Partial eta^2 for family after regressing out log #Params says
   whether "family" carries information beyond "size".
"""
import io
import json
import re
import sys

import numpy as np
import pandas as pd

BENCH = ["Average ⬆️", "IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR",
         "MMLU-PRO"]

LINEAGE = [
    (r"llama[-_ ]?3\.[123]|llama[-_ ]?3[._-]?[123]b|llama[-_ ]?3\.", "llama3.x"),
    (r"llama[-_ ]?3", "llama3"), (r"llama[-_ ]?2", "llama2"),
    (r"qwen2\.5|qwen[-_ ]?2\.5", "qwen2.5"), (r"qwen[-_ ]?2", "qwen2"),
    (r"qwen", "qwen1"), (r"mixtral", "mixtral"), (r"mistral|zephyr", "mistral"),
    (r"gemma[-_ ]?2", "gemma2"), (r"gemma", "gemma1"),
    (r"phi[-_ ]?3", "phi3"), (r"phi", "phi2"), (r"yi[-_ ]?1\.5|/yi-", "yi"),
    (r"solar", "solar"), (r"falcon", "falcon"), (r"pythia|neox", "neox"),
    (r"olmo", "olmo"), (r"internlm", "internlm"), (r"deepseek", "deepseek"),
    (r"granite", "granite"), (r"cohere|command", "cohere"),
    (r"stablelm", "stablelm"), (r"tinyllama", "tinyllama"),
]


def lineage(name):
    s = str(name).lower()
    for pat, tag in LINEAGE:
        if re.search(pat, s):
            return tag
    return None


def eta2(df, group, value, min_n=2):
    """One-way eta^2 = SS_between / SS_total, plus unbiased omega^2.

    Groups smaller than min_n are dropped rather than kept, because a size-1
    group contributes its point exactly to the between sum of squares and
    nothing to the within, which is how this statistic gets inflated to 1."""
    d = df[[group, value]].dropna()
    sizes = d.groupby(group)[value].transform("size")
    d = d[sizes >= min_n]
    if d[group].nunique() < 2 or len(d) < 10:
        return None
    gm = d[value].mean()
    grp = d.groupby(group)[value]
    ss_b = float((grp.count() * (grp.mean() - gm) ** 2).sum())
    ss_t = float(((d[value] - gm) ** 2).sum())
    k, n = d[group].nunique(), len(d)
    ss_w = ss_t - ss_b
    ms_w = ss_w / (n - k) if n > k else np.nan
    om = ((ss_b - (k - 1) * ms_w) / (ss_t + ms_w)) if n > k else np.nan
    return dict(eta2=ss_b / ss_t if ss_t > 0 else np.nan, omega2=om,
                n=n, k=k, median_group=float(grp.count().median()))


def partial_eta2_after_size(df, group, value, size_col="#Params (B)"):
    """eta^2 for family on the residuals of value ~ log(params). If family is a
    proxy for scale this collapses."""
    d = df[[group, value, size_col]].dropna()
    d = d[d[size_col] > 0]
    sizes = d.groupby(group)[value].transform("size")
    d = d[sizes >= 2]
    if d[group].nunique() < 2 or len(d) < 10:
        return None
    x = np.log(d[size_col].to_numpy())
    y = d[value].to_numpy()
    b = np.polyfit(x, y, 1)
    d = d.assign(_resid=y - np.polyval(b, x))
    r = eta2(d, group, "_resid")
    return None if r is None else dict(r, r2_size=float(np.corrcoef(x, y)[0, 1] ** 2))


d = pd.read_parquet("llb.parquet")
d["org"] = d["fullname"].astype(str).str.split("/").str[0]
d["lineage"] = d["fullname"].map(lineage)
d["base_org"] = d["Base Model"].astype(str).str.split("/").str[0]

PRETRAINED = ["\U0001f7e2 pretrained", "\U0001f7e9 continuously pretrained"]
POPS = {
    "ALL leaderboard (n=4576)": d,
    "official providers only": d[d["Official Providers"] == True],  # noqa: E712
    "pretrained/base only": d[d["Type"].isin(PRETRAINED)],
    "PAPER-LIKE (official & pretrained/chat, non-merge)":
        d[(d["Official Providers"] == True) &  # noqa: E712
          (~d["Type"].str.contains("merge", na=False))],
}

out = []
w = out.append
w("=" * 78)
w("ETA-SQUARED: MODEL FAMILY vs BENCHMARK SCORE")
w("Open LLM Leaderboard v2, 4576 models. Reference: language family eta^2=0.701")
w("=" * 78)

for pname, pop in POPS.items():
    w(f"\n### {pname}   (n={len(pop)})")
    for gname in ("Architecture", "lineage", "org"):
        r = eta2(pop, gname, "Average ⬆️")
        if r is None:
            w(f"  {gname:>14}: insufficient")
            continue
        w(f"  {gname:>14}: eta2={r['eta2']:.3f}  omega2={r['omega2']:.3f}  "
          f"n={r['n']:>4} k={r['k']:>3} med_grp={r['median_group']:.0f}")

w("\n" + "=" * 78)
w("PER-BENCHMARK, family = lineage, PAPER-LIKE population")
w("=" * 78)
pl = POPS["PAPER-LIKE (official & pretrained/chat, non-merge)"]
for b in BENCH:
    r = eta2(pl, "lineage", b)
    if r:
        w(f"  {b:>14}: eta2={r['eta2']:.3f}  omega2={r['omega2']:.3f}  "
          f"n={r['n']:>4} k={r['k']:>3}")

w("\n" + "=" * 78)
w("DOES FAMILY SURVIVE CONTROLLING FOR SCALE? (residuals of score ~ log params)")
w("=" * 78)
for pname in ("ALL leaderboard (n=4576)",
              "PAPER-LIKE (official & pretrained/chat, non-merge)"):
    pop = POPS[pname]
    for gname in ("Architecture", "lineage"):
        raw = eta2(pop, gname, "Average ⬆️")
        r = partial_eta2_after_size(pop, gname, "Average ⬆️")
        if raw and r:
            w(f"  {pname[:34]:>34} | {gname:>12}: raw eta2={raw['eta2']:.3f} "
              f"-> after size {r['eta2']:.3f}   (size alone R2={r['r2_size']:.3f})")

w("\n" + "=" * 78)
w("THE POPULATION A PAPER ACTUALLY SAMPLES: one model per family")
w("  Papers citing 'N models' usually take 1-3 per family. Drawing one member")
w("  per lineage at random leaves, by construction, ZERO within-family variance")
w("  to pool -- so the relevant question is how much variance a paper FORFEITS")
w("  by treating its N as independent. Reported as the design effect.")
w("=" * 78)
r = eta2(pl, "lineage", "Average ⬆️")
if r:
    # Kish design effect for cluster sampling: 1 + (m_bar - 1) * ICC
    icc = (r["eta2"] * (r["n"] - 1) - (r["k"] - 1)) / (r["n"] - r["k"])
    mbar = r["n"] / r["k"]
    deff = 1 + (mbar - 1) * max(icc, 0)
    w(f"  lineage on PAPER-LIKE: eta2={r['eta2']:.3f}  ICC~={icc:.3f}  "
      f"mean cluster={mbar:.1f}")
    w(f"  design effect = {deff:.2f}  ->  n_eff = {r['n']}/{deff:.2f} = "
      f"{r['n']/deff:.1f} of {r['n']} models")
    w(f"  i.e. {r['n']} leaderboard models carry the information of "
      f"~{r['n']/deff:.0f} independent ones.")

open("eta2_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
