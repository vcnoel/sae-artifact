# -*- coding: utf-8 -*-
"""Emit every number the paper prints as a LaTeX macro, from source CSVs.

No number is typed into main.tex. This writes paper/numbers.tex; main.tex uses
the macros; check_numbers.py re-derives them and fails loudly on any mismatch.
Same discipline as the ORACLE paper, where three hand-assembled tables were found
carrying values their sources no longer produced.

Convention, fixed here and used everywhere: t+k/t ratios are FEATURE-LEVEL --
per-feature median first, then ratio of medians. See PREREG E10.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(0)
B = 2000
OUT = "paper/numbers.tex"
M = {}


_D = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
      "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def sanitise(name):
    """LaTeX control sequences may contain letters only. Digits in a
    \\newcommand name produce 'Missing \\begin{document}', which is a
    thoroughly misleading error message."""
    return "".join(_D.get(c, c) for c in name)


def mac(name, val, fmt="{:.3f}"):
    M[sanitise(name)] = fmt.format(val) if isinstance(val, float) else str(val)


def hl(a, b):
    return float(10 ** np.median(np.log10(a)[:, None] - np.log10(b)[None, :]))


def hl_ci(a, b):
    boot = np.array([hl(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                     for _ in range(B)])
    return hl(a, b), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def feat(df, gcol, num, den):
    """feature-level ratio of medians"""
    f = df.groupby(gcol).agg(n=(num, "median"), d=(den, "median"))
    return float(f.n.median() / f.d.median())


# ---------------- endpoints (E6, uniform samples) ----------------
u = pd.read_csv("results/sae_rare_uniform.csv")
g = u.groupby(["kind", "fid"]).agg(kpn=("kl_per_norm", "median")).reset_index()
t = g[g.kind == "trained"].kpn.to_numpy()
r = g[g.kind == "random"].kpn.to_numpy()
p, lo, hi = hl_ci(t, r)
mac("GSvsRandHL", p); mac("GSvsRandLo", lo); mac("GSvsRandHi", hi)
mac("GSvsRandN", len(t), "{}")

e = pd.read_csv("results/eval_saes_uniform.csv")
ge = e.groupby(["arm", "fid"]).agg(kpn=("kl_per_norm", "median")).reset_index()
tt = ge[ge.arm == "trained"].kpn.to_numpy()
for ctrl, tag in (("frozen", "Frozen"), ("random", "Untr")):
    rr = ge[ge.arm == ctrl].kpn.to_numpy()
    p, lo, hi = hl_ci(tt, rr)
    mac(f"MineVs{tag}HL", p); mac(f"MineVs{tag}Lo", lo)
    mac(f"MineVs{tag}Hi", hi)
    mac(f"MineVs{tag}P", stats.mannwhitneyu(tt, rr).pvalue, "{:.4f}")
mac("MineN", len(tt), "{}")

# ---------------- tail generality (nine dictionaries) ----------------
s = pd.read_csv("results/multi_dict.csv")
mm, t5 = [], []
for tag, gg in s.groupby("dict_tag"):
    f = gg.groupby("fid").agg(k=("kpn_t0", "median"))
    v = f.k.to_numpy(); srt = np.sort(v)[::-1]
    mm.append(v.max() / np.median(v))
    t5.append(srt[:max(1, len(srt) // 20)].sum() / srt.sum())
mac("NDicts", s.dict_tag.nunique(), "{}")
mac("TailMaxMedMin", min(mm), "{:.0f}"); mac("TailMaxMedMax", max(mm), "{:.0f}")
mac("TailMassMin", 100 * min(t5), "{:.0f}")
mac("TailMassMax", 100 * max(t5), "{:.0f}")

# ---------------- depth series (E10) ----------------
d4 = pd.read_csv("results/depth4.csv")
for tag, gg in d4.groupby("dict_tag"):
    L = int(gg.layer.iloc[0])
    mac(f"DepthRatioL{L}", feat(gg, "fid", "kl_t3", "kl_t0"))
    f = gg.groupby("fid").agg(k=("kpn_t0", "median")).reset_index()
    mac(f"DepthNL{L}", len(f), "{}")

# ---------------- artifacts ----------------
sr = pd.read_csv("results/sae_rare.csv")
fs = sr.groupby(["kind", "fid"]).agg(kpn=("kl_per_norm", "median"),
                                     pn=("pnorm", "median"),
                                     bin=("bin", "first")).reset_index()
tr = fs[fs.kind == "trained"]
mac("PNormRare", tr[tr.bin == 0].pn.median(), "{:.2f}")
mac("PNormFreq", tr[tr.bin == 5].pn.median(), "{:.2f}")
v = np.sort(tr.kpn.to_numpy())[::-1]
rr = fs[fs.kind == "random"].kpn.to_numpy()
mac("SDRatioAll", v.std() / rr.std(), "{:.1f}")
mac("SDRatioDrop1", v[1:].std() / rr.std(), "{:.1f}")
srt = v; mac("Top5MassGS", 100 * srt[:max(1, len(srt) // 20)].sum() / srt.sum(),
             "{:.1f}")

# ---------------- curve ----------------
import glob, re
cur = []
for f in sorted(glob.glob("results/curve_*.csv"),
                key=lambda x: int(re.search(r"(\d+)M", x).group(1))):
    Mtok = int(re.search(r"(\d+)M", f).group(1))
    dd = pd.read_csv(f)
    gg = dd.groupby(["arm", "fid"]).agg(kpn=("kl_per_norm", "median")).reset_index()
    a = gg[gg.arm == "trained"].kpn.to_numpy()
    b = gg[gg.arm == "frozen"].kpn.to_numpy()
    if len(a) < 10:
        continue
    cur.append((Mtok, hl(a, b)))
    mac(f"CurveHL{Mtok}M", hl(a, b))
if len(cur) >= 3:
    arr = np.array(cur, dtype=float)
    sl = stats.linregress(arr[:, 0], arr[:, 1])
    mac("CurveSlope", sl.slope, "{:+.4f}"); mac("CurveSlopeP", sl.pvalue,
                                                "{:.2f}")

os.makedirs("paper", exist_ok=True)
with io.open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% GENERATED by src/make_macros.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(f"wrote {OUT} with {len(M)} macros")
for k in sorted(M):
    print(f"  \\{k} = {M[k]}")
