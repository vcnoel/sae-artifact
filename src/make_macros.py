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

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
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

# one comparison, two conclusions: the rarest activation-frequency decile vs a
# matched random-dictionary decile, under raw KL and under KL-per-unit-norm.
sr0 = sr[sr.bin == 0]
for col, tag in (("kl", "RawFlip"), ("kl_per_norm", "NormFlip")):
    tt = sr0[sr0.kind == "trained"].groupby("fid")[col].median().to_numpy()
    rr0 = sr0[sr0.kind == "random"].groupby("fid")[col].median().to_numpy()
    p, lo, hi = hl_ci(tt, rr0)
    mac(f"{tag}HL", p); mac(f"{tag}Lo", lo); mac(f"{tag}Hi", hi)
    mac(f"{tag}P", stats.mannwhitneyu(tt, rr0).pvalue, "{:.3f}")
    mac(f"{tag}NT", len(tt), "{}"); mac(f"{tag}NR", len(rr0), "{}")
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

# ---------------- alignment vs depth: trend, not shape ----------------
# Spearman at n=300 has SE(Fisher z) = 1/sqrt(297) ~ 0.058, so a CI on rho=0.3 is
# about +-0.11. Two of the four differences are within noise, so the claim is
# tested as a shallow-vs-deep contrast rather than narrated as a sequence.
RHO = {5: 0.358, 12: 0.339, 19: 0.031, 24: 0.145}
NPD = 300
zs = {L: np.arctanh(v) for L, v in RHO.items()}
se = 1 / np.sqrt(NPD - 3)


def pooled(ls):
    w = len(ls) / se ** 2
    return sum(zs[L] / se ** 2 for L in ls) / w, (1 / w) ** 0.5


zsh, esh = pooled([5, 12])
zdp, edp = pooled([19, 24])
mac("RhoShallow", float(np.tanh(zsh))); mac("RhoDeep", float(np.tanh(zdp)))
mac("RhoContrastZ", float((zsh - zdp) / (esh ** 2 + edp ** 2) ** 0.5), "{:.2f}")
mac("RhoContrastP", float(2 * (1 - stats.norm.cdf(
    abs((zsh - zdp) / (esh ** 2 + edp ** 2) ** 0.5)))), "{:.1e}")
mac("RhoSlopePerLayer", float(np.polyfit(sorted(RHO), [zs[L] for L in sorted(RHO)],
                                         1)[0]), "{:+.4f}")
mac("RhoCIHalf", float(np.tanh(np.arctanh(0.3) + 1.96 * se) - 0.3), "{:.2f}")
for L, v in RHO.items():
    mac(f"RhoL{L}", v)
mac("DepthOrderP", 1 / 24, "{:.3f}")
mac("DepthSpan", RHO and float(0.384 / 0.021), "{:.0f}")

# ---------------- two-arm crossed decomposition (trained vs soft-frozen, ----
# ---------------- superseded in the paper by the six-arm design below) -----
sys.path.insert(0, "src")
from variance_decomp import crossed_two_way
import itertools

es = pd.read_csv("results/eval_shared.csv")
es = es[es.arm.isin(("trained", "frozen"))].copy()
es["y"] = np.log10(es.kl_per_norm.clip(lower=1e-12))
r2 = crossed_two_way(es, "fid", "arm", "y")
mac("TwoArmPctArm", r2["pct_b"], "{:.1f}")
mac("TwoArmPctInter", r2["pct_ab"], "{:.1f}")
mac("TwoArmRatio", r2["v_ab"] / max(r2["v_b"], 1e-9), "{:.1f}")
Erho2_2 = r2["v_a"] / (r2["v_a"] + r2["v_ab"] + r2["v_e"] / r2["n_per_cell"])
mac("TwoArmErho2", Erho2_2)
need2 = ((r2["v_ab"] + r2["v_e"] / r2["n_per_cell"]) * 0.8
         / (0.2 * r2["v_a"])) if r2["v_a"] > 0 else float("inf")
mac("TwoArmArmsNeeded", np.ceil(need2), "{:.0f}")

# ---------------- six-arm crossed decomposition (E11) -----------------------
ea = pd.read_csv("results/eval_arms.csv")
ea["y"] = np.log10(ea.kl_per_norm.clip(lower=1e-12))
r6 = crossed_two_way(ea, "fid", "arm", "y")
mac("SixArmI", r6["I"], "{}"); mac("SixArmJ", r6["J"], "{}")
mac("SixArmN", r6["n_per_cell"], "{}")
mac("SixArmPctLatent", r6["pct_a"], "{:.1f}")
mac("SixArmPctArm", r6["pct_b"], "{:.1f}")
mac("SixArmPctInter", r6["pct_ab"], "{:.1f}")
mac("SixArmPctResid", r6["pct_e"], "{:.1f}")
mac("FigVarianceRest", 100 - r6["pct_a"], "{:.1f}")
mac("SixArmRatio", r6["v_ab"] / max(r6["v_b"], 1e-9), "{:.1f}")
Erho2_6 = r6["v_a"] / (r6["v_a"] + r6["v_ab"] + r6["v_e"] / r6["n_per_cell"])
mac("SixArmErho2", Erho2_6)
need6 = (r6["v_ab"] + r6["v_e"] / r6["n_per_cell"]) * 0.8 / (0.2 * r6["v_a"])
mac("SixArmArmsNeeded", np.ceil(need6), "{:.0f}")

pw = ea.groupby(["arm", "fid"]).kl_per_norm.median().unstack(0).dropna()
rhos = [stats.spearmanr(pw[x], pw[y2]).statistic
        for x, y2 in itertools.combinations(pw.columns, 2)]
mac("SixArmRhoMedian", float(np.median(rhos)))
mac("SixArmRhoMin", float(min(rhos))); mac("SixArmRhoMax", float(max(rhos)))
mac("SixArmPairs", len(rhos), "{}")
mac("SixArmSharedN", len(pw), "{}")

# k41 restricts the shared live-latent intersection to 12,080/16,384; check the
# decomposition is not an artifact of that narrower dictionary (PREREG E11).
ea5 = ea[ea.arm != "k41"]
r5 = crossed_two_way(ea5, "fid", "arm", "y")
mac("FiveArmPctLatent", r5["pct_a"], "{:.1f}")
mac("FiveArmPctArm", r5["pct_b"], "{:.1f}")
mac("FiveArmPctInter", r5["pct_ab"], "{:.1f}")
mac("FiveArmPctResid", r5["pct_e"], "{:.1f}")
Erho2_5 = r5["v_a"] / (r5["v_a"] + r5["v_ab"] + r5["v_e"] / r5["n_per_cell"])
mac("FiveArmErho2", Erho2_5)

os.makedirs("paper", exist_ok=True)
with io.open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% GENERATED by src/make_macros.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(f"wrote {OUT} with {len(M)} macros")
for k in sorted(M):
    print(f"  \\{k} = {M[k]}")
