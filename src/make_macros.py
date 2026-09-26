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
import re
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


def hl_ci(a, b, seed=None):
    """Bootstrap CI with a FRESH rng seeded from the data.

    A module-level rng makes every CI depend on how many draws happened before
    it, so inserting an unrelated macro upstream silently moves published
    intervals -- which is how PREREG's D3 table came to record [1.159, 2.459]
    for an interval this file now computes as [1.205, 2.459]. Seeding from the
    inputs makes each CI a pure function of its own data.
    """
    if seed is None:
        seed = abs(int(np.sum(np.round(np.concatenate([a, b]), 9) * 1e6))) % (2**31)
    r = np.random.default_rng(seed)
    boot = np.array([hl(r.choice(a, len(a), True), r.choice(b, len(b), True))
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
mac("TwoArmPctLatent", r2["pct_a"], "{:.1f}")
mac("TwoArmPctResid", r2["pct_e"], "{:.1f}")
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

# ------- E12: is position crossed, and is the residual structured? ----------
# Positions are chosen per arm (each arm's own top-activating tokens), so the
# design is nested on position; these quantify how far from crossed it is.
_sets = (ea.groupby(["fid", "arm"])["pos"]
           .apply(lambda s: frozenset(s.tolist())).unstack("arm"))
_j, _agree = [], []
_top1 = (ea.sort_values("act", ascending=False).groupby(["fid", "arm"]).head(1)
           .pivot(index="fid", columns="arm", values="pos"))
for _x, _y in itertools.combinations(list(_sets.columns), 2):
    for _, _rr in _sets[[_x, _y]].iterrows():
        if isinstance(_rr[_x], frozenset) and isinstance(_rr[_y], frozenset):
            _j.append(len(_rr[_x] & _rr[_y]) / len(_rr[_x] | _rr[_y]))
    _d = _top1[[_x, _y]].dropna()
    _agree.append(100.0 * float((_d[_x] == _d[_y]).mean()))
mac("PosJaccard", float(np.mean(_j)), "{:.2f}")
mac("PosTopAgree", float(np.mean(_agree)), "{:.0f}")

_ea = ea.copy()
_ea["resid"] = _ea["y"] - _ea.groupby(["fid", "arm"])["y"].transform("mean")
_ea["log_act"] = np.log10(_ea["act"].clip(lower=1e-12))
_ea["act_rank"] = (_ea.groupby(["fid", "arm"])["act"]
                      .rank(ascending=False, method="first") - 1)
_s = _ea.dropna(subset=["resid", "rel_pos", "log_act", "act_rank"])
_X = np.column_stack([np.ones(len(_s)), _s.rel_pos, _s.log_act, _s.act_rank])
_b = np.linalg.lstsq(_X, _s.resid.values, rcond=None)[0]
_r2 = 1.0 - (((_s.resid.values - _X @ _b) ** 2).sum()
             / (_s.resid.values ** 2).sum())
mac("PosResidRTwo", float(_r2), "{:.3f}")

# ------- E15: the endpoint as a distribution over defensible contrasts ------
_ac = pd.read_csv("results/arm_contrasts.csv")
_nk = _ac[_ac.free != "k41"]
# matched = differing from the comparator on decoder freedom alone
_mt = _ac[_ac.free == "free"].sort_values("frozen")
mac("MatchedTauEightyHL", float(_mt.iloc[0].paired_hl))
mac("MatchedTauEightyLo", float(_mt.iloc[0].paired_lo))
mac("MatchedTauEightyHi", float(_mt.iloc[0].paired_hi))
mac("MatchedTauNinetyHL", float(_mt.iloc[1].paired_hl))
mac("MatchedTauNinetyLo", float(_mt.iloc[1].paired_lo))
mac("MatchedTauNinetyHi", float(_mt.iloc[1].paired_hi))
mac("ContrastAllN", len(_ac), "{}")
mac("ContrastNoKFourOneN", len(_nk), "{}")
mac("ContrastLo", float(_nk.paired_hl.min()))
mac("ContrastHi", float(_nk.paired_hl.max()))
mac("ContrastMedian", float(_nk.paired_hl.median()))
mac("ContrastSigN", int((_nk.paired_lo > 1).sum()), "{}")
mac("ContrastAllLo", float(_ac.paired_hl.min()))
mac("ContrastAllHi", float(_ac.paired_hl.max()))
mac("ContrastAllSigN", int((_ac.paired_lo > 1).sum()), "{}")

# ------- E14: rate vs magnitude readout under the alignment sensitivity -----
# Layer 12 only: the alignment sensitivity is depth-dependent, and pooling over
# layer 19 (where alignment predicts nothing for either readout) would inflate
# the ratio by diluting only the numerator.
_rm = pd.read_csv("results/rate_vs_magnitude.csv")
_rm12 = _rm[_rm.layer == 12]
_zal = _rm12.groupby("dict_tag")["align"].rank(pct=True)
_rk = stats.spearmanr(_zal, _rm12.groupby("dict_tag")["kpn"].rank(pct=True))
_rf = stats.spearmanr(_zal, _rm12.groupby("dict_tag")["flip"].rank(pct=True))
# |dtop|, not dtop: the raw column is a SIGNED probability change, 46% of it
# negative, so a rank correlation on it measures direction not size.
_ra = stats.spearmanr(_zal, _rm12.groupby("dict_tag")["adtop"].rank(pct=True))
_rr = stats.spearmanr(_zal, _rm12.groupby("dict_tag")["kl"].rank(pct=True))
mac("RateRhoKL", float(_rk.statistic))
mac("RateRhoKLRaw", float(_rr.statistic))
mac("RateRhoAdtop", float(_ra.statistic))
mac("RateRhoDtopSigned", float(stats.spearmanr(
    _zal, _rm12.groupby("dict_tag")["dtop"].rank(pct=True)).statistic))
mac("RateDtopNegPct", 100.0 * float((pd.read_csv(
    "results/multi_dict.csv").dtop_t0 < 0).mean()), "{:.0f}")
mac("RateRhoFlip", float(_rf.statistic))
mac("RateRatio", abs(_rk.statistic) / max(abs(_rf.statistic), 1e-9), "{:.1f}")
mac("RateN", len(_rm12), "{}")
mac("RateFlipMin", float(_rm.groupby("dict_tag")["flip"].mean().min()), "{:.3f}")
mac("RateFlipMax", float(_rm.groupby("dict_tag")["flip"].mean().max()), "{:.3f}")

# ------- E16-E24: the two-model, two-corpus position result -----------------
# These are the abstract's numbers. They were hand-entered into main.tex for
# six turns because they live in helper scripts rather than here, which is
# exactly the shape of defect this file exists to prevent: the reproducibility
# statement on page 9 claimed every quantity resolves from a results file while
# page 1 was typed by hand.
#
# Naming: <Model><Corpus><Quantity>, e.g. \GTwoSThreeEightyFourPctAb. Digits
# are spelled out by sanitise(), so "G2" -> "GTwo" is written that way here to
# keep the source and the macro name greppable as the same string.
from boundary_check import comp_cube, to_cube          # noqa: E402
from moderator import build as _build_cubes            # noqa: E402
from position_structure import overlap_stats           # noqa: E402

CONDS = [
    # eval_arms_rerun.csv / eval_arms_shared.csv predate the sampler fix: they
    # drew from the live-set intersection, so their 240 latents share 7 with
    # the 384 run and 7 with the 1536 run, while 384 and 1536 share 232 with
    # each other. Every Gemma-2 statement putting a 96 value beside a larger
    # corpus was therefore a between-sample comparison wearing a corpus label.
    # The *_fixsamp files are the same 96 sequences under the current sampler
    # (231/240 with 384, 227/240 with 1536) and are the correct source. The
    # pre-fix files stay on disk so the difference remains inspectable.
    ("GTwo", "SNinetySix", "results/eval_arms_rerun_fixsamp.csv",
     "results/eval_arms_shared_fixsamp.csv"),
    ("GTwo", "SThreeEightyFour", "results/eval_arms_g2_s384.csv",
     "results/eval_arms_g2_s384_shared.csv"),
    ("GThree", "SNinetySix", "results/eval_arms_g3.csv",
     "results/eval_arms_g3_shared.csv"),
    ("GThree", "SThreeEightyFour", "results/eval_arms_g3_s384.csv",
     "results/eval_arms_g3_s384_shared.csv"),
    ("GTwo", "SFifteenThirtySix", "results/eval_arms_g2_s1536.csv",
     "results/eval_arms_g2_s1536_shared.csv"),
    ("GThree", "SFifteenThirtySix", "results/eval_arms_g3_s1536.csv",
     "results/eval_arms_g3_s1536_shared.csv"),
]
# Second architecture family: six Qwen3.5-2B-Base arms under the same settings as the Gemma rows
# (src/chain6_q35.sh, src/eval_q35.sh). Added only when the evaluation files exist, so the macros for the
# two Gemma bases are produced exactly as before on a checkout that has not run the Qwen arms.
for _c, _pa, _sh in (("SNinetySix", "results/eval_arms_q35.csv", "results/eval_arms_q35_shared.csv"),
                     ("SThreeEightyFour", "results/eval_arms_q35_s384.csv",
                      "results/eval_arms_q35_s384_shared.csv"),
                     ("SFifteenThirtySix", "results/eval_arms_q35_s1536.csv",
                      "results/eval_arms_q35_s1536_shared.csv")):
    if os.path.exists(_pa) and os.path.exists(_sh):
        CONDS.append(("QThreeFive", _c, _pa, _sh))
# The remaining families of the design, each under its own prefix and each
# guarded by existence in the same way: OLMo-2-0425-1B (o2), SmolLM3-3B-Base
# (s3), and the Qwen3.5 scale ladder at 4B and 9B (q354b, q359b). A family
# whose files are absent emits nothing, so every other macro is unchanged.
FAMILIES = [("OlmoTwo", "o2"), ("SmolThree", "s3"),
            ("QThreeFiveFourB", "q354b"), ("QThreeFiveNineB", "q359b")]
for _fam, _ft in FAMILIES:
    for _c, _sfx in (("SNinetySix", ""), ("SThreeEightyFour", "_s384"),
                     ("SFifteenThirtySix", "_s1536")):
        _pa = f"results/eval_arms_{_ft}{_sfx}.csv"
        _sh = f"results/eval_arms_{_ft}{_sfx}_shared.csv"
        if os.path.exists(_pa) and os.path.exists(_sh):
            CONDS.append((_fam, _c, _pa, _sh))
# The 48M-token Qwen3.5-2B arms (model config qwen35-2b-full): the same six
# fitting choices from the same initialisation at four times the 12M budget.
# Their own prefix, QThreeFiveFull, and deliberately absent from _ALL below, so
# no range, count or exception list over the base models moves when they land.
for _c, _sfx in (("SNinetySix", ""), ("SThreeEightyFour", "_s384"),
                 ("SFifteenThirtySix", "_s1536")):
    _pa = f"results/eval_arms_q35full{_sfx}.csv"
    _sh = f"results/eval_arms_q35full{_sfx}_shared.csv"
    if os.path.exists(_pa) and os.path.exists(_sh):
        CONDS.append(("QThreeFiveFull", _c, _pa, _sh))
B_PAIRED = 2000

for _model, _corpus, _pap, _shp in CONDS:
    _tag = _model + _corpus

    # (1) uncontrolled: per-arm design, ALL latents. This is the protocol in
    # common use and the source of the 18.4% / 24.8% in the abstract.
    _pa_all = pd.read_csv(_pap)
    _pa_all["y"] = np.log10(_pa_all.kl_per_norm.clip(lower=1e-12))
    _cu, _ = to_cube(_pa_all)
    _u = comp_cube(_cu)
    mac(f"{_tag}UncPctAb", _u["pct_ab"], "{:.1f}")
    mac(f"{_tag}UncErho", _u["erho2"])
    mac(f"{_tag}UncN", _cu.shape[0], "{}")

    # (2) position selection: do the arms agree where to measure?
    _ov, _ = overlap_stats(_pa_all)
    mac(f"{_tag}Jaccard", float(_ov.mean_jaccard.mean()), "{:.3f}")
    _t1 = (_pa_all.sort_values("act", ascending=False)
           .groupby(["fid", "arm"]).head(1)
           .pivot(index="fid", columns="arm", values="pos"))
    _ag = [100.0 * float((_t1[[x, y2]].dropna()[x]
                          == _t1[[x, y2]].dropna()[y2]).mean())
           for x, y2 in itertools.combinations(list(_t1.columns), 2)]
    mac(f"{_tag}TopAgree", float(np.mean(_ag)), "{:.1f}")

    # (3) like-for-like: the SAME latents under both designs, so the pair
    # differs only in where the measurement was taken.
    _, _, _ca, _cs, _fids = _build_cubes(_pap, _shp)
    _idx = np.arange(_ca.shape[0])
    _a, _b = comp_cube(_ca[_idx]), comp_cube(_cs[_idx])
    mac(f"{_tag}PerArmPctAb", _a["pct_ab"], "{:.1f}")
    mac(f"{_tag}SharedPctAb", _b["pct_ab"], "{:.1f}")
    mac(f"{_tag}PerArmErho", _a["erho2"])
    mac(f"{_tag}SharedErho", _b["erho2"])
    mac(f"{_tag}PairedN", len(_fids), "{}")

    # Position WITHIN latent, and the between latent term it is compared
    # against. Under SHARED positions every arm sits at the same tokens, so the
    # within cell term is position alone. Under PER ARM positions it mixes
    # position with each arm's own selection and cannot be read that way, which
    # is why the manuscript quotes the shared figure. SixArmPctResid, computed
    # from the superseded 96-sequence per-arm file, is the per-arm quantity and
    # is not interchangeable with these.
    for _lbl, _c in (("PerArm", _a), ("Shared", _b)):
        _tot = _c["v_a"] + _c["v_b"] + _c["v_ab"] + _c["v_e"]
        mac(f"{_tag}{_lbl}PctResid", 100 * _c["v_e"] / _tot, "{:.1f}")
        mac(f"{_tag}{_lbl}PctLatent", 100 * _c["v_a"] / _tot, "{:.1f}")

    # (4) paired bootstrap on the gain: ONE latent resample applied to both
    # designs per draw (E18a). Seeded from the data, not from a module-level
    # generator -- see hl_ci's docstring for why that distinction is load-bearing.
    _seed = abs(int(np.sum(np.round(_ca, 9) * 1e6))) % (2 ** 31)
    _r = np.random.default_rng(_seed)
    _I = _ca.shape[0]
    _diffs = []
    _shab = []
    for _ in range(B_PAIRED):
        _pick = _r.integers(0, _I, _I)
        _dr = comp_cube(_ca[_pick])["erho2"]
        _csb = comp_cube(_cs[_pick])
        _ds = _csb["erho2"]
        _shab.append(_csb["pct_ab"])
        if np.isfinite(_dr) and np.isfinite(_ds):
            _diffs.append(_ds - _dr)
    _diffs = np.array(_diffs)
    # Upper end of the same paired bootstrap for the shared-position
    # interaction. Where the point estimate sits at the variance boundary this
    # is what the data bound the component by. Reads the draws above without
    # consuming the generator, so every other macro is unchanged.
    mac(f"{_tag}SharedPctAbHi", float(np.percentile(_shab, 97.5)), "{:.1f}")
    mac(f"{_tag}Gain", float(np.median(_diffs)), "{:+.3f}")
    mac(f"{_tag}GainLo", float(np.percentile(_diffs, 2.5)), "{:+.3f}")
    mac(f"{_tag}GainHi", float(np.percentile(_diffs, 97.5)), "{:+.3f}")
    mac(f"{_tag}GainWidth", float(np.percentile(_diffs, 97.5)
                                  - np.percentile(_diffs, 2.5)))

# ------- position-structure R^2 per family, read from position_structure.py's --
# own text output rather than recomputed, so there is one definition of it.
for _t, _f in (("QThreeFive", "q35"), ("OlmoTwo", "o2"), ("SmolThree", "s3"),
               ("QThreeFiveFourB", "q354b"), ("QThreeFiveNineB", "q359b")):
    _pp = f"results/position_structure_{_f}.txt"
    if not os.path.exists(_pp):
        continue
    _m = re.search(r"rel_pos \+ log_act \+ act_rank\s*=\s*([\d.]+)",
                   io.open(_pp, encoding="utf-8").read())
    if _m:
        mac(f"{_t}PosResidRTwo", float(_m.group(1)), "{:.3f}")

# ------- the 48M-token arms against the 12M arms of the same base -----------
# One rule for all three quantities at 384 sequences: a 48M value matches when
# its printed value lies inside the 95% latent-level bootstrap interval of the
# 12M value. The gain already carries that interval (paired bootstrap above).
# The position share and top agreement get theirs here, from fresh generators
# seeded from their own data, so no existing macro moves. The phrases are
# generated from the verdicts, so the sentence quoting them cannot disagree.
_pp = "results/position_structure_q35full.txt"
if os.path.exists(_pp):
    _m = re.search(r"rel_pos \+ log_act \+ act_rank\s*=\s*([\d.]+)",
                   io.open(_pp, encoding="utf-8").read())
    if _m:
        mac("QThreeFiveFullPosResidRTwo", float(_m.group(1)), "{:.3f}")
if (sanitise("QThreeFiveFullSThreeEightyFourGain") in M
        and sanitise("QThreeFiveSThreeEightyFourGain") in M):
    _p12, _s12 = ("results/eval_arms_q35_s384.csv",
                  "results/eval_arms_q35_s384_shared.csv")
    _, _, _ca12, _cs12, _ = _build_cubes(_p12, _s12)
    _r = np.random.default_rng(abs(int(np.sum(np.round(_cs12, 9) * 1e6)))
                               % (2 ** 31))
    _sh = []
    for _ in range(B_PAIRED):
        _c = comp_cube(_cs12[_r.integers(0, _cs12.shape[0], _cs12.shape[0])])
        _sh.append(100 * _c["v_e"] / (_c["v_a"] + _c["v_b"] + _c["v_ab"]
                                      + _c["v_e"]))
    mac("QThreeFiveFullRefPosShareLo", float(np.percentile(_sh, 2.5)), "{:.1f}")
    mac("QThreeFiveFullRefPosShareHi", float(np.percentile(_sh, 97.5)), "{:.1f}")
    _d12 = pd.read_csv(_p12)
    _t12 = (_d12.sort_values("act", ascending=False).groupby(["fid", "arm"])
            .head(1).pivot(index="fid", columns="arm", values="pos"))
    _pairs = list(itertools.combinations(list(_t12.columns), 2))
    _r = np.random.default_rng(int(_t12.fillna(-1).to_numpy().sum()) % (2 ** 31))
    _ta = []
    for _ in range(B_PAIRED):
        _tb = _t12.iloc[_r.integers(0, len(_t12), len(_t12))]
        _ta.append(np.mean([100.0 * float((_tb[[x, y2]].dropna()[x]
                                           == _tb[[x, y2]].dropna()[y2]).mean())
                            for x, y2 in _pairs]))
    mac("QThreeFiveFullRefTopAgreeLo", float(np.percentile(_ta, 2.5)), "{:.1f}")
    mac("QThreeFiveFullRefTopAgreeHi", float(np.percentile(_ta, 97.5)), "{:.1f}")
    _gv = lambda k: float(M[sanitise(k)])
    _pl = lambda xs: (", ".join(list(xs)[:-1]) + " and " + list(xs)[-1]
                      if len(list(xs)) > 1 else "".join(xs))
    _S = "QThreeFiveSThreeEightyFour"
    _F = "QThreeFiveFullSThreeEightyFour"
    _cmp = [("the position share", _gv(_F + "SharedPctResid"),
             _gv(_S + "SharedPctResid"), _gv("QThreeFiveFullRefPosShareLo"),
             _gv("QThreeFiveFullRefPosShareHi")),
            ("the gain", _gv(_F + "Gain"), _gv(_S + "Gain"),
             _gv(_S + "GainLo"), _gv(_S + "GainHi")),
            ("top agreement", _gv(_F + "TopAgree"), _gv(_S + "TopAgree"),
             _gv("QThreeFiveFullRefTopAgreeLo"),
             _gv("QThreeFiveFullRefTopAgreeHi"))]
    _in = [n for n, v, _, lo, hi in _cmp if lo <= v <= hi]
    _out = [(n, "higher" if v > v12 else "lower")
            for n, v, v12, lo, hi in _cmp if not lo <= v <= hi]
    mac("QThreeFiveFullMatchN", len(_in), "{}")
    _iv = r"the $95\%$ interval of the $12$M value"
    if not _out:
        mac("QThreeFiveFullMatchPhrase", "each inside " + _iv, "{}")
        mac("QThreeFiveFullVerdict",
            "A fourfold budget leaves all three unchanged within bootstrap "
            "error on this base model", "{}")
    else:
        _dir = ", ".join([f"{_out[0][0]} is {_out[0][1]}"]
                         + [f"{n} {d}" for n, d in _out[1:]])
        _dir = (_dir.rsplit(", ", 1)[0] + " and " + _dir.rsplit(", ", 1)[1]
                if len(_out) > 1 else _dir)
        # A shift is in the direction of the finding when the effect the standard corrects grows:
        # a larger position share or gain, or arms agreeing less often on the top position.
        _strengthen = {("the position share", "higher"), ("the gain", "higher"),
                       ("top agreement", "lower")}
        _all_strengthen = all(o in _strengthen for o in _out)
        if not _in:
            mac("QThreeFiveFullMatchPhrase", "each outside " + _iv, "{}")
            _v = ("longer training sharpens the effect rather than removing it" if _all_strengthen
                  else "the result depends on the training budget")
        else:
            mac("QThreeFiveFullMatchPhrase",
                f"with {_pl(_in)} inside {_iv} and "
                f"{_pl([n for n, _ in _out])} outside it", "{}")
            _v = ("longer training sharpens the disagreement the standard corrects rather than "
                  "removing it" if _all_strengthen
                  else "part of the result depends on the training budget")
        mac("QThreeFiveFullVerdict",
            f"At four times the budget {_dir}, so on this base model {_v}",
            "{}")

# ------- summaries across the base models of the design ---------------------
# Ranges and counts over every model present at 384 sequences, derived from the
# macros above so a sentence quoting the range cannot disagree with the table.
_ALL = ["QThreeFiveNineB", "QThreeFiveFourB", "QThreeFive", "SmolThree",
        "OlmoTwo", "GThree", "GTwo"]
_have = [x for x in _ALL if sanitise(x + "SThreeEightyFourGain") in M]
mac("NBaseModels", len(_have), "{}")
_wd = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
       7: "seven", 8: "eight", 9: "nine"}
mac("NBaseModelsWord", _wd.get(len(_have), str(len(_have))), "{}")
# families the base models belong to: the Qwen3.5 ladder is one family
_FAMOF = {"QThreeFiveNineB": "Qwen3.5", "QThreeFiveFourB": "Qwen3.5",
          "QThreeFive": "Qwen3.5", "SmolThree": "SmolLM3",
          "OlmoTwo": "OLMo-2", "GThree": "Gemma", "GTwo": "Gemma"}
_fams = {_FAMOF[x] for x in _have}
mac("NFamilies", len(_fams), "{}")
mac("NFamiliesWord", _wd.get(len(_fams), str(len(_fams))), "{}")
_fv = lambda k: float(M[sanitise(k)])
for _q, _tq in (("SharedPctResid", "PosShare"), ("SharedPctLatent", "LatShare"),
                ("PerArmErho", "PerArmErho"), ("SharedErho", "SharedErho"),
                ("TopAgree", "TopAgree"), ("Gain", "Gain"),
                ("Jaccard", "Jaccard")):
    _v = [_fv(x + "SThreeEightyFour" + _q) for x in _have]
    _fmt = ("{:+.3f}" if _q == "Gain" else
            "{:.3f}" if _q.endswith(("Erho", "Jaccard")) else "{:.1f}")
    mac(f"Across{_tq}Min", min(_v), _fmt)
    mac(f"Across{_tq}Max", max(_v), _fmt)
# every corpus rung of every model: the position share's full range
_vv = [_fv(x + c + "SharedPctResid") for x in _have
       for c in ("SNinetySix", "SThreeEightyFour", "SFifteenThirtySix")
       if sanitise(x + c + "SharedPctResid") in M]
mac("AcrossPosShareAllMin", min(_vv), "{:.1f}")
mac("AcrossPosShareAllMax", max(_vv), "{:.1f}")
mac("AcrossCells", len(_vv), "{}")
# how many (model, corpus) gains have a 95% interval excluding zero
_cells = [(x, c) for x in _have
          for c in ("SNinetySix", "SThreeEightyFour", "SFifteenThirtySix")
          if sanitise(x + c + "GainLo") in M]
mac("AcrossGainCells", len(_cells), "{}")
mac("AcrossGainExclZero",
    sum(1 for x, c in _cells if _fv(x + c + "GainLo") > 0), "{}")
# the largest bootstrap upper bound on the shared interaction, over the cells
# whose point estimate sits at the boundary, and the same at 384 sequences
_zc = [(x, c) for x, c in _cells if _fv(x + c + "SharedPctAb") == 0.0]
if _zc:
    mac("AcrossClippedAbHiMax",
        max(_fv(x + c + "SharedPctAbHi") for x, c in _zc), "{:.1f}")
_pa = [_fv(x + "SThreeEightyFourPerArmPctAb") for x in _have]
mac("AcrossPerArmAbMin", min(_pa), "{:.1f}")
mac("AcrossPerArmAbMax", max(_pa), "{:.1f}")
mac("NTrainedDicts", 6 * len(_have), "{}")
# the shared-position interaction: how many cells sit at the variance boundary
mac("AcrossClippedCells",
    sum(1 for x, c in _cells if _fv(x + c + "SharedPctAb") == 0.0), "{}")

# ------- E25b: composition vs data, read from the analysis script's own CSV --
# Not recomputed here: src/fixed_latent_curve.py owns this estimate (it has to
# intersect three latent sets before it can build a cube) and duplicating the
# logic would create two definitions of the same number.
_fl = pd.read_csv("results/fixed_latent_curve.csv")
_TAG = {"gemma2-2b": "GTwo", "gemma3-1b": "GThree"}
_SZ = {96: "SNinetySix", 384: "SThreeEightyFour", 1536: "SFifteenThirtySix"}
for _mod, _t in _TAG.items():
    _f = _fl[(_fl.model == _mod) & (_fl.latent_set == "fixed")]
    if len(_f):
        mac(f"{_t}FixedN", int(_f.n_lat.iloc[0]), "{}")
        for _, _r in _f.iterrows():
            mac(f"{_t}{_SZ[int(_r.n_seq)]}FixedGain", float(_r.gain), "{:+.3f}")
    _g = _fl[(_fl.model == _mod) & (_fl.latent_set.str.startswith("new at"))]
    _b = _fl[(_fl.model == _mod)
             & (_fl.latent_set.str.startswith("balanced already"))]
    if len(_g) and len(_b):
        mac(f"{_t}NewN", int(_g.n_lat.iloc[0]), "{}")
        mac(f"{_t}NewCons", float(_g.consistency.iloc[0]))
        mac(f"{_t}OldCons", float(_b.consistency.iloc[0]))
        mac(f"{_t}NewPctAb", float(_g.pct_ab_perarm.iloc[0]), "{:.1f}")
        mac(f"{_t}OldPctAb", float(_b.pct_ab_perarm.iloc[0]), "{:.1f}")

# ------- E28: the union rule, read from union_check.py's own CSV ------------
_uc_path = "results/union_check.csv"
if os.path.exists(_uc_path):
    _uc = pd.read_csv(_uc_path)
    for _mod, _t in (("gemma2-2b", "GTwo"), ("gemma3-1b", "GThree")):
        _u = _uc[(_uc.model == _mod) & (_uc.rule == "union")]
        _s = _uc[(_uc.model == _mod) & (_uc.rule == "shared")]
        if not (len(_u) and len(_s)):
            continue
        _u, _s = _u.iloc[0], _s.iloc[0]
        mac(f"{_t}UnionPctAb", float(_u.pct_ab_shared), "{:.1f}")
        mac(f"{_t}UnionErho", float(_u.erho_shared))
        mac(f"{_t}UnionN", int(_u.n_paired), "{}")
        mac(f"{_t}UnionRetained", int(_u.retained), "{}")
        mac(f"{_t}UnionGain", float(_u.gain), "{:+.3f}")
        mac(f"{_t}UnionGainLo", float(_u.lo), "{:+.3f}")
        mac(f"{_t}UnionGainHi", float(_u.hi), "{:+.3f}")
        mac(f"{_t}MinGatedRetained", int(_s.retained), "{}")
    # the families added after the two Gemma bases, same fields, same guard
    for _mod, _t in (("qwen35-2b", "QThreeFive"), ("olmo2-1b", "OlmoTwo"),
                     ("smollm3-3b", "SmolThree"),
                     ("qwen35-4b", "QThreeFiveFourB"),
                     ("qwen35-9b", "QThreeFiveNineB")):
        _u = _uc[(_uc.model == _mod) & (_uc.rule == "union")]
        _s = _uc[(_uc.model == _mod) & (_uc.rule == "shared")]
        if not (len(_u) and len(_s)):
            continue
        _u, _s = _u.iloc[0], _s.iloc[0]
        mac(f"{_t}UnionPctAb", float(_u.pct_ab_shared), "{:.1f}")
        mac(f"{_t}UnionErho", float(_u.erho_shared))
        mac(f"{_t}UnionN", int(_u.n_paired), "{}")
        mac(f"{_t}UnionRetained", int(_u.retained), "{}")
        mac(f"{_t}UnionGain", float(_u.gain), "{:+.3f}")
        mac(f"{_t}UnionGainLo", float(_u.lo), "{:+.3f}")
        mac(f"{_t}UnionGainHi", float(_u.hi), "{:+.3f}")
        mac(f"{_t}MinGatedRetained", int(_s.retained), "{}")

# ------- exceptions named in prose, derived rather than typed ---------------
# Sentences of the form "on every model but X" or "the exception is X" name a
# model that a newly landed rung could join. Each list below is derived from
# the macros above over whichever models are present. A macro whose list is
# empty is NOT emitted, so the manuscript tests it with \ifdefined and prints
# the unqualified claim only while no model contradicts it.
_PROSE = {"GTwo": "Gemma-2-2B", "GThree": "Gemma-3-1B",
          "QThreeFive": "Qwen3.5-2B", "QThreeFiveFourB": "Qwen3.5-4B",
          "QThreeFiveNineB": "Qwen3.5-9B", "OlmoTwo": "OLMo-2",
          "SmolThree": "SmolLM3"}
_ORDER = ["GTwo", "GThree", "QThreeFive", "OlmoTwo", "SmolThree",
          "QThreeFiveFourB", "QThreeFiveNineB"]
_SEQ = {"SNinetySix": "96", "SThreeEightyFour": "384",
        "SFifteenThirtySix": "1536"}


def _prose_list(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _exc(name, models):
    _m = [x for x in _ORDER if x in models]
    if _m:
        mac(name, _prose_list(_PROSE[x] for x in _m), "{}")
        mac(name + "N", len(_m), "{}")


def _has(k):
    return sanitise(k) in M


_R3 = ("SNinetySix", "SThreeEightyFour", "SFifteenThirtySix")
# paired gain at 384 sequences with an interval reaching zero
_exc("ExcGainMain", [x for x in _have
                     if _fv(x + "SThreeEightyFourGainLo") <= 0])
# smallest point gain at 384 over the models whose interval excludes zero
_sig = [x for x in _have if _fv(x + "SThreeEightyFourGainLo") > 0]
if _sig:
    mac("AcrossGainSigMin", min(_fv(x + "SThreeEightyFourGain") for x in _sig),
        "{:+.3f}")
# shared-position interaction at 384 not clipped to the boundary
_exc("ExcSharedOff", [x for x in _have
                      if _fv(x + "SThreeEightyFourSharedPctAb") > 0])
# paired gain not rising from each corpus rung to the next
_exc("ExcGainGrowth", [x for x in _have
                       if all(_has(x + c + "Gain") for c in _R3)
                       and not (_fv(x + "SNinetySixGain")
                                < _fv(x + "SThreeEightyFourGain")
                                < _fv(x + "SFifteenThirtySixGain"))])
# top-position agreement not lower at 1536 than at 96
_exc("ExcAgreeFall", [x for x in _have
                      if all(_has(x + c + "TopAgree") for c in _R3)
                      and not (_fv(x + "SFifteenThirtySixTopAgree")
                               < _fv(x + "SNinetySixTopAgree"))])
# top-position agreement not falling at each rung
_exc("ExcAgreeMono", [x for x in _have
                      if all(_has(x + c + "TopAgree") for c in _R3)
                      and not (_fv(x + "SNinetySixTopAgree")
                               > _fv(x + "SThreeEightyFourTopAgree")
                               > _fv(x + "SFifteenThirtySixTopAgree"))])
# mean Jaccard of the top-position sets not falling at each rung
_exc("ExcJaccardFall", [x for x in _have
                        if all(_has(x + c + "Jaccard") for c in _R3)
                        and not (_fv(x + "SNinetySixJaccard")
                                 > _fv(x + "SThreeEightyFourJaccard")
                                 > _fv(x + "SFifteenThirtySixJaccard"))])
# (model, corpus) cells whose gain interval reaches zero, as prose
_gz = [(x, c) for x in _ORDER for c in _R3
       if (x, c) in _cells and _fv(x + c + "GainLo") <= 0]
if _gz:
    mac("ExcGainCells", _prose_list(f"{_PROSE[x]} at ${_SEQ[c]}$ sequences"
                                    for x, c in _gz), "{}")
    mac("ExcGainCellsN", len(_gz), "{}")
    mac("ExcGainCellsLead",
        "The exception is" if len(_gz) == 1 else "The exceptions are", "{}")
# union rule: gain interval reaching zero, and interaction above one point
_un = [x for x in _ORDER if _has(x + "UnionGainLo")]
_exc("ExcUnionGain", [x for x in _un if _fv(x + "UnionGainLo") <= 0])
_exc("ExcUnion", [x for x in _un if _fv(x + "UnionGainLo") <= 0
                  or _fv(x + "UnionPctAb") > 1.0])
_ug = [x for x in _un if _fv(x + "UnionGainLo") > 0]
if _ug:
    mac("UnionGainExclN", len(_ug), "{}")
    mac("UnionGainExclWord", _wd.get(len(_ug), str(len(_ug))), "{}")
_ub = [x for x in _un if _fv(x + "UnionPctAb") == 0.0]
if _ub:
    mac("UnionBoundaryModels", _prose_list(_PROSE[x] for x in _ub), "{}")

# ------- share of the evaluated sample the minimum-gated rule retains -------
# MinGatedRetained over the latents drawn for the 384-sequence per-arm run, so
# the "retained share" sentences are macros rather than typed percentages.
for _t, _f in (("GTwo", "g2"), ("GThree", "g3"), ("QThreeFive", "q35"),
               ("OlmoTwo", "o2"), ("SmolThree", "s3"),
               ("QThreeFiveFourB", "q354b"), ("QThreeFiveNineB", "q359b")):
    _pp = f"results/eval_arms_{_f}_s384.csv"
    if sanitise(_t + "MinGatedRetained") in M and os.path.exists(_pp):
        _nf = pd.read_csv(_pp, usecols=["fid"]).fid.nunique()
        mac(f"{_t}RetainedPct",
            100.0 * float(M[sanitise(_t + "MinGatedRetained")]) / _nf, "{:.0f}")
        mac(f"{_t}SampledN", _nf, "{}")
_rp = [float(M[k]) for k in M if k.endswith("RetainedPct")]
if _rp:
    mac("AcrossRetainedMin", min(_rp), "{:.0f}")
    mac("AcrossRetainedMax", max(_rp), "{:.0f}")

# ------- figure-backing quantities: displacement null, released-pair matrix --
if os.path.exists("results/displacement.csv"):
    _dp = pd.read_csv("results/displacement.csv")
    for _mod, _t in (("gemma2-2b", "GTwo"), ("gemma3-1b", "GThree")):
        for _k, _kt in (("observed", "Obs"), ("shuffled", "Null")):
            _r = _dp[(_dp.model == _mod) & (_dp.kind == _k)]
            if not len(_r):
                continue
            _r = _r.iloc[0]
            mac(f"{_t}Disp{_kt}Same", float(_r.pct_same), "{:.1f}")
            mac(f"{_t}Disp{_kt}Median", float(_r["median"]), "{:.0f}")
            mac(f"{_t}Disp{_kt}Far", float(_r.pct_far), "{:.0f}")
    mac("DispN", int(_dp[_dp.kind == "observed"].n.iloc[0]), "{}")
    # the Qwen3.5 rungs, same rows and fields as the Gemma pair above
    for _mod, _t in (("qwen35-2b", "QThreeFive"), ("qwen35-4b", "QThreeFiveFourB"),
                     ("qwen35-9b", "QThreeFiveNineB")):
        for _k, _kt in (("observed", "Obs"), ("shuffled", "Null")):
            _r = _dp[(_dp.model == _mod) & (_dp.kind == _k)]
            if not len(_r):
                continue
            _r = _r.iloc[0]
            mac(f"{_t}Disp{_kt}Same", float(_r.pct_same), "{:.1f}")
            mac(f"{_t}Disp{_kt}Median", float(_r["median"]), "{:.0f}")
            mac(f"{_t}Disp{_kt}Far", float(_r.pct_far), "{:.0f}")
        _o = _dp[(_dp.model == _mod) & (_dp.kind == "observed")]
        _s = _dp[(_dp.model == _mod) & (_dp.kind == "shuffled")]
        if len(_o) and len(_s):
            # observed over null same-token share, from the unrounded rows
            mac(f"{_t}DispRatio", float(_o.pct_same.iloc[0])
                / float(_s.pct_same.iloc[0]), "{:.1f}")

# The chance baseline. Without it the result reads the wrong way round: a
# reviewer who thinks about the denominator sees two dictionaries picking from
# ~500 candidate tokens and concludes the measured agreement is remarkably
# HIGH. It is. That is the point -- they agree far more than chance and far
# less than a published comparison assumes, and the error lives in the gap.
if os.path.exists("results/firing_density_2b.csv"):
    _fd = pd.read_csv("results/firing_density_2b.csv")
    _live = _fd[_fd.chance_agree_pct > 0]
    mac("FireMedian", float(_live.fire_median.median()), "{:.0f}")
    mac("FireMin", float(_live.fire_median.min()), "{:.0f}")
    mac("FireMax", float(_live.fire_median.max()), "{:.0f}")
    _ch = float(_live.chance_agree_pct.median())
    mac("ChanceAgree", _ch, "{:.2f}")
    mac("ChanceAgreeMin", float(_live.chance_agree_pct.min()), "{:.2f}")
    mac("ChanceAgreeMax", float(_live.chance_agree_pct.max()), "{:.2f}")
    # the ratios are emitted after the E26 block below, which is where the
    # agreement figures they divide are defined

# How much of the matched population each band actually holds. Without this the
# band figures are a tail described as a centre: 0.50-0.60 is under 4% of
# mutual matches while a third sit above 0.90.
if os.path.exists("results/cosine_dist_2b.csv"):
    _cd = pd.read_csv("results/cosine_dist_2b.csv")
    for _lo, _hi, _t in ((0.50, 0.60, "Low"), (0.90, 1.00, "Top")):
        _c = "band_%.2f_%.2f" % (_lo, _hi)
        if _c in _cd:
            mac(f"ScopeShare{_t}", float(_cd[_c].mean()), "{:.1f}")
    mac("ScopeCosPairMedian", float(_cd.cos_median.median()))
    _ra2 = pd.read_csv("results/released_agreement_2b.csv")
    _bands = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]
    _w = [float(_cd["band_%s" % b.replace("-", "_")].mean()) for b in _bands]
    for _pl, _pt in (("same width", "Sparsity"), ("16k vs", "Width")):
        _sub = _ra2[_ra2.pair.str.startswith(_pl.split()[0])
                    & _ra2.matching.isin(_bands)]
        if len(_sub) == len(_bands):
            _a = [float(_sub[_sub.matching == b].top_agree.iloc[0])
                  for b in _bands]
            mac(f"Rel{_pt}Weighted",
                sum(x * y for x, y in zip(_w, _a)) / sum(_w), "{:.0f}")

if os.path.exists("results/scope_matrix.csv"):
    _sm = pd.read_csv("results/scope_matrix.csv")
    mac("ScopePairs", len(_sm), "{}")
    mac("ScopeAgreeMin", float(_sm.top_agree.min()), "{:.0f}")
    mac("ScopeAgreeMax", float(_sm.top_agree.max()), "{:.0f}")
    mac("ScopeCosMin", float(_sm.median_cos.min()), "{:.2f}")
    mac("ScopeCosMax", float(_sm.median_cos.max()), "{:.2f}")
    mac("ScopeNDicts", len(set(_sm.a) | set(_sm.b)), "{}")

# ------- E26: released production dictionaries ------------------------------
_ra = pd.read_csv("results/released_agreement_2b.csv")
_PAIR = {"same width, L0 82 vs 22": "RelSparsity",
         "16k vs 65k, L0 82 vs 72": "RelWidth"}
_BAND = {">= 0.70": "Near", ">= 0.50": "Loose", "0.50-0.60": "Band",
         "0.90-1.00": "Top"}
for _pl, _pt in _PAIR.items():
    for _bl, _bt in _BAND.items():
        _r = _ra[(_ra.pair == _pl) & (_ra.matching == _bl)]
        if not len(_r):
            continue
        _r = _r.iloc[0]
        mac(f"{_pt}{_bt}Agree", float(_r.top_agree), "{:.1f}")
        mac(f"{_pt}{_bt}Jaccard", float(_r.jaccard))
        mac(f"{_pt}{_bt}MedCos", float(_r.median_cos), "{:.2f}")
        mac(f"{_pt}{_bt}N", int(_r.n_live), "{}")

# NO ChanceRatio macros. A first version divided the released-dictionary
# agreement by a UNIFORM null: one over the number of positions a latent fires
# at, about 0.19%, giving ratios of 52x and 288x. That null is wrong and wrong
# in the flattering direction. Nothing picks uniformly -- both dictionaries take
# the ARGMAX, and activation is concentrated on a few positions, so the
# effective candidate count is far below the firing count. It also divided a
# median chance figure by one band of one pair type, mixing populations. The
# defensible baseline is the shuffle null already computed for
# Figure 1(a) (GTwoDispNullSame), which gives roughly 4x, not several hundred.
# ------- E26q: released Qwen-Scope dictionaries (released_qwen35.py) --------
# A second lab's release. Qwen-Scope ships one width per size, so the only pair
# is k 100 vs 50 at equal width: these macros are the counterpart of the
# RelSparsity* set and there is no RelWidth* counterpart. Guarded by existence,
# so until the run lands this block emits nothing and numbers.tex is unchanged.
# The band shares come from the run itself (share_mutual, the cosine_dist.py
# definition), not from cosine_dist_2b.csv, which holds only Gemma pairs.
_QBAND = {">= 0.70": "Near", ">= 0.50": "Loose", "0.50-0.60": "Band",
          "0.90-1.00": "Top"}
for _qs, _qt in (("2b", "QwenTwoB"), ("9b", "QwenNineB")):
    _qp = f"results/released_qwen35_{_qs}.csv"
    if not os.path.exists(_qp):
        continue
    _q = pd.read_csv(_qp)
    for _bl, _bt in _QBAND.items():
        _r = _q[_q.matching == _bl]
        if not len(_r):
            continue
        _r = _r.iloc[0]
        mac(f"{_qt}{_bt}Agree", float(_r.top_agree), "{:.1f}")
        mac(f"{_qt}{_bt}Jaccard", float(_r.jaccard))
        mac(f"{_qt}{_bt}MedCos", float(_r.median_cos), "{:.2f}")
        mac(f"{_qt}{_bt}N", int(_r.n_live), "{}")
    _r0 = _q.iloc[0]
    mac(f"{_qt}Layer", int(_r0.layer), "{}")
    mac(f"{_qt}Width", int(_r0.width_left), "{}")
    mac(f"{_qt}KLeft", int(_r0.k_left), "{}")
    mac(f"{_qt}KRight", int(_r0.k_right), "{}")
    mac(f"{_qt}FVULeft", float(_r0.fvu_left))
    mac(f"{_qt}FVURight", float(_r0.fvu_right))
    mac(f"{_qt}MutualPct", 100.0 * float(_r0.n_mutual) / float(_r0.width_left),
        "{:.1f}")
    mac(f"{_qt}CosMedian", float(_r0.cos_median_mutual))
    _qb = _q[_q.matching.isin(["0.50-0.60", "0.60-0.70", "0.70-0.80",
                               "0.80-0.90", "0.90-1.00"])]
    for _lo, _t in (("0.50-0.60", "Low"), ("0.90-1.00", "Top")):
        _r = _qb[_qb.matching == _lo]
        if len(_r):
            mac(f"{_qt}Share{_t}", float(_r.share_mutual.iloc[0]), "{:.1f}")
    # same weighting as RelSparsityWeighted: each band's agreement by its share
    # of mutual matches, over the five disjoint bands only
    if len(_qb) == 5:
        mac(f"{_qt}Weighted", float((_qb.share_mutual * _qb.top_agree).sum()
                                    / _qb.share_mutual.sum()), "{:.0f}")

# ------- anchor model of the body --------------------------------------------
# The body's worked examples quote one base model, named once in the setup.
# Every sentence that quotes it reads an Anchor* macro copied from that slug's
# own macros, so swapping the anchor is this one line plus a prose pass.
# The anchor is the largest Qwen3.5 rung whose inputs are all on disk, chosen by
# src/anchor.py, which figures_v2.py reads too, so Figure 1 and the prose
# cannot quote different models.
from anchor import choose as _choose_anchor          # noqa: E402
ANCHOR = _choose_anchor()[0]
mac("AnchorModel", _PROSE[ANCHOR], "{}")
for _q, _src in (("SampledN", "SampledN"),
                 ("UncPctAb", "SThreeEightyFourUncPctAb"),
                 ("DispObsSame", "DispObsSame"),
                 ("DispNullSame", "DispNullSame")):
    if _has(ANCHOR + _src):
        M[sanitise("Anchor" + _q)] = M[sanitise(ANCHOR + _src)]
# the observed-over-null ratio as the words the prose rounds it to, to the
# nearest half: 2.9 reads "three", 3.5 reads "three and a half"
if _has(ANCHOR + "DispRatio"):
    _hr = round(2 * _fv(ANCHOR + "DispRatio")) / 2
    _hw = _wd.get(int(_hr), str(int(_hr)))
    mac("AnchorDispRatioWord",
        _hw + (" and a half" if _hr != int(_hr) else ""), "{}")
# every Anchor* macro the body quotes must exist for the chosen slug
for _q in ("Model", "SampledN", "UncPctAb", "DispObsSame", "DispNullSame",
           "DispRatioWord"):
    if not _has("Anchor" + _q):
        raise SystemExit(f"anchor {ANCHOR}: Anchor{_q} has no source macro")
# the full-sample per-arm interaction at 384 sequences over every model present,
# the range the anchor's own value is quoted against
_uv = [_fv(x + "SThreeEightyFourUncPctAb") for x in _have
       if _has(x + "SThreeEightyFourUncPctAb")]
if _uv:
    mac("AcrossUncPctAbMin", min(_uv), "{:.1f}")
    mac("AcrossUncPctAbMax", max(_uv), "{:.1f}")

# ------- E25c: the Gemma-2 fixed-latent tie at 384 and 1536 -----------------
# Appendix D quotes the components under the two tied fixed-set gains. Read from
# the same fixed_latent_curve.csv rows as E25b, so the tie paragraph cannot
# drift from the gains it explains. New macros only; E25b is unchanged.
_ft = _fl[(_fl.model == "gemma2-2b") & (_fl.latent_set == "fixed")]
_ft = {int(_r.n_seq): _r for _, _r in _ft.iterrows()}
if 384 in _ft and 1536 in _ft:
    for _n in (384, 1536):
        _r = _ft[_n]
        mac(f"GTwo{_SZ[_n]}FixedPerArmPctAb", float(_r.pct_ab_perarm), "{:.1f}")
        mac(f"GTwo{_SZ[_n]}FixedPerArmErho", float(_r.erho_perarm))
        mac(f"GTwo{_SZ[_n]}FixedSharedErho", float(_r.erho_shared))
    _gap = abs(float(_ft[1536].gain) - float(_ft[384].gain))
    mac("GTwoFixedTieGap", _gap, "{:.4f}")
    mac("GTwoFixedTieCIRatio",
        min(float(_ft[_n].hi - _ft[_n].lo) for _n in (384, 1536)) / _gap,
        "{:.0f}")

# The scope condition. Derived from the macros above so the two cannot disagree.
mac("GainWidthRatio",
    float(M[sanitise("GThreeSThreeEightyFourGainWidth")])
    / float(M[sanitise("GTwoSThreeEightyFourGainWidth")]), "{:.1f}")

# ------- final-read additions, v2 manuscript --------------------------------
# Gemma-3-1B's per-arm residual regression at 96 sequences, from the same
# position_structure.py output format as the other families above.
_pp = "results/position_structure_g3.txt"
if os.path.exists(_pp):
    _m = re.search(r"rel_pos \+ log_act \+ act_rank\s*=\s*([\d.]+)",
                   io.open(_pp, encoding="utf-8").read())
    if _m:
        mac("GThreePosResidRTwo", float(_m.group(1)), "{:.3f}")
# Frequency bin against causal mass on the rare-decile eval (sae_rare.csv),
# per latent medians as in the block that feeds PNorm*, Spearman over the six
# activation-frequency bins, for raw KL and for KL per unit perturbation norm.
_fq = sr.groupby(["kind", "fid"]).agg(kpn=("kl_per_norm", "median"),
                                      kl=("kl", "median"),
                                      bin=("bin", "first")).reset_index()
for _k, _kt in (("trained", "Trained"), ("random", "Random")):
    _f = _fq[_fq.kind == _k]
    for _c, _ct in (("kpn", "Norm"), ("kl", "Raw")):
        _sp = stats.spearmanr(_f.bin, _f[_c])
        mac(f"FreqRho{_ct}{_kt}", float(_sp.statistic), "{:+.3f}")
        mac(f"FreqRho{_ct}{_kt}P", float(_sp.pvalue), "{:.3f}")
    mac(f"FreqRho{_kt}N", len(_f), "{}")
# the released-suite dictionary count as a word, for prose
if sanitise("ScopeNDicts") in M:
    _nd = int(M[sanitise("ScopeNDicts")])
    mac("ScopeNDictsWord", _wd.get(_nd, str(_nd)), "{}")
# the displacement comparison count with a thousands separator, for math mode
if sanitise("DispN") in M:
    mac("DispNComma", "{:,}".format(int(M[sanitise("DispN")])).replace(",", "{,}"), "{}")
# the balanced share of the sampled latents at 384 sequences: PairedN (the
# latents in Table 1) over SampledN, the population every Table 1 cell uses
_bp = [100.0 * float(M[sanitise(_t + "SThreeEightyFourPairedN")])
       / float(M[sanitise(_t + "SampledN")])
       for _t in ("GTwo", "GThree", "QThreeFive", "OlmoTwo", "SmolThree",
                  "QThreeFiveFourB", "QThreeFiveNineB")
       if sanitise(_t + "SThreeEightyFourPairedN") in M
       and sanitise(_t + "SampledN") in M]
if _bp:
    mac("AcrossBalancedMin", min(_bp), "{:.0f}")
    mac("AcrossBalancedMax", max(_bp), "{:.0f}")

# ---------------------------------------------------------------------------
# POPULATION TAGS. check_numbers.py verifies that each macro equals what its
# source produces -- provenance. It cannot tell that two macros describe
# DIFFERENT populations, and three errors in one session were exactly that:
# a matrix-wide maximum quoted beside a single-band figure; a median chance
# rate divided by one band of one pair type; a bottom-4% tail reported as the
# centre. Each macro was individually correct. Placing them in one sentence was
# not. Tags are derived from the naming scheme rather than hand-annotated on
# ~200 call sites, so they stay correct as macros are added.
_FAM = r"(GTwo|GThree|QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)"
POP_RULES = [
    (r"^Across", "ours/all-base-models/corpus-384-unless-named/range"),
    (r"^(NBaseModels|NTrainedDicts|NFamilies)", "ours/all-base-models/design-count"),
    (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)"
     r"S(NinetySix|ThreeEightyFour|FifteenThirtySix)Unc",
     "ours/{model}/corpus-{corpus}/full-sample"),
    (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)"
     r"S(NinetySix|ThreeEightyFour|FifteenThirtySix)",
     "ours/{model}/corpus-{corpus}/paired-cube"),
    (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)Union",
     "ours/{model}/corpus-384/union-rule"),
    (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)MinGated",
     "ours/{model}/corpus-384/paired-cube"),
    (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)PosResid",
     "ours/{model}/corpus-96/per-arm-residual-regression"),
    (r"^(GTwo|GThree)S(NinetySix|ThreeEightyFour|FifteenThirtySix)Unc",
     "ours/{model}/corpus-{corpus}/full-sample"),
    (r"^(GTwo|GThree)S(NinetySix|ThreeEightyFour|FifteenThirtySix)Fixed",
     "ours/{model}/corpus-{corpus}/latents-fixed-across-corpora"),
    (r"^(GTwo|GThree)S(NinetySix|ThreeEightyFour|FifteenThirtySix)",
     "ours/{model}/corpus-{corpus}/paired-cube"),
    (r"^(GTwo|GThree)Union", "ours/{model}/corpus-384/union-rule"),
    (r"^(GTwo|GThree)MinGated", "ours/{model}/corpus-384/paired-cube"),
    (r"^(GTwo|GThree|QThreeFiveFourB|QThreeFiveNineB|QThreeFive|OlmoTwo|SmolThree)"
     r"(RetainedPct|SampledN)", "ours/{model}/corpus-384/sampled-latents"),
    (r"^(GTwo|GThree)Disp", "ours/{model}/corpus-384/arm-pair-displacements"),
    (r"^(GTwo|GThree)(New|Old)", "ours/{model}/corpus-1536/composition-groups"),
    (r"^(GTwo|GThree)Fixed", "ours/{model}/latents-fixed-across-corpora"),
    (r"^Rel(Sparsity|Width)Band", "released/pair-{pair}/band-0.50-0.60"),
    (r"^Rel(Sparsity|Width)Top", "released/pair-{pair}/band-0.90-1.00"),
    (r"^Rel(Sparsity|Width)Near", "released/pair-{pair}/cos>=0.70"),
    (r"^Rel(Sparsity|Width)Loose", "released/pair-{pair}/cos>=0.50"),
    (r"^Rel(Sparsity|Width)Weighted",
     "released/pair-{pair}/population-weighted"),
    (r"^Scope(Share|CosPair)", "released/all-pairs/cosine-distribution"),
    (r"^Scope", "released/all-pairs/cos>=0.50"),
    (r"^(Fire|Chance)", "released/single-dictionaries/firing-density"),
    (r"^(SixArm|FiveArm|PosJaccard|PosTopAgree|PosResidRTwo|FigVariance)",
     "ours/gemma2/corpus-96-prefix/original-eval"),
    (r"^TwoArm", "ours/gemma2/corpus-96/two-arm-eval"),
    (r"^(RawFlip|NormFlip|PNorm|SDRatio|TopFiveMass)",
     "ours/gemma2/rare-decile-eval"),
    (r"^(Depth|Rho)", "ours/gemma2/depth-series"),
    (r"^(GSvsRand|MineVs|Mine)", "ours/gemma2/uniform-endpoint-eval"),
    (r"^(Tail|NDicts)", "ours/nine-dictionary-sweep"),
    (r"^Curve", "ours/gemma2/training-budget-curve"),
    (r"^(Rate|Contrast|Matched)", "ours/gemma2/readout-and-contrast-eval"),
    (r"^Qwen(TwoB|NineB)Band", "released/qwen-scope-{model}/band-0.50-0.60"),
    (r"^Qwen(TwoB|NineB)Top(Agree|Jaccard|MedCos|N)$",
     "released/qwen-scope-{model}/band-0.90-1.00"),
    (r"^Qwen(TwoB|NineB)Near", "released/qwen-scope-{model}/cos>=0.70"),
    (r"^Qwen(TwoB|NineB)Loose", "released/qwen-scope-{model}/cos>=0.50"),
    (r"^Qwen(TwoB|NineB)Weighted",
     "released/qwen-scope-{model}/population-weighted"),
    (r"^Qwen(TwoB|NineB)(Share|CosMedian|MutualPct)",
     "released/qwen-scope-{model}/cosine-distribution"),
    (r"^Qwen(TwoB|NineB)", "released/qwen-scope-{model}/dictionary-spec"),
]
_MODEL = {"GTwo": "gemma2-2b", "GThree": "gemma3-1b",
          "QThreeFive": "qwen3.5-2b", "QThreeFiveFourB": "qwen3.5-4b",
          "QThreeFiveNineB": "qwen3.5-9b", "OlmoTwo": "olmo2-1b",
          "SmolThree": "smollm3-3b",
          "TwoB": "qwen3.5-2b", "NineB": "qwen3.5-9b"}
_CORPUS = {"NinetySix": "96", "ThreeEightyFour": "384",
           "FifteenThirtySix": "1536"}
_PAIR = {"Sparsity": "same-width", "Width": "different-width"}
# exception lists name models across the design, so they share the Across tag
POP_RULES.insert(0, (r"^(Exc|UnionGainExcl|UnionBoundary)",
                     "ours/all-base-models/corpus-384-unless-named/exception-list"))
# the anchor copies carry the population of the slug they are copied from
_AM = _MODEL[ANCHOR]
POP_RULES.insert(0, (r"^AnchorModel$", "ours/all-base-models/design-count"))
POP_RULES.insert(0, (r"^AnchorSampledN$", f"ours/{_AM}/corpus-384/sampled-latents"))
POP_RULES.insert(0, (r"^AnchorUncPctAb$", f"ours/{_AM}/corpus-384/full-sample"))
POP_RULES.insert(0, (r"^AnchorDisp", f"ours/{_AM}/corpus-384/arm-pair-displacements"))
POP_RULES.insert(0, (r"^(QThreeFiveFourB|QThreeFiveNineB|QThreeFive)Disp",
                     "ours/{model}/corpus-384/arm-pair-displacements"))
# the 48M-token Qwen3.5-2B arms, a unit of their own beside the 12M arms
_MODEL["QThreeFiveFull"] = "qwen3.5-2b-48M-tokens"
POP_RULES[:0] = [
    (r"^(QThreeFiveFull)S(NinetySix|ThreeEightyFour|FifteenThirtySix)Unc",
     "ours/{model}/corpus-{corpus}/full-sample"),
    (r"^(QThreeFiveFull)S(NinetySix|ThreeEightyFour|FifteenThirtySix)",
     "ours/{model}/corpus-{corpus}/paired-cube"),
    (r"^(QThreeFiveFull)PosResid",
     "ours/{model}/corpus-96/per-arm-residual-regression"),
    (r"^QThreeFiveFullRef",
     "ours/qwen3.5-2b/corpus-384/latent-bootstrap-interval"),
    (r"^QThreeFiveFull(Match|Verdict)",
     "ours/qwen3.5-2b-12M-against-48M-tokens/corpus-384/comparison"),
]

# final-read additions
POP_RULES[:0] = [
    (r"^GThreePosResid", "ours/gemma3-1b/corpus-96/per-arm-residual-regression"),
    (r"^FreqRho", "ours/gemma2/rare-decile-eval"),
    (r"^DispNComma$", "ours/all-base-models/corpus-384/arm-pair-displacements"),
]

def population(name):
    for pat, tmpl in POP_RULES:
        m = re.match(pat, name)
        if not m:
            continue
        # patterns carry zero, one or two groups; index defensively rather than
        # assuming, which is how the first version raised IndexError on ^Scope
        g = m.groups()
        g0 = g[0] if len(g) > 0 else ""
        g1 = g[1] if len(g) > 1 else ""
        return tmpl.format(model=_MODEL.get(g0, g0 or "?"),
                           corpus=_CORPUS.get(g1, "?"),
                           pair=_PAIR.get(g0, g0 or "?"))
    return "unclassified"


os.makedirs("paper", exist_ok=True)
with io.open("paper/populations.tsv", "w", encoding="utf-8") as fh:
    fh.write("# macro\tpopulation  -- GENERATED by make_macros.py\n")
    for k in sorted(M):
        fh.write("%s\t%s\n" % (k, population(k)))
_unc = sum(1 for k in M if population(k) == "unclassified")
print(f"population tags written for {len(M)} macros ({_unc} unclassified)")

with io.open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% GENERATED by src/make_macros.py -- do not edit\n")
    for k, v in sorted(M.items()):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))
print(f"wrote {OUT} with {len(M)} macros")
for k in sorted(M):
    print(f"  \\{k} = {M[k]}")
