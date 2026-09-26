# -*- coding: utf-8 -*-
"""The four body figures of main-v2.tex, all from results/*.csv.

  fig1_selection.pdf   (a) distance between the tokens two arms select for the
                           same latent, against a shuffle null drawn as a
                           percentile band; (b) every latent's effect across the
                           six positions one arm selects for itself.
  fig2_released.pdf    (a) top-1 position agreement against decoder cosine in
                           disjoint bands, with binomial intervals; (b) the same
                           share for each of the fifteen released pairs, ranked.
  fig3_collapse.pdf    (a) paired gain in the generalizability coefficient with
                           its bootstrap interval; (b) the coefficient itself
                           under both position modes, with bootstrap bands.
  fig4_ladder.pdf      (a) top-1 agreement and (b) position set overlap against
                           evaluation corpus size, every arm pair drawn.

LEAD MODEL. The Qwen3.5 rungs lead every figure: figure 1 is drawn on the
anchor model of the body, the rule of src/anchor.py (the largest Qwen3.5
rung whose inputs are all on disk, the same rule the Anchor* macros use)
with every other base model's observed distribution in grey, figure 2 draws the
Qwen-Scope sparsity pairs in colour over Gemma Scope in grey, and figures 3 and
4 draw the Qwen3.5 ladder in a blue ramp over grey context. A rung or release
that is absent drops out and the figure is complete without it.

    python src/figures_v2.py [fig_selection fig_released fig_collapse fig_ladder]

HOUSE STYLE. The document's serif at the document's text width, inserted
unscaled, so a label reaches the page at the size it was set. Panel headings
name the quantity rather than narrating it, series are labelled at their ends
instead of in a legend box, reference values are drawn and named where they are
drawn, and ranges read "to" rather than as a dash.

UNCERTAINTY. Every point estimate carries the spread it was computed from: the
null is a band over repeated shuffles rather than one draw, agreement over a
matched population carries a binomial interval, the variance components and the
coefficient carry a latent level bootstrap, and the corpus ladder draws all
fifteen arm pairs behind each mean rather than drawing the mean alone.
"""
import io
import itertools
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

sys.path.insert(0, "src")
from boundary_check import comp_cube
from moderator import build as build_cubes
from position_structure import overlap_stats

# Text block width of the ICLR 2027 style, in inches. Figures are built at it.
TW = 5.5

# Palette: ink, two greys, and three hues, colourblind safe.
INK, MID, FAINT = "#1A1A1A", "#7A7A7A", "#DBDBDB"
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{times}"
                           r"\usepackage{amsmath}",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "figure.dpi": 200,
    "pdf.fonttype": 42, "savefig.pad_inches": 0.02,
})

PER_ARM = "results/eval_arms_g2_s384.csv"
REF = "free"
B_BOOT = 600
NAMES = {"gemma2-2b": "Gemma-2-2B", "gemma3-1b": "Gemma-3-1B",
         "qwen35-2b": "Qwen3.5-2B", "olmo2-1b": "OLMo-2-1B",
         "smollm3-3b": "SmolLM3-3B", "qwen35-4b": "Qwen3.5-4B",
         "qwen35-9b": "Qwen3.5-9B"}
# Newest first, as in Table 1. A model enters a figure only when all three of
# its corpus rungs exist, so the Qwen3.5 ladder appears as soon as it lands.
ORDER = ["qwen35-9b", "qwen35-4b", "qwen35-2b", "smollm3-3b", "olmo2-1b",
         "gemma3-1b", "gemma2-2b"]

# Every (model, corpus) cell, with the per arm and shared evaluation behind it.
# The 96-sequence Gemma-2 pair is the *_fixsamp files: the pre-fix ones drew a
# nearly disjoint latent sample and are not comparable with the larger corpora.
CONDS = [
    ("gemma2-2b", 96, "results/eval_arms_rerun_fixsamp.csv",
     "results/eval_arms_shared_fixsamp.csv"),
    ("gemma2-2b", 384, "results/eval_arms_g2_s384.csv",
     "results/eval_arms_g2_s384_shared.csv"),
    ("gemma2-2b", 1536, "results/eval_arms_g2_s1536.csv",
     "results/eval_arms_g2_s1536_shared.csv"),
    ("gemma3-1b", 96, "results/eval_arms_g3.csv",
     "results/eval_arms_g3_shared.csv"),
    ("gemma3-1b", 384, "results/eval_arms_g3_s384.csv",
     "results/eval_arms_g3_s384_shared.csv"),
    ("gemma3-1b", 1536, "results/eval_arms_g3_s1536.csv",
     "results/eval_arms_g3_s1536_shared.csv"),
]
import os
for _m, _t in (("qwen35-2b", "q35"), ("olmo2-1b", "o2"), ("smollm3-3b", "s3"),
               ("qwen35-4b", "q354b"), ("qwen35-9b", "q359b")):
    _c = [(_m, n, f"results/eval_arms_{_t}{sfx}.csv",
           f"results/eval_arms_{_t}{sfx}_shared.csv")
          for n, sfx in ((96, ""), (384, "_s384"), (1536, "_s1536"))]
    if all(os.path.exists(a) and os.path.exists(b) for _, _, a, b in _c):
        CONDS.extend(_c)
MODELS = [m for m in ORDER if any(c[0] == m for c in CONDS)]


def end_labels(ax, ys, names, x, colour=INK, gap=None, fs=6.8):
    """Label each series at its right end, nudged apart so no two touch."""
    lo, hi = ax.get_ylim()
    gap = gap if gap is not None else 0.095 * (hi - lo)
    order = np.argsort(ys)
    placed = []
    for i in order:
        y = ys[i]
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
    # push the stack down if it ran past the top
    over = placed[-1] - (hi - 0.02 * (hi - lo)) if placed else 0
    if over > 0:
        placed = [p - over for p in placed]
    for i, y in zip(order, placed):
        c = colour[i] if isinstance(colour, (list, tuple)) else colour
        ax.annotate(names[i], xy=(x, ys[i]), xytext=(x, y),
                    textcoords="data", fontsize=fs, color=c, ha="left",
                    va="center", annotation_clip=False)


def save(fig, name):
    """Write the figure and report anything that reaches past the canvas."""
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    over = [(-tb.x0) * 72, (-tb.y0) * 72,
            (tb.x1 - fig.get_figwidth()) * 72,
            (tb.y1 - fig.get_figheight()) * 72]
    os.makedirs("paper", exist_ok=True)
    fig.savefig(f"paper/{name}")
    flag = ""
    if max(over) > 0.5:
        sides = dict(zip(("left", "bottom", "right", "top"),
                         [round(v, 2) for v in over]))
        flag = f"  CLIPPED {sides}"
    print(f"wrote {name}  {fig.get_figwidth() * 72:.1f} pt wide"
          f" x {fig.get_figheight() * 72:.1f} pt{flag}")


def wilson(k, n, z=1.96):
    """Wilson interval for a binomial proportion, in percent.

    The normal approximation misbehaves near the ends, and the lowest cosine
    band sits around one in ten, so the interval that is drawn has to be one
    that stays inside zero and one there.
    """
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * min(c + h, 1.0)


def top_positions(d):
    """Each arm's single top activating position, per latent."""
    return (d.sort_values("act", ascending=False).groupby(["fid", "arm"]).head(1)
            .pivot(index="fid", columns="arm", values="pos"))


def pair_agreement(d):
    """Top-1 agreement for every arm pair, as a percentage. Fifteen values."""
    top = top_positions(d)
    out = []
    for x, y in itertools.combinations(list(top.columns), 2):
        p = top[[x, y]].dropna()
        out.append(100.0 * float((p[x] == p[y]).mean()))
    return np.array(out)


def displacements(d, n_null=400):
    """Selection distance, observed and under a constrained shuffle null.

    The null shuffles, for each latent, which of ITS OWN firing positions each
    arm picks. Shuffling over all 511 tokens would be a straw man, since an arm
    can only choose where the latent fires. It is repeated so the null can be
    drawn as a band rather than as one draw that happens to be the one we took.
    """
    top = top_positions(d)
    arms = list(top.columns)
    obs = []
    for x, y in itertools.combinations(arms, 2):
        p = top[[x, y]].dropna()
        obs.append((p[x] - p[y]).abs().to_numpy())
    obs = np.concatenate(obs)

    pool = [np.unique(g.to_numpy()) for _, g in d.groupby("fid")["pos"]]
    pool = [c for c in pool if len(c) >= 2]
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(n_null):
        draw = []
        for cand in pool:
            a, b = rng.choice(cand, 2, replace=True)
            draw.append(abs(int(a) - int(b)))
        reps.append(np.array(draw))
    return obs, reps


def boot_components(pap, shp, b=B_BOOT):
    """Latent level bootstrap of both designs on one resample per draw.

    Resampling latents rather than observations is what matches the estimand:
    the question is whether this ranking of latents survives, so the latent is
    the sampling unit. Both designs take the SAME resample in each draw, so the
    gain is paired and its interval is not the difference of two marginals.
    """
    _, _, ca, cs, fids = build_cubes(pap, shp)
    n = ca.shape[0]
    # Seeded from the data rather than from a module level generator: a shared
    # generator let an unrelated quantity upstream move these intervals.
    seed = abs(int(np.sum(np.round(ca, 9) * 1e6))) % (2 ** 31)
    rng = np.random.default_rng(seed)
    keys = ("pa_ab", "sh_ab", "pa_er", "sh_er", "gain")
    draws = {k: [] for k in keys}
    for _ in range(b):
        idx = rng.integers(0, n, n)
        A, S = comp_cube(ca[idx]), comp_cube(cs[idx])
        draws["pa_ab"].append(A["pct_ab"])
        draws["sh_ab"].append(S["pct_ab"])
        draws["pa_er"].append(A["erho2"])
        draws["sh_er"].append(S["erho2"])
        draws["gain"].append(S["erho2"] - A["erho2"])
    A, S = comp_cube(ca), comp_cube(cs)
    out = dict(n=n, pa_ab=A["pct_ab"], sh_ab=S["pct_ab"],
               pa_er=A["erho2"], sh_er=S["erho2"],
               gain=S["erho2"] - A["erho2"])
    for k in keys:
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        out[k + "_lo"], out[k + "_hi"] = lo, hi
    return out


# ------------------------------------------------------- series styling --
# The newest family leads every figure. The Qwen3.5 rungs are drawn in a blue
# ramp, darkest for the largest rung present, and every other base model is
# grey context behind them, labelled in grey. A rung that lands later (9B)
# enters the ramp at the dark end without any change here.
QWEN = [m for m in MODELS if m.startswith("qwen35")]
# Figure 1 is drawn on the anchor model of the body, chosen by the one rule
# in src/anchor.py that make_macros.py also reads, so the figure and the
# prose quoting \AnchorModel cannot disagree.
from anchor import choose as choose_anchor
LEAD = choose_anchor()[1]
assert LEAD in MODELS, (LEAD, MODELS)
RAMP = ["#08457A", "#0072B2", "#5A9FD0"]
CTX = "#ABABAB"


def style(model):
    """Colour, line width, label colour and z-order for one base model."""
    if model in QWEN:
        c = RAMP[QWEN.index(model)]
        return dict(c=c, lw=1.15 if model == LEAD else 0.95, lab=c, z=5)
    return dict(c=CTX, lw=0.7, lab=MID, z=3)


def lead_per_arm():
    """The per-arm file at 384 sequences of the lead model."""
    return next(p for m, n, p, _ in CONDS if m == LEAD and n == 384)


def spine_to(ax, lo, hi):
    """End the x spine at the data range, leaving room for end labels."""
    ax.spines["bottom"].set_bounds(lo, hi)


# ---------------------------------------------------------------- figure 1 --
def fig_selection():
    fig = plt.figure(figsize=(TW, 2.02))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.30,
                           left=0.082, right=0.985, top=0.855, bottom=0.205)

    ax = fig.add_subplot(gs[0])
    lead = pd.read_csv(lead_per_arm())
    obs, reps = displacements(lead)
    # The caption quotes the comparison count from the displacement table,
    # which is the same fifteen pairs over the same sampled latents on every
    # model. Refuse to draw a panel whose count would disagree with it.
    if os.path.exists("results/displacement.csv"):
        _dp = pd.read_csv("results/displacement.csv")
        n_ref = int(_dp[_dp.kind == "observed"].n.iloc[0])
        assert len(obs) == n_ref, (len(obs), n_ref)
    bins = np.linspace(0, 511, 34)
    # Every other base model's observed distribution as grey context, so the
    # lead model is read against the whole set rather than alone.
    for m in MODELS:
        if m == LEAD:
            continue
        o, _ = displacements(pd.read_csv(
            next(p for mm, n, p, _ in CONDS if mm == m and n == 384)),
            n_null=0)
        h, _ = np.histogram(o, bins=bins, density=True)
        ax.step(bins[:-1], h, where="post", color=CTX, lw=0.55, alpha=0.8,
                zorder=2)
    # The null as a second to ninety eighth percentile band over repeated
    # shuffles with its mean, and the observed distribution drawn on top of it,
    # so the reader sees the null's own spread rather than one realisation.
    H = np.array([np.histogram(r, bins=bins, density=True)[0] for r in reps])
    lo, hi = np.percentile(H, [2, 98], axis=0)
    mid = H.mean(axis=0)
    ho, _ = np.histogram(obs, bins=bins, density=True)
    ax.fill_between(bins[:-1], lo, hi, step="post", color=MID, alpha=0.22,
                    linewidth=0, zorder=2)
    ax.step(bins[:-1], mid, where="post", color=MID, lw=0.8,
            ls=(0, (3, 2)), zorder=3)
    lc = style(LEAD)["c"]
    ax.step(bins[:-1], ho, where="post", color=lc, lw=1.15, zorder=4)
    med = float(np.median(obs))
    ax.axvline(med, color=lc, lw=0.8, ls=(0, (1, 1.2)), zorder=5)
    top = ax.get_ylim()[1]
    ax.annotate(f"median {med:.0f}", xy=(med, top * 0.80), xytext=(4, 0),
                textcoords="offset points", fontsize=7, color=lc,
                va="center")
    ax.annotate(f"null, {len(reps)} shuffles", xy=(400, float(hi[25])),
                xytext=(0, 9), textcoords="offset points", fontsize=7,
                color=MID, ha="center")
    ax.annotate(NAMES[LEAD], xy=(232, float(ho[15])), xytext=(232, top * 0.52),
                fontsize=7, color=lc, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=MID,
                                shrinkA=2, shrinkB=1))
    ax.set_xlim(0, 511)
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.set_yticks([])
    ax.set_xlabel("distance between the two selected tokens")
    ax.set_ylabel("density")
    ax.set_title(f"(a) selection distance, {len(obs):,} latent comparisons", loc="left",
                 pad=3)

    ax = fig.add_subplot(gs[1])
    g = (lead.query("arm == @REF").groupby("fid").kl_per_norm
         .agg(lo="min", hi="max", med="median"))
    g = g[g.lo > 0].sort_values("med")
    s = g.iloc[::max(1, len(g) // 80)].reset_index(drop=True)
    yy = np.arange(len(s))
    ax.hlines(yy, s.lo, s.hi, color=FAINT, lw=1.1, zorder=2)
    ax.plot(s.med, yy, "o", ms=1.7, color=lc, zorder=3)
    ax.set_xscale("log")
    # limits from the data and one tick every second decade inside them, so
    # the last tick label never reaches past the panel
    x0, x1 = float(s.lo.min()) / 2, float(s.hi.max()) * 2
    ax.set_xlim(x0, x1)
    e0, e1 = int(np.ceil(np.log10(x0))), int(np.floor(np.log10(x1)))
    ax.set_xticks([10.0 ** e for e in range(e1 - 1, e0 - 1, -2)])
    ax.minorticks_off()
    ax.set_yticks([])
    ax.set_ylim(-2, len(s) + 2)
    ratio = float(np.median(g.hi / g.lo))
    ax.annotate(f"median latent spans {ratio:.0f} times",
                xy=(s.lo.min() * 1.4, len(s) * 0.86), fontsize=7, color=MID,
                ha="left", va="center")
    ax.set_xlabel("effect per unit perturbation norm")
    ax.set_ylabel("latents, ranked by median")
    ax.set_title("(b) effect across one arm's own positions", loc="left",
                 pad=3)
    save(fig, "fig1_selection.pdf")


# ---------------------------------------------------------------- figure 2 --
BANDS = ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]


def qwen_released():
    """Qwen-Scope sparsity pairs present on disk, largest model first."""
    out = []
    for size, name in (("9b", "Qwen3.5-9B"), ("2b", "Qwen3.5-2B")):
        p = f"results/released_qwen35_{size}.csv"
        if os.path.exists(p):
            q = pd.read_csv(p)
            q = q[q.matching.isin(BANDS)].sort_values("median_cos")
            if len(q) == 5:
                out.append((name, q))
    return out


def fig_released():
    d = pd.read_csv("results/released_agreement_2b.csv")
    bands = d[d.matching.str.contains("-")].copy()
    pairs = sorted(bands.pair.unique())
    gemma = list(zip(pairs, ["Gemma Scope, width", "Gemma Scope, sparsity"]))
    qwen = qwen_released()
    qcol = dict(zip([n for n, _ in qwen], RAMP[:len(qwen)]))

    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.0),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig.subplots_adjust(left=0.082, right=0.985, top=0.855, bottom=0.205,
                        wspace=0.30)

    ax = axes[0]
    ends = []
    # Gemma Scope's two pairs as grey context, no interval drawn. Without the
    # Qwen-Scope files they are the only series and are drawn in colour with
    # their intervals, as before the second release was added.
    for (pair, lab), hue in zip(gemma, [BLUE, ORANGE]):
        g = bands[bands.pair == pair].sort_values("median_cos")
        if not qwen:
            ci = np.array([wilson(round(r.top_agree / 100 * r.n_live),
                                  r.n_live) for r in g.itertuples()])
            ax.fill_between(g.median_cos, ci[:, 0], ci[:, 1], color=hue,
                            alpha=0.16, linewidth=0, zorder=2)
        ax.plot(g.median_cos, g.top_agree, marker="o",
                ms=2.0 if qwen else 2.6, lw=0.7 if qwen else 0.9,
                color=CTX if qwen else hue, zorder=2)
        ends.append((g.top_agree.iloc[-1], lab, MID if qwen else hue))
    # Qwen-Scope's sparsity pair on each model, with its binomial interval.
    for name, q in qwen:
        c = qcol[name]
        ci = np.array([wilson(round(r.top_agree / 100 * r.n_live), r.n_live)
                       for r in q.itertuples()])
        ax.fill_between(q.median_cos, ci[:, 0], ci[:, 1], color=c,
                        alpha=0.14, linewidth=0, zorder=3)
        ax.plot(q.median_cos, q.top_agree, marker="o", ms=2.6, lw=1.0,
                color=c, zorder=4)
        ends.append((q.top_agree.iloc[-1], name, c))
    ax.set_xlim(0.5, 1.2)
    ax.set_xticks([0.6, 0.8, 1.0])
    spine_to(ax, 0.5, 1.0)
    ax.set_ylim(4, 76)
    end_labels(ax, [e[0] for e in ends], [e[1] for e in ends], 0.955,
               colour=[e[2] for e in ends], gap=6.4, fs=6.6)
    ax.set_xlabel("median decoder cosine")
    ax.set_ylabel(r"same token (\%)")
    ax.set_title("(a) agreement by decoder similarity", loc="left", pad=3)

    # Every released pair as its own row: the fifteen Gemma Scope pairs in
    # grey, and the Qwen-Scope pair of each model, weighted over the five
    # bands by each band's share of mutual matches, in its own colour.
    ax = axes[1]
    s = pd.read_csv("results/scope_matrix.csv")
    rows = [(float(v), None) for v in s.top_agree]
    for name, q in qwen:
        w = float((q.share_mutual * q.top_agree).sum() / q.share_mutual.sum())
        rows.append((w, name))
    rows.sort(key=lambda r: r[0])
    y = np.arange(len(rows))
    mean = float(s.top_agree.mean())
    for yi, (v, name) in zip(y, rows):
        c = qcol[name] if name else MID
        ax.hlines(yi, 0, v, color=c if name else FAINT,
                  lw=1.6, alpha=0.45 if name else 1.0, zorder=2)
        ax.plot(v, yi, "o", ms=3.2 if name else 2.6, color=c, zorder=3)
        if name:
            # named right of the mean line so the label never sits on it
            ax.annotate(name, xy=(max(v, mean) + 1.6, yi), fontsize=6.8,
                        color=c, va="center")
    ax.axvline(mean, color=MID, lw=0.8, ls=(0, (3, 2)), zorder=1)
    # named in the empty band right of the line, above the lowest rows
    ax.annotate(f"Gemma Scope\nmean {mean:.0f}", xy=(mean, 3.2),
                xytext=(4, 0), textcoords="offset points", fontsize=6.8,
                color=MID, va="center", linespacing=1.35)
    ax.set_xlim(0, 60)
    ax.set_xticks([0, 20, 40, 60])
    ax.set_ylim(-1.2, len(rows) + 0.4)
    ax.set_yticks([])
    ax.set_xlabel(r"same token (\%)")
    ax.set_ylabel(f"{len(rows)} released pairs")
    ax.set_title("(b) agreement, every released pair", loc="left", pad=3)
    save(fig, "fig2_released.pdf")


# ---------------------------------------------------------------- figure 3 --
def fig_collapse():
    rows = {}
    for model, n_seq, pap, shp in CONDS:
        rows[(model, n_seq)] = boot_components(pap, shp)

    models = MODELS
    corpora = [96, 384, 1536]
    # panel (b) widens with the number of models so its two-line tick labels
    # keep clear of each other as the Qwen3.5 ladder grows
    _wb = 1.0 + 0.1 * max(len(models) - 5, 0)
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.12),
                             gridspec_kw={"width_ratios": [2.0 - _wb, _wb]})
    fig.subplots_adjust(left=0.085, right=0.975, top=0.87, bottom=0.235,
                        wspace=0.52)

    # (a) the gain, which is the quantity with a paired interval. The Qwen3.5
    # ladder in blue, every other model in grey behind it.
    ax = axes[0]
    ax.axhline(0, color=MID, lw=0.6, zorder=1)
    offs = np.linspace(0.92, 1.08, len(models))
    ends = []
    for model, off in sorted(zip(models, offs),
                             key=lambda t: style(t[0])["z"]):
        st = style(model)
        xs = [c * off for c in corpora]
        v = [rows[(model, c)] for c in corpora]
        ax.errorbar(xs, [r["gain"] for r in v],
                    yerr=[[r["gain"] - r["gain_lo"] for r in v],
                          [r["gain_hi"] - r["gain"] for r in v]],
                    fmt="-o", ms=2.6 if model in QWEN else 2.0, lw=st["lw"],
                    elinewidth=0.6, capsize=1.4, color=st["c"],
                    zorder=st["z"])
        ends.append((v[-1]["gain"], NAMES[model], st["lab"]))
    ax.set_xscale("log", base=2)
    ax.set_xticks(corpora)
    ax.set_xticklabels([str(c) for c in corpora])
    ax.minorticks_off()
    ax.set_xlim(78, 1800)
    ax.set_ylim(-0.1, 0.56)
    ax.annotate("no gain", xy=(620, 0), xytext=(0, -2.5),
                textcoords="offset points", fontsize=6.8, color=MID,
                ha="center", va="top")
    end_labels(ax, [e[0] for e in ends], [e[1] for e in ends], 1900,
               colour=[e[2] for e in ends])
    ax.set_xlabel("evaluation corpus (sequences)")
    ax.set_ylabel(r"gain in $E\rho^2$")
    ax.set_title("(a) gain from holding position fixed", loc="left", pad=3)

    # (b) the component Table 1 reports at 384 sequences, per-arm against
    # shared on the same latents, with the same bootstrap. The Qwen3.5 ladder
    # sits first on a pale band and the other models follow at lower contrast.
    ax = axes[1]
    xs = np.arange(len(models))
    nq = len([m for m in models if m in QWEN])
    if nq:
        ax.axvspan(-0.5, nq - 0.5, color=FAINT, alpha=0.35, linewidth=0,
                   zorder=0)
    for key, colour, dx in (("pa_ab", ORANGE, -0.13), ("sh_ab", BLUE, 0.13)):
        for i, m in enumerate(models):
            r = rows[(m, 384)]
            ax.errorbar([i + dx], [r[key]],
                        yerr=[[max(r[key] - r[key + "_lo"], 0)],
                              [max(r[key + "_hi"] - r[key], 0)]],
                        fmt="o", ms=3.0 if m in QWEN else 2.4, lw=0.8,
                        capsize=1.6, color=colour,
                        alpha=1.0 if m in QWEN else 0.45, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([NAMES[m].replace("-", "\n", 1) for m in models],
                       fontsize={5: 6.6, 6: 6.3}.get(len(models), 6.0),
                       linespacing=1.4)
    for t, m in zip(ax.get_xticklabels(), models):
        t.set_color(INK if m in QWEN else MID)
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(-1.0, 26)
    ax.set_ylabel(r"latent $\times$ arm (\%)", labelpad=6)
    # series named beside their own last point, in their own colour
    last = rows[(models[-1], 384)]
    ax.annotate("per-arm", xy=(len(models) - 1 - 0.13, last["pa_ab_hi"]),
                xytext=(0, 3), textcoords="offset points", fontsize=6.8,
                color=ORANGE, ha="center", va="bottom")
    ax.annotate("shared", xy=(len(models) - 1 + 0.13, last["sh_ab_hi"]),
                xytext=(0, 3), textcoords="offset points", fontsize=6.8,
                color=BLUE, ha="center", va="bottom")
    ax.set_title(r"(b) the component at 384 sequences", loc="left", pad=3)
    save(fig, "fig3_collapse.pdf")


# ---------------------------------------------------------------- figure 4 --
def fig_ladder():
    """Agreement about where to measure, against evaluation corpus size.

    The mean over the fifteen arm pairs is drawn for each base model, the
    same statistic the TopAgree and Jaccard macros quote, the
    Qwen3.5 ladder in blue and the others in grey, and every pair of every
    model sits behind it as faint dots, because the mean alone hides a
    spread of several to one across pairs.
    """
    agree, jacc = {}, {}
    for model, n_seq, pap, _ in CONDS:
        d = pd.read_csv(pap)
        agree[(model, n_seq)] = pair_agreement(d)
        jacc[(model, n_seq)] = overlap_stats(d)[0].mean_jaccard.to_numpy()

    models = MODELS
    corpora = [96, 384, 1536]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.0))
    fig.subplots_adjust(left=0.075, right=0.885, top=0.87, bottom=0.215,
                        wspace=0.60)

    for ax, store, ylab, title, top in (
            (axes[0], agree, r"same token (\%)", "(a) top-1 agreement", 44),
            (axes[1], jacc, "Jaccard", "(b) position set overlap", 0.42)):
        for c in corpora:
            v = np.concatenate([store[(m, c)] for m in models])
            x = c * np.exp(rng.normal(0, 0.035, len(v)))
            ax.plot(x, v, "o", ms=1.6, color=FAINT, markeredgewidth=0,
                    zorder=1)
        ends = []
        for model in sorted(models, key=lambda m: style(m)["z"]):
            st = style(model)
            meds = [float(np.mean(store[(model, c)])) for c in corpora]
            ax.plot(corpora, meds, lw=st["lw"], marker="o",
                    ms=2.6 if model in QWEN else 2.0, color=st["c"],
                    zorder=st["z"])
            ends.append((meds[-1], NAMES[model], st["lab"]))
        ax.set_xscale("log", base=2)
        ax.set_xticks(corpora)
        ax.set_xticklabels([str(c) for c in corpora])
        ax.minorticks_off()
        ax.set_xlim(72, 1800)
        ax.set_ylim(0, top)
        end_labels(ax, [e[0] for e in ends], [e[1] for e in ends], 1900,
                   colour=[e[2] for e in ends])
        ax.set_xlabel("evaluation corpus (sequences)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", pad=3)
    save(fig, "fig4_ladder.pdf")


if __name__ == "__main__":
    only = set(sys.argv[1:])
    for f in (fig_selection, fig_released, fig_collapse, fig_ladder):
        if not only or f.__name__ in only:
            f()
