# -*- coding: utf-8 -*-
"""Figure 1: how far apart two dictionaries measure the same latent, and what
that costs inside a single latent.

(a) The distance between the token dictionary A picks and the token dictionary B
    picks, pooled over all 15 arm pairs and every latent, against a null in
    which each arm's positions are shuffled among the positions that latent
    actually fires at. The null is what makes 10% agreement interpretable: it is
    far above chance -- the dictionaries are not choosing at random -- and still
    leaves most of the mass hundreds of tokens away.
(b) The span of a latent's own measured effect across the six positions it was
    measured at, sorted. Position does not merely move the measurement, it
    changes the answer.

WHAT THIS PANEL REPLACED, AND WHY. The first version led with one hand-picked
latent whose two argmaxes were 14 tokens apart. The population median is 126, so
the lead panel understated the paper's own claim by nine times, from an anecdote,
while 240 latents of population data sat unused. It also omitted the null, which
cuts the other way: without it a reader cannot tell whether 10% is bad. Both are
fixed here.

FORM. (a) is a distribution against a reference distribution -> step histogram
plus a line for the null, one hue plus grey (emphasis, not categorical: the
null is context, not a peer series). (b) is a per-item range -> dumbbell sorted
by the ranked value.

COLOUR. blue #2a78d6 for the observed, muted grey for the null; validated with
scripts/validate_palette.js.
"""
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

BLUE, B_LIGHT, B_DARK = "#2a78d6", "#86b6ef", "#1c5cab"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
TW = 5.5

plt.rcParams.update({
    "font.size": 6.4, "axes.labelsize": 6.4, "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0, "axes.titlesize": 7.2,
    "font.family": "sans-serif", "pdf.fonttype": 42,
})

PER_ARM = "results/eval_arms_g2_s384.csv"
REF = "free"


def chrome(ax, gridaxis="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=INK2, length=2.5, width=0.7, pad=1.5)
    ax.grid(True, axis=gridaxis, color=GRID, lw=0.5, ls="-", zorder=0)
    ax.set_axisbelow(True)


def displacements(d):
    """|position chosen by X - position chosen by Y| over every arm pair.

    The null shuffles, for each latent, which of ITS OWN firing positions each
    arm picks. Shuffling over all 511 tokens would be a straw man: the arms can
    only choose where the latent fires, so the null has to be constrained the
    same way or the comparison flatters us.
    """
    top = (d.sort_values("act", ascending=False).groupby(["fid", "arm"]).head(1)
           .pivot(index="fid", columns="arm", values="pos"))
    arms = list(top.columns)
    obs = []
    for x, y in itertools.combinations(arms, 2):
        p = top[[x, y]].dropna()
        obs.append((p[x] - p[y]).abs().to_numpy())
    obs = np.concatenate(obs)

    pool = d.groupby("fid")["pos"].apply(lambda s: np.unique(s.to_numpy()))
    rng = np.random.default_rng(0)
    null = []
    for _ in range(len(arms)):
        for fid, cand in pool.items():
            if len(cand) < 2:
                continue
            a, b = rng.choice(cand, 2, replace=True)
            null.append(abs(int(a) - int(b)))
    return obs, np.array(null)


def panel_a(ax):
    d = pd.read_csv(PER_ARM)
    obs, null = displacements(d)
    bins = np.linspace(0, 511, 34)
    ax.hist(obs, bins=bins, density=True, color=BLUE, alpha=0.85, zorder=3,
            label=f"observed ({len(obs):,} arm pairs)")
    hn, _ = np.histogram(null, bins=bins, density=True)
    ax.step(bins[:-1], hn, where="post", color=MUTED, lw=1.3, zorder=4,
            label="null (same latent, shuffled)")
    ax.set_xlim(0, 511)
    ax.set_xticks([0, 128, 256, 384, 511])
    ax.set_xlabel("distance between the two chosen tokens", color=INK2)
    ax.set_ylabel("density", color=INK2)
    ax.set_yticks([])
    med = np.median(obs)
    ax.axvline(med, color=B_DARK, lw=1.0, zorder=5)
    ax.annotate(f"median {med:.0f} tokens", xy=(med, ax.get_ylim()[1] * 0.62),
                xytext=(6, 0), textcoords="offset points", color=INK,
                fontsize=6.0, va="center")
    ax.legend(frameon=False, loc="upper right", fontsize=6.0, handlelength=1.1,
              handletextpad=0.4, labelspacing=0.22, borderpad=0.1,
              labelcolor=INK2)
    chrome(ax, gridaxis="y")
    ax.set_title("(a) where two dictionaries measure the same latent",
                 color=INK, loc="left", pad=4)


def panel_b(ax, n_show=70):
    d = pd.read_csv(PER_ARM)
    g = (d[d.arm == REF].groupby("fid").kl_per_norm
         .agg(lo="min", hi="max", med="median"))
    g = g[g.lo > 0]
    span = (g.hi / g.lo).median()
    g = g.sort_values("med")
    s = g.iloc[::max(1, len(g) // n_show)].reset_index(drop=True)
    y = np.arange(len(s))
    ax.hlines(y, s.lo, s.hi, color=B_LIGHT, lw=1.0, zorder=2)
    ax.plot(s.med, y, "o", ms=2.0, color=B_DARK, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks([])
    ax.set_ylim(-2, len(s) + 2)
    ax.set_xlabel("effect per unit norm (log)", color=INK2)
    ax.set_ylabel("latents, ranked by median", color=INK2, labelpad=1)
    ax.text(0.96, 0.06, f"median latent spans {span:.0f}$\\times$",
            transform=ax.transAxes, ha="right", va="bottom", color=INK,
            fontsize=6.0)
    ax.legend(handles=[
        Line2D([], [], color=B_LIGHT, lw=1.6, label="min to max over 6 positions"),
        Line2D([], [], marker="o", ls="", ms=3.0, color=B_DARK, label="median")],
        frameon=False, loc="upper left", fontsize=6.0, handlelength=1.1,
        handletextpad=0.4, labelspacing=0.22, borderpad=0.1, labelcolor=INK2)
    chrome(ax, gridaxis="x")
    ax.set_title("(b) the same latent, measured six ways", color=INK,
                 loc="left", pad=4)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(TW, 1.76))
    fig.patch.set_facecolor(SURFACE)
    panel_a(axes[0])
    panel_b(axes[1])
    fig.subplots_adjust(wspace=0.30, left=0.06, right=0.985, top=0.855,
                        bottom=0.195)
    fig.savefig("paper/fig_one.pdf", facecolor=SURFACE)
    print("wrote paper/fig_one.pdf")


if __name__ == "__main__":
    main()
