# -*- coding: utf-8 -*-
"""Is the surviving 2x median gap just direct-path alignment? CPU, one matmul.

THE THREAT. A trained decoder direction is by construction aligned with
directions the model actually writes to the residual stream, and those are the
directions that feed the unembedding. A random direction has no such alignment.
So trained features should have larger DIRECT-PATH effects than random ones
regardless of any computational role -- which means the surviving median gap may
be direct path too, not only the tail. This measures the alignment directly
instead of inferring it from downstream decay.

THE MEASURE. For decoder direction d, the direct logit contribution is W_U d.
A constant shift across all logits leaves the softmax unchanged, so the quantity
that matters is the SPREAD, not the norm:
    align(d) = std_v( W_U[v] . d )
Gemma-2 ties embeddings, so W_U is embed_tokens. Computed in vocab chunks with
running sums so the [vocab x n_feat] product is never materialised.

THEN: (a) does alignment predict causal mass within each arm, (b) do trained
directions have systematically higher alignment than random ones, and (c) does
the median gap survive residualising on alignment -- asymmetrically, if the
random arm shows no relationship (the lesson from the max|cos| control).
"""
import glob
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy import stats

torch.set_num_threads(8)
rng = np.random.default_rng(0)
B = 4000


def align_of(E_files, D, chunk=16384):
    """std over vocab of (E @ d) for each column of D, streamed over shards."""
    n = D.shape[1]
    s1 = torch.zeros(n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    cnt = 0
    for f in E_files:
        t = load_file(f)
        key = next((k for k in t if "embed_tokens" in k), None)
        if key is None:
            continue
        E = t[key]
        for a in range(0, E.shape[0], chunk):
            blk = E[a:a + chunk].float()
            V = blk @ D                      # [chunk, n]
            s1 += V.sum(0).double()
            s2 += (V ** 2).sum(0).double()
            cnt += blk.shape[0]
        del t, E
    mean = s1 / cnt
    var = s2 / cnt - mean ** 2
    return torch.sqrt(var.clamp_min(0)).float().numpy(), cnt


def tail(v):
    s = np.sort(v)[::-1]
    k = max(1, len(s) // 20)
    return s[:k].sum() / s.sum()


def main():
    df = pd.read_csv("sae_rare.csv")
    f = (df.groupby(["kind", "fid"])
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    T = f[f.kind == "trained"].copy()
    R = f[f.kind == "random"].copy()

    z = np.load(hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        "layer_12/width_16k/average_l0_82/params.npz"))
    W = torch.tensor(z["W_dec"], dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)
    d = W.shape[1]
    g = torch.Generator().manual_seed(1)
    Rd = torch.randn(W.shape[0], d, generator=g)
    Rd = Rd / Rd.norm(dim=-1, keepdim=True)

    Dt = W[torch.tensor(T.fid.values, dtype=torch.long)].T.contiguous()
    Dr = Rd[torch.tensor(R.fid.values, dtype=torch.long)].T.contiguous()
    D = torch.cat([Dt, Dr], dim=1)

    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    files = sorted(glob.glob(snap + "*.safetensors"))
    al, vocab = align_of(files, D)
    T['ua'] = al[:Dt.shape[1]]
    R['ua'] = al[Dt.shape[1]:]
    print(f"vocab rows used: {vocab}")

    print("\n" + "=" * 78)
    print("UNEMBEDDING ALIGNMENT  std_v(W_U . d)")
    print("=" * 78)
    for lab, S in (("trained", T), ("random", R)):
        print(f"  {lab:>8}: median={S['ua'].median():.4f} "
              f"mean={S['ua'].mean():.4f} p90={S['ua'].quantile(.9):.4f} "
              f"max={S['ua'].max():.4f}")
    print(f"  ratio of medians trained/random = "
          f"{T['ua'].median()/R['ua'].median():.3f}")
    print(f"  Mann-Whitney p = "
          f"{stats.mannwhitneyu(T['ua'], R['ua']).pvalue:.2e}")

    print("\n  does alignment predict causal mass, within arm?")
    for lab, S in (("trained", T), ("random", R)):
        r = stats.spearmanr(S.kpn, S['ua'])
        print(f"    {lab:>8}: Spearman(kpn, align) = {r.statistic:+.3f}  "
              f"p={r.pvalue:.4f}")

    # feature 14119 in context
    if 14119 in set(T.fid):
        row = T[T.fid == 14119].iloc[0]
        pct = (T['ua'] < row['ua']).mean()
        print(f"\n  feature 14119: align={row['ua']:.4f} "
              f"({pct:.1%} percentile of trained arm)")

    print("\n" + "=" * 78)
    print("DOES THE MEDIAN GAP SURVIVE RESIDUALISING ON ALIGNMENT?")
    print("=" * 78)
    rho_t = stats.spearmanr(T.kpn, T['ua'])
    rho_r = stats.spearmanr(R.kpn, R['ua'])
    y = np.log10(T.kpn.values)
    X = np.column_stack([np.ones(len(T)), T['ua'].values])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    ref = float(R['ua'].median())
    T["adj"] = 10 ** (y - b[1] * (T['ua'].values - ref))
    print(f"  trained slope={b[1]:+.3f} (rho={rho_t.statistic:+.3f}, "
          f"p={rho_t.pvalue:.4f});  random rho={rho_r.statistic:+.3f}, "
          f"p={rho_r.pvalue:.4f}")
    print(f"  trained adjusted to random's median alignment ({ref:.4f})")
    r = R.kpn.to_numpy()
    for col, lab in (("kpn", "raw"), ("adj", "alignment-adjusted")):
        t = T[col].to_numpy()
        mb = []
        for _ in range(B):
            tb = rng.choice(t, size=len(t), replace=True)
            rb = rng.choice(r, size=len(r), replace=True)
            mb.append(np.median(tb) / np.median(rb))
        mb = np.array(mb)
        print(f"    {lab:<20} median ratio = "
              f"{np.median(t)/np.median(r):5.2f}  95% CI "
              f"[{np.percentile(mb,2.5):.2f}, {np.percentile(mb,97.5):.2f}]"
              f"   top5%mass={tail(t):.1%}")


if __name__ == "__main__":
    main()
