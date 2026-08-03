# -*- coding: utf-8 -*-
"""Two controls on the causal-tail claim, both CPU. Usage: controls2.py [csv]

CONTROL A -- ASYMMETRIC RESIDUALISATION.
The earlier symmetric version fitted a +2.506 slope on the random arm against a
predictor with rho=0.016, p=0.82: residualising against noise. The random arm's
max|cos| is essentially constant (~0.085), so there is nothing there to partial
out. Only the trained arm gets corrected -- slope +0.241, rho=0.22, p=0.0006.

UNITS. Removing the full fitted value (intercept included) puts trained
residuals on a different scale from random raw values, making the comparison
meaningless. Instead each trained feature is adjusted to what its causal mass
would be if its max|cos| sat at the RANDOM arm's level:
    adjusted = 10 ** ( log10(kpn) - beta1 * (mc - mc_ref) ),  mc_ref = 0.085
Scale preserved, so "fraction above random p99" stays interpretable.

CONTROL B -- JACKKNIFE THE TOP-K.
A bootstrap resamples the few extreme features in and out but still treats them
as draws from a population. If dropping three features of 240 takes the SD ratio
from 42x to ~2x, the statistic is not estimating anything population-level, and
saying so directly is cleaner than a wide interval.

SUBSETTING'S OWN BIAS, stated for the record: thresholding on max|cos| truncates
the trained arm's upper tail on a variable correlated with the outcome while
leaving the random arm intact, so part of any collapse is mechanical. That is
why both controls are reported rather than either alone.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from scipy import stats

torch.set_num_threads(8)
SRC = sys.argv[1] if len(sys.argv) > 1 else "results/sae_rare.csv"
MC_REF = 0.085
B = 4000
rng = np.random.default_rng(0)


def summarise(t, r, label):
    q = np.percentile(r, 99)
    s = np.sort(t)[::-1]
    k = max(1, len(s) // 20)
    fb, mb = [], []
    for _ in range(B):
        tb = rng.choice(t, size=len(t), replace=True)
        rb = rng.choice(r, size=len(r), replace=True)
        fb.append((tb > np.percentile(rb, 99)).mean())
        ss = np.sort(tb)[::-1]
        mb.append(ss[:max(1, len(ss) // 20)].sum() / ss.sum())
    fb, mb = np.array(fb), np.array(mb)
    print(f"  {label:<32} n={len(t):5d} medratio={np.median(t)/np.median(r):5.2f}"
          f"  sd={t.std()/r.std():7.1f}x  frac>p99={(t > q).mean():5.1%} "
          f"CI[{np.percentile(fb,2.5):.1%},{np.percentile(fb,97.5):.1%}]"
          f"  mass5={s[:k].sum()/s.sum():5.1%} "
          f"CI[{np.percentile(mb,2.5):.1%},{np.percentile(mb,97.5):.1%}]")


def main():
    df = pd.read_csv(SRC)
    f = (df.groupby(["kind", "fid"])
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    W = torch.tensor(np.load(hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        "layer_12/width_16k/average_l0_82/params.npz"))["W_dec"],
        dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)

    T = f[f.kind == "trained"].copy()
    R = f[f.kind == "random"].copy()
    idx = torch.tensor(T.fid.values, dtype=torch.long)
    mc = np.zeros(len(idx))
    for a in range(0, len(idx), 512):
        blk = (W[idx[a:a + 512]] @ W.T).abs()
        for r_, i in enumerate(idx[a:a + 512]):
            blk[r_, i] = 0.0
        mc[a:a + 512] = blk.max(dim=1).values.numpy()
    T["mc"] = mc
    r = R.kpn.to_numpy()
    print(f"source={SRC}  n_trained={len(T)}  n_random={len(R)}")

    y = np.log10(T.kpn.values)
    X = np.column_stack([np.ones(len(T)), T.mc.values])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    rho = stats.spearmanr(T.kpn, T.mc)
    print(f"trained-arm fit: slope={b[1]:+.3f}  rho={rho.statistic:+.3f} "
          f"p={rho.pvalue:.4f}   (random arm left uncorrected by design)")
    T["adj"] = 10 ** (y - b[1] * (T.mc.values - MC_REF))

    print("\n" + "=" * 96)
    print("CONTROL A: asymmetric residualisation (trained adjusted to "
          f"max|cos|={MC_REF})")
    print("=" * 96)
    summarise(T.kpn.to_numpy(), r, "raw")
    summarise(T["adj"].to_numpy(), r, "adjusted for max|cos|")

    print("\n" + "=" * 96)
    print("CONTROL B: jackknife -- drop the top-k trained features by mass")
    print("=" * 96)
    s = np.sort(T.kpn.to_numpy())[::-1]
    for kdrop in (0, 1, 2, 3, 5, 10):
        t = s[kdrop:]
        ss = np.sort(t)[::-1]
        kk = max(1, len(ss) // 20)
        print(f"  drop top-{kdrop:<2d} n={len(t):4d}  "
              f"sd_ratio={t.std()/r.std():7.1f}x  "
              f"mass5={ss[:kk].sum()/ss.sum():5.1%}  "
              f"medratio={np.median(t)/np.median(r):5.2f}  "
              f"max={t.max():.5f}")

    print("\n" + "=" * 96)
    print("CONTROL C (reference): threshold subsetting, with its known bias")
    print("=" * 96)
    for thr in (0.35, 0.25, 0.20):
        sub = T[T.mc < thr]
        if len(sub) >= 20:
            summarise(sub.kpn.to_numpy(), r, f"max|cos| < {thr}")


if __name__ == "__main__":
    main()
