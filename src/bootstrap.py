# -*- coding: utf-8 -*-
"""How unstable are the headline tail numbers at n=240/198? CPU only.

Three quantities carry the claim and all three are tail estimates from small
samples. This bootstraps each so the resample decision is made on measured
instability rather than on intuition:

  1. the random-arm p99 -- at n=198 this is effectively the 2nd-largest value
  2. "fraction of trained features above the random p99" -- the headline
  3. "share of causal mass in the top 5% of trained features" -- rests on 12

Also answers a separate question: is there a MEAN-based geometric statistic that
separates a trained decoder from a random one? Mean |cos| over PAIRS does not,
and arguably cannot -- 16384 unit vectors in 2304 dimensions are near-orthogonal
on average whether trained or not, which is what an overcomplete dictionary that
spans the space is supposed to look like. But the mean over FEATURES of each
feature's nearest-neighbour cosine is also a mean, and it is not obliged to be
at chance. That is computed here for both dictionaries.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

torch.set_num_threads(8)
B = 4000
rng = np.random.default_rng(0)


def ci(v, lo=2.5, hi=97.5):
    return np.percentile(v, lo), np.percentile(v, hi)


def main():
    df = pd.read_csv("sae_rare.csv")
    f = (df.groupby(["kind", "fid"])
         .agg(kl=("kl", "median"), kpn=("kl_per_norm", "median"))
         .reset_index())
    T = f[f.kind == "trained"]
    R = f[f.kind == "random"]

    print("=" * 76)
    print("BOOTSTRAP OF THE THREE TAIL QUANTITIES")
    print(f"  n_trained={len(T)}  n_random={len(R)}   B={B}")
    print("=" * 76)

    for col in ("kl", "kpn"):
        t, r = T[col].to_numpy(), R[col].to_numpy()
        p99s, fracs, mass = [], [], []
        for _ in range(B):
            rb = rng.choice(r, size=len(r), replace=True)
            tb = rng.choice(t, size=len(t), replace=True)
            q = np.percentile(rb, 99)
            p99s.append(q)
            fracs.append((tb > q).mean())
            s = np.sort(tb)[::-1]
            k = max(1, len(s) // 20)
            mass.append(s[:k].sum() / s.sum())
        p99s, fracs, mass = map(np.array, (p99s, fracs, mass))
        print(f"\n[{col}]")
        print(f"  random p99          point={np.percentile(r,99):.6f}  "
              f"95% CI=[{ci(p99s)[0]:.6f}, {ci(p99s)[1]:.6f}]  "
              f"spread={ci(p99s)[1]/max(ci(p99s)[0],1e-12):.1f}x")
        print(f"  frac trained > p99  point={(t>np.percentile(r,99)).mean():.1%}"
              f"  95% CI=[{ci(fracs)[0]:.1%}, {ci(fracs)[1]:.1%}]")
        s = np.sort(t)[::-1]
        k = max(1, len(s) // 20)
        print(f"  top-5% mass share   point={s[:k].sum()/s.sum():.1%}  "
              f"95% CI=[{ci(mass)[0]:.1%}, {ci(mass)[1]:.1%}]   "
              f"(rests on {k} features)")

    # ---------------- mean-based geometric statistic ----------------
    print()
    print("=" * 76)
    print("IS THERE A MEAN-BASED GEOMETRIC STATISTIC THAT SEPARATES?")
    print("  mean over PAIRS of |cos|  vs  mean over FEATURES of "
          "nearest-neighbour |cos|")
    print("=" * 76)
    p = hf_hub_download("google/gemma-scope-2b-pt-res",
                        "layer_12/width_16k/average_l0_82/params.npz")
    W = torch.tensor(np.load(p)["W_dec"], dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)
    n, d = W.shape
    g = torch.Generator().manual_seed(1)
    Rd = torch.randn(n, d, generator=g)
    Rd = Rd / Rd.norm(dim=-1, keepdim=True)

    for lab, M in (("gemma_scope", W), ("random", Rd)):
        nn, pair = [], []
        for i in range(0, n, 2048):
            blk = (M[i:i + 2048] @ M.T).abs()
            for r_ in range(blk.shape[0]):
                blk[r_, i + r_] = 0.0
            nn.append(blk.max(dim=1).values)
            pair.append(blk.sum(dim=1))
        nn = torch.cat(nn)
        mean_pair = float(torch.cat(pair).sum() / (n * (n - 1)))
        print(f"  {lab:>12}  mean-over-pairs |cos| = {mean_pair:.5f}   "
              f"mean-over-features NN|cos| = {nn.mean():.4f}  "
              f"median={nn.median():.4f}  p99={nn.quantile(.99):.4f}")


if __name__ == "__main__":
    main()
