# -*- coding: utf-8 -*-
"""Does the causal tail survive partialling out near-duplicate structure?

WHY THIS REPLACES THE TWO-DOMAIN FRAMING. Mean-over-features nearest-neighbour
|cos| and Control 1's per-feature max |cos| are the same quantity. It cannot be
independent corroboration for the causal tail AND the confound on that tail. So
geometry appears exactly once, as the control: regress causal mass on max |cos|,
residualise, and recompute the tail statistics on the residual. If the tail
survives, the claim is "the causal tail is not a splitting artifact", and the
obvious alternative explanation is ruled out rather than double-counted.

Residualisation is done in log space because the quantity is heavy-tailed by
construction; the residual is reported multiplicatively (10**resid) so the
mass-share statistic remains interpretable as "causal effect with the
splitting-predicted component divided out".

Each arm is residualised against ITS OWN dictionary's max |cos|. For the random
dictionary that is nearly a no-op -- max |cos| there has almost no variance --
which is the honest treatment rather than a favourable one.

CPU only. Usage: python residualise.py [csv]
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
SRC = sys.argv[1] if len(sys.argv) > 1 else "sae_rare.csv"
B = 4000
rng = np.random.default_rng(0)
EPS = 1e-9


def maxcos(W, fids):
    out = np.zeros(len(fids))
    for a in range(0, len(fids), 512):
        idx = fids[a:a + 512]
        blk = (W[idx] @ W.T).abs()
        for r, i in enumerate(idx):
            blk[r, i] = 0.0
        out[a:a + 512] = blk.max(dim=1).values.numpy()
    return out


def tail_stats(v):
    """v: positive per-feature masses."""
    s = np.sort(v)[::-1]
    k = max(1, len(s) // 20)
    return s[:k].sum() / s.sum(), k


def main():
    df = pd.read_csv(SRC)
    f = (df.groupby(["kind", "fid"])
         .agg(kpn=("kl_per_norm", "median"), kl=("kl", "median"),
              freq=("freq", "first")).reset_index())
    print(f"source={SRC}  n: {f.groupby('kind').size().to_dict()}")

    p = hf_hub_download("google/gemma-scope-2b-pt-res",
                        "layer_12/width_16k/average_l0_82/params.npz")
    W = torch.tensor(np.load(p)["W_dec"], dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)
    n, d = W.shape
    g = torch.Generator().manual_seed(1)
    Rd = torch.randn(n, d, generator=g)
    Rd = Rd / Rd.norm(dim=-1, keepdim=True)

    parts = []
    for kind, D in (("trained", W), ("random", Rd)):
        sub = f[f.kind == kind].copy()
        # random arm's fids index the random dictionary's own columns
        sub["mc"] = maxcos(D, torch.tensor(sub.fid.values, dtype=torch.long))
        y = np.log10(sub.kpn.values + EPS)
        X = np.column_stack([np.ones(len(sub)), sub.mc.values])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        sub["resid_mass"] = 10 ** (y - X @ beta)
        r = stats.spearmanr(sub.kpn, sub.mc)
        sub["_slope"] = beta[1]
        sub["_rho"] = r.statistic
        parts.append(sub)
        print(f"  {kind:>8}: max|cos| median={np.median(sub.mc):.3f}  "
              f"slope={beta[1]:+.3f}  Spearman(kpn,max|cos|)="
              f"{r.statistic:+.3f} p={r.pvalue:.4f}")
    f = pd.concat(parts)

    T, R = f[f.kind == "trained"], f[f.kind == "random"]
    print()
    print("=" * 74)
    print("TAIL STATISTICS, RAW vs RESIDUALISED ON max|cos|")
    print("=" * 74)
    for col, lab in (("kpn", "raw (norm-controlled)"),
                     ("resid_mass", "residualised on max|cos|")):
        t, r = T[col].to_numpy(), R[col].to_numpy()
        q = np.percentile(r, 99)
        frac = (t > q).mean()
        ms, k = tail_stats(t)
        fb, mb = [], []
        for _ in range(B):
            tb = rng.choice(t, size=len(t), replace=True)
            rb = rng.choice(r, size=len(r), replace=True)
            fb.append((tb > np.percentile(rb, 99)).mean())
            mb.append(tail_stats(tb)[0])
        fb, mb = np.array(fb), np.array(mb)
        print(f"\n[{lab}]")
        print(f"  median ratio t/r     = "
              f"{np.median(t)/max(np.median(r),1e-12):.2f}")
        print(f"  sd ratio t/r         = {t.std()/max(r.std(),1e-12):.1f}x")
        print(f"  frac trained > r-p99 = {frac:.1%}  "
              f"95% CI [{np.percentile(fb,2.5):.1%}, "
              f"{np.percentile(fb,97.5):.1%}]")
        print(f"  top-5% mass share    = {ms:.1%}  "
              f"95% CI [{np.percentile(mb,2.5):.1%}, "
              f"{np.percentile(mb,97.5):.1%}]  (on {k} features)")


if __name__ == "__main__":
    main()
