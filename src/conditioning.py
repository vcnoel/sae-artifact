# -*- coding: utf-8 -*-
"""Is the alignment regression well conditioned enough to report at all? CPU.

The alignment covariate spans 0.0312-0.0575 in the trained arm and the fitted
slope is +48.8 in log10 space. A narrow covariate range with a large coefficient
is the same shape as the +2.506-on-noise fit already caught once on the random
arm. If a quadratic term moves the adjusted ratio materially, the regression arm
should not be reported and the matched-support design carries the estimate alone
-- which is no loss, since matching involves no modelling assumption.
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

torch.set_num_threads(8)
rng = np.random.default_rng(0)
B = 4000


def align_of(files, D, chunk=16384):
    n = D.shape[1]
    s1 = torch.zeros(n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    cnt = 0
    for fp in files:
        t = load_file(fp)
        key = next((k for k in t if "embed_tokens" in k), None)
        if key is None:
            continue
        E = t[key]
        for a in range(0, E.shape[0], chunk):
            V = E[a:a + chunk].float() @ D
            s1 += V.sum(0).double()
            s2 += (V ** 2).sum(0).double()
            cnt += V.shape[0]
        del t, E
    mu = s1 / cnt
    return torch.sqrt((s2 / cnt - mu ** 2).clamp_min(0)).float().numpy()


def main():
    df = pd.read_csv("results/sae_rare.csv")
    f = (df.groupby(["kind", "fid"])
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    T, R = f[f.kind == "trained"].copy(), f[f.kind == "random"].copy()
    z = np.load(hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        "layer_12/width_16k/average_l0_82/params.npz"))
    W = torch.tensor(z["W_dec"], dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)
    g = torch.Generator().manual_seed(1)
    Rd = torch.randn(*W.shape, generator=g)
    Rd = Rd / Rd.norm(dim=-1, keepdim=True)
    Dt = W[torch.tensor(T.fid.values, dtype=torch.long)].T.contiguous()
    Dr = Rd[torch.tensor(R.fid.values, dtype=torch.long)].T.contiguous()
    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    al = align_of(sorted(glob.glob(snap + "*.safetensors")),
                  torch.cat([Dt, Dr], 1))
    T["ua"], R["ua"] = al[:Dt.shape[1]], al[Dt.shape[1]:]

    r = R.kpn.to_numpy()
    y = np.log10(T.kpn.values)
    x = T["ua"].values
    ref = float(R["ua"].median())

    def ratio(t):
        mb = [np.median(rng.choice(t, len(t), True)) /
              np.median(rng.choice(r, len(r), True)) for _ in range(B)]
        return np.median(t) / np.median(r), np.percentile(mb, 2.5), \
            np.percentile(mb, 97.5)

    print("ALIGNMENT FIT CONDITIONING")
    print(f"  covariate range [{x.min():.4f}, {x.max():.4f}]  "
          f"span={x.max()-x.min():.4f}  sd={x.std():.5f}  "
          f"span/sd={(x.max()-x.min())/x.std():.1f}")
    for deg in (1, 2, 3):
        X = np.column_stack([x ** k for k in range(deg + 1)])
        b, _, _, sv = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ b
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        cond = sv.max() / sv.min()
        Xr = np.column_stack([np.full(len(x), ref) ** k
                              for k in range(deg + 1)])
        adj = 10 ** (y - (pred - Xr @ b))
        pt, lo, hi = ratio(adj)
        print(f"  degree {deg}: R2={r2:.4f}  cond={cond:.2e}  "
              f"adjusted ratio={pt:.2f} [{lo:.2f}, {hi:.2f}]")

    print("\nMATCHED-SUPPORT (no model) -- the primary estimate")
    lo_, hi_ = R["ua"].min(), R["ua"].max()
    ins = T[(T["ua"] >= lo_) & (T["ua"] <= hi_)]
    pt, lo, hi = ratio(ins.kpn.to_numpy())
    print(f"  random alignment range [{lo_:.4f}, {hi_:.4f}]")
    print(f"  n={len(ins)}/{len(T)}   ratio={pt:.2f} [{lo:.2f}, {hi:.2f}]")
    tt = np.sort(ins.kpn.to_numpy())[::-1]
    print(f"  14119 inside subset: {14119 in set(ins.fid)}")
    for k in (1, 3):
        print(f"  drop top-{k}: ratio={np.median(tt[k:])/np.median(r):.2f}  "
              f"sd_ratio={tt[k:].std()/r.std():.1f}x")
    print(f"  trained alignment quantiles inside subset: "
          f"median={np.median(ins['ua']):.4f} vs random median={ref:.4f}")


if __name__ == "__main__":
    main()
