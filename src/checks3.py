# -*- coding: utf-8 -*-
"""Three checks on the 1.56x endpoint. CPU.

(1) EXTRAPOLATION RANGE. The adjustment moves trained features to the random
    arm's median alignment (0.0338). If the trained distribution sits well above
    that, the linear-in-log fit is extrapolating outside observed support. Also
    run a MATCHED-SUPPORT comparison: trained features whose alignment falls
    inside the random arm's observed range, compared to random with no
    adjustment at all. Matching instead of extrapolating.

(2) SLOPE SHIFT. The joint estimate (1.56) exceeds the alignment-only estimate
    (1.42). Report the single-fit and joint-fit slopes so the shift is visible
    rather than asserted.

(3) JACKKNIFE THE ADJUSTED VALUES. Top-5% mass is still 50.3% post-adjustment.
    Is that still feature 14119, or has adjustment redistributed the tail?
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


def maxcos(D, fids):
    out = np.zeros(len(fids))
    for a in range(0, len(fids), 512):
        idx = fids[a:a + 512]
        blk = (D[idx] @ D.T).abs()
        for r_, i in enumerate(idx):
            blk[r_, i] = 0.0
        out[a:a + 512] = blk.max(dim=1).values.numpy()
    return out


def ratio_ci(t, r):
    mb = [np.median(rng.choice(t, len(t), True)) /
          np.median(rng.choice(r, len(r), True)) for _ in range(B)]
    return np.median(t) / np.median(r), np.percentile(mb, 2.5), \
        np.percentile(mb, 97.5)


def main():
    df = pd.read_csv("sae_rare.csv")
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
    T["mc"] = maxcos(W, torch.tensor(T.fid.values, dtype=torch.long))
    R["mc"] = maxcos(Rd, torch.tensor(R.fid.values, dtype=torch.long))
    Dt = W[torch.tensor(T.fid.values, dtype=torch.long)].T.contiguous()
    Dr = Rd[torch.tensor(R.fid.values, dtype=torch.long)].T.contiguous()
    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    al = align_of(sorted(glob.glob(snap + "*.safetensors")),
                  torch.cat([Dt, Dr], 1))
    T["ua"], R["ua"] = al[:Dt.shape[1]], al[Dt.shape[1]:]
    r = R.kpn.to_numpy()

    print("=" * 78)
    print("(1) EXTRAPOLATION RANGE")
    print("=" * 78)
    q = [0, 5, 25, 50, 75, 95, 100]
    print("  alignment quantiles      " + " ".join(f"p{x:<3}" for x in q))
    for lab, S in (("trained", T), ("random", R)):
        print(f"  {lab:>22}   " +
              " ".join(f"{np.percentile(S['ua'], x):.4f}" for x in q))
    lo, hi = R["ua"].min(), R["ua"].max()
    inside = T[(T["ua"] >= lo) & (T["ua"] <= hi)]
    print(f"\n  random observed range: [{lo:.4f}, {hi:.4f}]")
    print(f"  trained features inside that range: {len(inside)}/{len(T)} "
          f"({len(inside)/len(T):.1%})")
    print(f"  adjustment target 0.0338 is at trained percentile "
          f"{(T['ua'] < 0.0338).mean():.1%}")
    if len(inside) >= 20:
        pt, a, b = ratio_ci(inside.kpn.to_numpy(), r)
        print(f"\n  MATCHED-SUPPORT (no adjustment, overlapping alignment "
              f"only):")
        print(f"    median ratio = {pt:.2f}  95% CI [{a:.2f}, {b:.2f}]   "
              f"n={len(inside)}")
        m2 = inside[inside.mc <= R['mc'].max()]
        if len(m2) >= 20:
            pt, a, b = ratio_ci(m2.kpn.to_numpy(), r)
            print(f"    + also matched on max|cos|: ratio = {pt:.2f}  "
                  f"95% CI [{a:.2f}, {b:.2f}]   n={len(m2)}")

    print()
    print("=" * 78)
    print("(2) SLOPE SHIFT, single vs joint fit")
    print("=" * 78)
    y = np.log10(T.kpn.values)
    for cols in (["mc"], ["ua"], ["mc", "ua"]):
        X = np.column_stack([np.ones(len(T))] + [T[c].values for c in cols])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        print(f"  fit on {str(cols):<14} " +
              "  ".join(f"beta_{c}={b[k+1]:+8.3f}" for k, c in enumerate(cols)))
    rr = stats.spearmanr(T["mc"], T["ua"])
    print(f"  Spearman(max|cos|, alignment) within trained = "
          f"{rr.statistic:+.3f} p={rr.pvalue:.4f}   <- the collinearity that "
          f"moves the slopes")

    print()
    print("=" * 78)
    print("(3) JACKKNIFE THE ADJUSTED VALUES")
    print("=" * 78)
    X = np.column_stack([np.ones(len(T)), T["mc"].values, T["ua"].values])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    adj = y - b[1] * (T["mc"].values - float(R["mc"].median())) \
            - b[2] * (T["ua"].values - float(R["ua"].median()))
    T["adj"] = 10 ** adj
    ordr = T.sort_values("adj", ascending=False)
    print("  top-5 adjusted features:",
          [(int(x.fid), round(x.adj, 5)) for _, x in ordr.head(5).iterrows()])
    v = ordr.adj.to_numpy()
    for k in (0, 1, 2, 3, 5, 10):
        t = v[k:]
        s = np.sort(t)[::-1]
        print(f"  drop top-{k:<2d} n={len(t):4d} sd_ratio={t.std()/r.std():7.1f}x"
              f"  top5%mass={s[:max(1,len(s)//20)].sum()/s.sum():5.1%}"
              f"  medratio={np.median(t)/np.median(r):5.2f}")


if __name__ == "__main__":
    main()
