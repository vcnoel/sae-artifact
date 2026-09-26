# -*- coding: utf-8 -*-
"""The endpoint number: trained-vs-random gap after controlling every measured
artifact simultaneously. CPU.

Covariates, each established as a real relationship in the trained arm and no
relationship in the random arm (which is what licenses one-sided adjustment):
  max|cos| to any other decoder row   near-duplicate / splitting structure
  std_v(W_U . d)                      direct-path unembedding alignment
Perturbation magnitude is already controlled by using KL per unit norm, and BOS
is excluded upstream.

log frequency is reported as a sensitivity line only, NOT folded into the primary
number: how often a feature fires is arguably part of what makes it useful rather
than an artifact of the measurement, so controlling for it would be
over-adjustment.
"""
import glob
import io
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
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
    for fpath in files:
        t = load_file(fpath)
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


def tail(v):
    s = np.sort(v)[::-1]
    return s[:max(1, len(s) // 20)].sum() / s.sum()


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "results/sae_rare.csv"
    df = pd.read_csv(src)
    f = (df.groupby(["kind", "fid"])
         .agg(kpn=("kl_per_norm", "median"), freq=("freq", "first"))
         .reset_index())
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
    snap = glob.glob("~/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    al = align_of(sorted(glob.glob(snap + "*.safetensors")),
                  torch.cat([Dt, Dr], 1))
    T["ua"], R["ua"] = al[:Dt.shape[1]], al[Dt.shape[1]:]
    T["lf"] = np.log10(T.freq.values)
    R["lf"] = np.log10(R.freq.values)

    print(f"source={src}  n_trained={len(T)} n_random={len(R)}")
    print("covariate relationships (trained | random), Spearman vs causal mass:")
    for c in ("mc", "ua", "lf"):
        a = stats.spearmanr(T.kpn, T[c])
        b = stats.spearmanr(R.kpn, R[c])
        print(f"  {c:>3}: trained {a.statistic:+.3f} (p={a.pvalue:.4f}) | "
              f"random {b.statistic:+.3f} (p={b.pvalue:.4f})")
    print(f"\nreference values (random medians): "
          f"max|cos|={R['mc'].median():.4f} align={R['ua'].median():.4f}\n")

    r = R.kpn.to_numpy()
    y = np.log10(T.kpn.values)
    specs = [("nothing", []), ("max|cos|", ["mc"]), ("alignment", ["ua"]),
             ("BOTH (primary)", ["mc", "ua"]),
             ("BOTH + log freq (sens.)", ["mc", "ua", "lf"])]
    for lab, cols in specs:
        if not cols:
            t = T.kpn.values
        else:
            X = np.column_stack([np.ones(len(T))] + [T[c].values for c in cols])
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            adj = y.copy()
            for k, c in enumerate(cols):
                adj = adj - b[k + 1] * (T[c].values - float(R[c].median()))
            t = 10 ** adj
        mb = []
        for _ in range(B):
            tb = rng.choice(t, size=len(t), replace=True)
            rb = rng.choice(r, size=len(r), replace=True)
            mb.append(np.median(tb) / np.median(rb))
        mb = np.array(mb)
        print(f"  adjust for {lab:<24} ratio={np.median(t)/np.median(r):5.2f} "
              f"95% CI [{np.percentile(mb,2.5):.2f},{np.percentile(mb,97.5):.2f}]"
              f"  sd={t.std()/r.std():7.1f}x  top5%={tail(t):5.1%}")


if __name__ == "__main__":
    main()
