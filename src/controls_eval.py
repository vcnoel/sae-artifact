# -*- coding: utf-8 -*-
"""Control battery on the trained-vs-SOFT-FROZEN comparison. CPU only -- the
nine-dictionary sweep owns the GPU.

The headline (median ratio 1.13 norm-controlled, p=0.022; 0.83 raw, n.s.) needs
the same three controls the untrained-baseline comparison got, because the
artifacts are properties of the measurement, not of which control arm is used:

  jackknife       is the 8.2x sd ratio again one latent?
  alignment       does direct-path alignment inflate this comparison too?
                  computed on MY decoders, both arms, from the tied embedding.
  max|cos|        near-duplicate structure -- both arms have their own.

Both arms here are trained dictionaries, so unlike the untrained comparison the
covariate relationships may be symmetric. If both arms show a real alignment
relationship, one-sided adjustment is NOT licensed and the correct control is
matched-support or two-sided adjustment. That is checked before adjusting.
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


def main():
    df = pd.read_csv("results/eval_saes.csv")
    f = (df.groupby(["arm", "fid"])
         .agg(kpn=("kl_per_norm", "median"), kl=("kl", "median"),
              freq=("freq", "first")).reset_index())
    sd = torch.load("data/saes.pt", map_location="cpu")
    W = {}
    for arm in ("trained", "frozen"):
        w = sd[arm]["W_dec"].float()
        W[arm] = w / w.norm(dim=-1, keepdim=True)

    snap = glob.glob("~/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    files = sorted(glob.glob(snap + "*.safetensors"))

    parts = []
    Ds = []
    for arm in ("trained", "frozen"):
        s = f[f.arm == arm].copy()
        ids = torch.tensor(s.fid.values, dtype=torch.long)
        s["mc"] = maxcos(W[arm], ids)
        Ds.append(W[arm][ids].T.contiguous())
        parts.append(s)
    al = align_of(files, torch.cat(Ds, 1))
    n0 = Ds[0].shape[1]
    parts[0]["ua"], parts[1]["ua"] = al[:n0], al[n0:]
    T, R = parts[0], parts[1]

    print(f"n_trained={len(T)}  n_frozen={len(R)}")
    print("\nDECODER GEOMETRY OF THE TWO ARMS")
    for lab, S in (("trained", T), ("frozen", R)):
        print(f"  {lab:>8}: max|cos| med={S['mc'].median():.3f}  "
              f"alignment med={S['ua'].median():.5f}")

    print("\nCOVARIATE RELATIONSHIPS (both arms are trained dictionaries here,\n"
          "  so symmetry must be checked before one-sided adjustment)")
    for c in ("mc", "ua"):
        a = stats.spearmanr(T.kpn, T[c])
        b = stats.spearmanr(R.kpn, R[c])
        print(f"  {c:>3}: trained {a.statistic:+.3f} (p={a.pvalue:.4f})  |  "
              f"frozen {b.statistic:+.3f} (p={b.pvalue:.4f})")

    r = R.kpn.to_numpy()
    print("\nJACKKNIFE (norm-controlled)")
    v = np.sort(T.kpn.to_numpy())[::-1]
    for k in (0, 1, 2, 3, 5, 10):
        t = v[k:]
        s2 = np.sort(t)[::-1]
        print(f"  drop top-{k:<2d} n={len(t):4d} sd_ratio={t.std()/r.std():6.1f}x"
              f"  top5%mass={s2[:max(1,len(s2)//20)].sum()/s2.sum():5.1%}"
              f"  medratio={np.median(t)/np.median(r):5.2f}")

    print("\nMATCHED-SUPPORT ON ALIGNMENT (primary design)")
    lo, hi = R["ua"].min(), R["ua"].max()
    ins = T[(T["ua"] >= lo) & (T["ua"] <= hi)]
    if len(ins) >= 20:
        t = ins.kpn.to_numpy()
        mb = [np.median(rng.choice(t, len(t), True)) /
              np.median(rng.choice(r, len(r), True)) for _ in range(B)]
        print(f"  frozen alignment range [{lo:.5f}, {hi:.5f}]")
        print(f"  n={len(ins)}/{len(T)}  ratio={np.median(t)/np.median(r):.2f} "
              f"95% CI [{np.percentile(mb,2.5):.2f}, {np.percentile(mb,97.5):.2f}]")
    else:
        print(f"  only {len(ins)} trained latents inside frozen's alignment "
              f"range -- matching infeasible, report adjustment only")


if __name__ == "__main__":
    main()
