# -*- coding: utf-8 -*-
"""Two deflationary controls on the Gemma Scope decoder. CPU only -- the GPU is
busy training and this needs no acceleration.

CONTROL 1: is the geometric tail just feature splitting?
  max |cos| = 0.964 is the documented signature of feature splitting: one
  concept fragmented across several latents pointing nearly the same way. If the
  features carrying the causal mass are the SAME features involved in
  near-duplicate pairs, then "use the causally dense subset" is recommending
  split features and the convergence story needs rewriting. Test: correlate each
  feature's max |cos| to any other row against its measured causal mass.

CONTROL 2: does the low stable rank survive centering?
  Stable rank 0.069 * 2304 ~ 159 effective dimensions for 16384 features is
  striking, but a shared bias-like direction among decoder rows produces exactly
  that: one dominant singular value, collapsed stable rank, no low-rank structure
  among the features themselves. sigma_1 being ~2.8x the random matrix's is the
  size of effect one shared direction would give. Test: recompute the spectrum
  with the mean decoder row removed.
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
NSUB = 4000


def spec(W, label):
    sv = torch.linalg.svdvals(W)
    lam = sv ** 2
    d = W.shape[1]
    pr = float(lam.sum() ** 2 / (lam ** 2).sum()) / d
    stable = float(W.norm() ** 2 / sv[0] ** 2) / d
    return dict(label=label, sigma1=float(sv[0]), sigma2=float(sv[1]),
                s1_over_s2=float(sv[0] / sv[1]), part_ratio=pr,
                stable_frac=stable, eff_dims=stable * d)


def main():
    p = hf_hub_download("google/gemma-scope-2b-pt-res",
                        "layer_12/width_16k/average_l0_82/params.npz")
    W = torch.tensor(np.load(p)["W_dec"], dtype=torch.float32)
    W = W / W.norm(dim=-1, keepdim=True)
    n, d = W.shape
    g = torch.Generator().manual_seed(1)
    R = torch.randn(n, d, generator=g)
    R = R / R.norm(dim=-1, keepdim=True)

    # ================= CONTROL 1 =================
    print("=" * 78)
    print("CONTROL 1: does the causal tail coincide with near-duplicate pairs?")
    print("=" * 78)
    df = pd.read_csv("sae_rare.csv")
    f = (df[df.kind == "trained"].groupby("fid")
         .agg(kl=("kl", "median"), kpn=("kl_per_norm", "median"),
              freq=("freq", "first")).reset_index())
    fids = torch.tensor(f.fid.values, dtype=torch.long)
    sims = (W[fids] @ W.T).abs()               # [n_sel, 16384]
    # Zero the self-similarity AFTER abs. Setting it to -2 before abs made
    # every row's max 2.0 and produced an all-constant column.
    sims[torch.arange(len(fids)), fids] = 0.0
    f["max_abs_cos"] = sims.max(dim=1).values.numpy()
    f["n_gt_0p3"] = (sims > 0.3).sum(dim=1).numpy()

    for col in ("kl", "kpn"):
        r = stats.spearmanr(f[col], f.max_abs_cos)
        print(f"  Spearman(max|cos|, {col:>3}) = {r.statistic:+.3f}  "
              f"p={r.pvalue:.3f}")
    cut = f[col].quantile(0.95)
    top = f[f.kpn >= f.kpn.quantile(0.95)]
    rest = f[f.kpn < f.kpn.quantile(0.95)]
    print(f"\n  top-5% causal features (n={len(top)}): "
          f"median max|cos| = {top.max_abs_cos.median():.3f}")
    print(f"  the rest        (n={len(rest)}): "
          f"median max|cos| = {rest.max_abs_cos.median():.3f}")
    print(f"  Mann-Whitney p = "
          f"{stats.mannwhitneyu(top.max_abs_cos, rest.max_abs_cos).pvalue:.3f}")
    print(f"\n  features with any |cos|>0.3 partner: "
          f"{(f.n_gt_0p3 > 0).mean():.1%} of sampled features")
    print(f"  max|cos| distribution over sampled features: "
          f"median={f.max_abs_cos.median():.3f} "
          f"p90={f.max_abs_cos.quantile(.9):.3f} max={f.max_abs_cos.max():.3f}")

    # ================= CONTROL 2 =================
    print()
    print("=" * 78)
    print("CONTROL 2: does low stable rank survive removing the mean row?")
    print("=" * 78)
    mu_W, mu_R = W.mean(0), R.mean(0)
    print(f"  ||mean decoder row||  gemma_scope={mu_W.norm():.4f}   "
          f"random={mu_R.norm():.4f}   (chance ~ 1/sqrt(n) = "
          f"{1/np.sqrt(n):.4f})")
    print(f"  -> shared-direction mass is "
          f"{float(mu_W.norm()/mu_R.norm()):.1f}x the random level\n")

    rows = []
    for lab, M in (("gemma_scope raw", W), ("random raw", R)):
        rows.append(spec(M, lab))
    for lab, M, mu in (("gemma_scope centred", W, mu_W),
                       ("random centred", R, mu_R)):
        C = M - mu
        rows.append(spec(C, lab))
        Cn = C / C.norm(dim=-1, keepdim=True)
        rows.append(spec(Cn, lab + "+renorm"))

    hdr = ["sigma1", "s1_over_s2", "part_ratio", "stable_frac", "eff_dims"]
    print(f"{'':>26} " + " ".join(f"{h:>12}" for h in hdr))
    for r in rows:
        print(f"{r['label']:>26} " + " ".join(f"{r[h]:12.4f}" for h in hdr))


if __name__ == "__main__":
    main()
