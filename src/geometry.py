# -*- coding: utf-8 -*-
"""Is a production-scale SAE decoder distinguishable from a random dictionary?

The budget objection to any small-scale SAE result is that 20M tokens is not
billions. This bounds that objection without training anything: compare DECODER
GEOMETRY across dictionaries. Cosine-to-init is unavailable for Gemma Scope (no
init released), but geometry is intrinsic and needs no reference point.

MEASURES, all on unit-norm decoder rows:
  mean |cos|      chance level is sqrt(2/(pi*d)) = 0.0166 at d=2304
  p99, max |cos|  near-duplicate structure a random dictionary cannot produce
  frac |cos|>0.1  fraction of pairs with structure well beyond chance
  participation ratio of squared singular values: (sum L)^2 / sum L^2,
                  normalised by d -- 1.0 means isotropic, lower means the
                  dictionary concentrates in a subspace
  stable rank     ||W||_F^2 / ||W||_2^2

If Gemma Scope is indistinguishable from random on these, the lazy-dynamics
reading extends to production scale and the budget objection collapses. If it is
clearly distinguishable while a 20M-token arm is not, that is undertraining and
should be said plainly.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import torch
from huggingface_hub import hf_hub_download

DEV = "cuda"
NSUB = 4000          # rows subsampled for the pairwise-cosine statistics


def stats_of(W, name):
    """W: [n_rows, d] on GPU, float32."""
    W = W / W.norm(dim=-1, keepdim=True)
    n, d = W.shape
    g = torch.Generator(device=DEV).manual_seed(0)
    idx = torch.randperm(n, generator=g, device=DEV)[:min(NSUB, n)]
    S = W[idx]
    G = (S @ S.T).abs()
    m = ~torch.eye(len(S), dtype=torch.bool, device=DEV)
    off = G[m]
    sv = torch.linalg.svdvals(W.float())
    lam = sv ** 2
    pr = float((lam.sum() ** 2 / (lam ** 2).sum()).item()) / d
    stable = float((W.norm() ** 2 / sv[0] ** 2).item())
    return dict(name=name, n=n, d=d,
                mean_abs_cos=float(off.mean().item()),
                p99_abs_cos=float(off.quantile(0.99).item()),
                max_abs_cos=float(off.max().item()),
                frac_gt_0p1=float((off > 0.1).float().mean().item()),
                part_ratio=pr, stable_rank_frac=stable / d)


def main():
    rows = []

    # --- Gemma Scope, production scale ---
    p = hf_hub_download("google/gemma-scope-2b-pt-res",
                        "layer_12/width_16k/average_l0_82/params.npz")
    gs = torch.tensor(np.load(p)["W_dec"], dtype=torch.float32, device=DEV)
    rows.append(stats_of(gs, "gemma_scope_16k_l0_82"))
    n, d = gs.shape

    # --- random unit-norm dictionary, matched shape ---
    g = torch.Generator(device=DEV).manual_seed(1)
    R = torch.randn(n, d, generator=g, device=DEV)
    rows.append(stats_of(R, "random_matched"))

    # --- my two arms, if trained yet ---
    if os.path.exists("data/saes.pt"):
        sd = torch.load("data/saes.pt", map_location=DEV)
        for arm in ("trained", "frozen"):
            if arm in sd and "W_dec" in sd[arm]:
                rows.append(stats_of(sd[arm]["W_dec"].float().to(DEV),
                                     f"mine_{arm}_20M"))
            if arm == "trained" and "W0" in sd[arm]:
                rows.append(stats_of(sd[arm]["W0"].float().to(DEV),
                                     "mine_init"))
    else:
        print("(saes.pt not present yet - my arms omitted)\n")

    hdr = ["name", "mean_abs_cos", "p99_abs_cos", "max_abs_cos",
           "frac_gt_0p1", "part_ratio", "stable_rank_frac"]
    print(f"chance mean|cos| for d={d}: {np.sqrt(2/(np.pi*d)):.4f}\n")
    print(f"{hdr[0]:>24} " + " ".join(f"{h:>13}" for h in hdr[1:]))
    for r in rows:
        print(f"{r['name']:>24} " + " ".join(f"{r[h]:13.5f}" for h in hdr[1:]))


if __name__ == "__main__":
    main()
