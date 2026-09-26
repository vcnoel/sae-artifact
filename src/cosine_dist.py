# -*- coding: utf-8 -*-
"""E29b: is a cosine band the same thing at 2B as at 9B?

The band comparison across model scales assumes "0.50-0.60" denotes the same
degree of feature agreement at both. It may not. Gemma-2-9B has d_model 3584
against 2B's 2304, so a 16k dictionary is relatively sparser in a
higher-dimensional space, and the typical mutual-nearest-neighbour cosine
between two independently trained dictionaries can shift with that. If 9B's
matched cosines sit systematically lower, then a fixed band is a different
degree of agreement at the two scales and the comparison is not like-for-like.

This reports the FULL mutual-nearest-neighbour cosine distribution per pair:
median, quartiles, and the share of the dictionary in each band. Two readings
then become available and both are reported:

  matched ABSOLUTE cosine  -- the band comparison as originally written, valid
                              only if the distributions overlap;
  matched PERCENTILE       -- the same quantile of each scale's own
                              distribution, which is like-for-like by
                              construction.

NO BASE MODEL IS LOADED. Matching is decoder geometry alone, so this costs a
download and a chunked matrix multiply, not a forward pass. That is why it can
run beside the main job instead of needing its own rental.
"""
import argparse
import io
import itertools
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from released_agreement import JumpReLU, match
from scope_matrix import SUITES as GRIDS

BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="2b", choices=list(GRIDS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--all_pairs", action="store_true",
                    help="every pair of the grid, not just the first two")
    a = ap.parse_args()
    cfg = GRIDS[a.suite]
    layer = a.layer if a.layer is not None else cfg["layer"]
    out = f"results/cosine_dist_{a.suite}.csv"
    done = pd.read_csv(out) if os.path.exists(out) else None
    seen = set(zip(done.a, done.b)) if done is not None else set()
    rows = done.to_dict("records") if done is not None else []

    pairs = list(itertools.combinations(cfg["grid"], 2))
    if not a.all_pairs:
        pairs = pairs[:3]
    print(f"{a.suite}: layer {layer}, {len(pairs)} pairs, decoders only",
          flush=True)

    for (na, pa), (nb, pb) in pairs:
        if (na, nb) in seen:
            print(f"  SKIP {na} vs {nb}", flush=True)
            continue
        A = JumpReLU(hf_hub_download(cfg["repo"], f"layer_{layer}/{pa}/params.npz"))
        B = JumpReLU(hf_hub_download(cfg["repo"], f"layer_{layer}/{pb}/params.npz"))
        _, best, mutual = match(A, B, (0.5,))
        # the distribution over MUTUAL nearest neighbours: those are the pairs
        # the band analysis draws from, so the marginal over all latents would
        # describe a population the bands never sample
        c = best[mutual]
        r = dict(suite=a.suite, a=na, b=nb, width_a=A.width, width_b=B.width,
                 d_model=A.W_dec.shape[1], n_latents=int(A.width),
                 n_mutual=int(mutual.sum()),
                 frac_mutual=float(mutual.mean()),
                 cos_median=float(np.median(c)),
                 cos_q25=float(np.percentile(c, 25)),
                 cos_q75=float(np.percentile(c, 75)),
                 cos_q05=float(np.percentile(c, 5)),
                 cos_q95=float(np.percentile(c, 95)))
        for lo, hi in BANDS:
            r["band_%.2f_%.2f" % (lo, min(hi, 1.0))] = \
                100.0 * float(((c >= lo) & (c < hi)).mean())
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print("  %-8s vs %-8s  mutual %6d (%4.1f%%)  cos med %.3f "
              "[q25 %.3f, q75 %.3f]"
              % (na, nb, r["n_mutual"], 100 * r["frac_mutual"],
                 r["cos_median"], r["cos_q25"], r["cos_q75"]), flush=True)
        del A, B
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    print(f"\nwrote {out}: {len(d)} pairs")
    print("  median of per-pair median cosine: %.3f" % d.cos_median.median())
    bandcols = [c for c in d.columns if c.startswith("band_")]
    print("  mean share of mutual pairs per band:")
    for c in bandcols:
        print("    %-16s %5.1f%%" % (c.replace("band_", ""), d[c].mean()))


if __name__ == "__main__":
    main()
