# -*- coding: utf-8 -*-
"""Bad-control check: are the trained arm's HIGH-ALIGNMENT latents trivial?

The HL=0.992 result adjusts away unembedding alignment, and here the covariate is
nearly the treatment: the only difference between arms IS decoder freedom, and
what a free decoder does with it is partly to increase alignment. Two readings:

  CONFOUND  the metric rewards output-adjacent directions and trained decoders
            drift toward them -> adjusting is correct, headline stands
  MECHANISM finding directions the model uses to write output is what decoder
            training is FOR -> adjusting removes the treatment, different paper

Against Gemma Scope, feature 14119 (an orthographic 'k' latent) was the argument
for CONFOUND. There is no such exhibit for my own trained arm. So: logit-lens the
top latents by alignment. Orthographic/token-identity -> triviality argument
transfers. Semantic -> alignment is doing real work and the headline is wrong.

Scoped to the 240 latents the adjustment was fit on, since the result rests on
those. CPU only.
"""
import glob
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

torch.set_num_threads(8)


def main():
    df = pd.read_csv("results/eval_saes.csv")
    f = (df[df.arm == "trained"].groupby("fid")
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    sd = torch.load("data/saes.pt", map_location="cpu")
    W = sd["trained"]["W_dec"].float()
    W = W / W.norm(dim=-1, keepdim=True)
    ids = torch.tensor(f.fid.values, dtype=torch.long)
    D = W[ids].T.contiguous()

    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    E = None
    for fp in sorted(glob.glob(snap + "*.safetensors")):
        t = load_file(fp)
        k = next((k for k in t if "embed_tokens" in k), None)
        if k is not None:
            E = t[k].float()
            break
    En = E / E.norm(dim=-1, keepdim=True)
    # alignment = spread of the direct logit contribution
    V = E @ D                                   # [vocab, 240]
    f["ua"] = V.std(0).numpy()
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    proj = En @ D                               # [vocab, 240] cosine-ish

    def lens(j, n=10):
        v = proj[:, j]
        top = torch.topk(v, n).indices
        return " ".join(repr(tok.decode([int(i)])) for i in top)

    for key, lab in (("ua", "ALIGNMENT"), ("kpn", "CAUSAL MASS")):
        order = f.sort_values(key, ascending=False).head(20)
        print("=" * 92)
        print(f"TOP 20 TRAINED LATENTS BY {lab}")
        print("=" * 92)
        for _, row in order.iterrows():
            j = int(f.index[f.fid == row.fid][0])
            print(f"  fid={int(row.fid):>5} ua={row['ua']:.5f} "
                  f"kpn={row['kpn']:.6f} | {lens(j)}")
        print()

    r = f[["ua", "kpn"]].corr(method="spearman").iloc[0, 1]
    print(f"Spearman(alignment, causal mass) over these 240 = {r:+.3f}")
    print(f"alignment: med={f['ua'].median():.5f} "
          f"p90={f['ua'].quantile(.9):.5f} max={f['ua'].max():.5f}")


if __name__ == "__main__":
    main()
