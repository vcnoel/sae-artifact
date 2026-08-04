# -*- coding: utf-8 -*-
"""Quantify the population dependence of the alignment confound. CPU only.

DISCRIMINATOR. n-gram overlap missed the punctuation class ('\\n', '.', ' и'
share nothing yet are equally non-semantic). The property that unites the junk is
NOT BEING WORDS. So: wordlike(token) = stripped token is alphabetic and >= 3
characters. Scores ~1 for 'governor' 'agreements' 'achievements', ~0 for
'\\n' '.' 'Z' 'ch' 'ul', and ~0 for 'k' 'ak' 'nk' -- which correctly folds 14119
in with the rest rather than treating it as a special case. A dictionary wordlist
would refine this; the length+alpha rule is enough for the class boundary the two
exhibits sit either side of.

THE MEASUREMENT. Wordlike fraction of high-alignment latents in two populations
per dictionary:
  (a) all live latents            -- extreme-alignment latents live here
  (b) the frequency-stratified 240 the adjustments were actually fit on
If (a) is junk and (b) is words, the alignment confound's magnitude is a function
of the sampling scheme, which makes any alignment-adjusted causal number
uninterpretable without one. That is artifact 4 refined, not weakened.
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
from transformers import AutoTokenizer

torch.set_num_threads(8)
TOPK = 10


def wordlike(s):
    t = s.strip()
    return 1.0 if (len(t) >= 3 and t.isalpha()) else 0.0


def main():
    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    E = None
    for fp in sorted(glob.glob(snap + "*.safetensors")):
        t = load_file(fp)
        k = next((k for k in t if "embed_tokens" in k), None)
        if k is not None:
            E = t[k].float()
            break
    # C = cov(W_U) so alignment is cheap for every latent
    d = E.shape[1]
    s1 = torch.zeros(d, dtype=torch.float64)
    G = torch.zeros(d, d, dtype=torch.float64)
    n = 0
    for a in range(0, E.shape[0], 16384):
        B = E[a:a + 16384].double()
        s1 += B.sum(0)
        G += B.T @ B
        n += B.shape[0]
    mu = s1 / n
    C = (G / n - torch.outer(mu, mu)).float()
    En = E / E.norm(dim=-1, keepdim=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    dicts = {}
    z = np.load(hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        "layer_12/width_16k/average_l0_82/params.npz"))
    W = torch.tensor(z["W_dec"], dtype=torch.float32)
    dicts["gemma_scope"] = W / W.norm(dim=-1, keepdim=True)
    sd = torch.load("data/saes.pt", map_location="cpu")
    w = sd["trained"]["W_dec"].float()
    dicts["mine_trained"] = w / w.norm(dim=-1, keepdim=True)

    # the populations the adjustments were fit on
    samp = {}
    gs = pd.read_csv("results/sae_rare.csv")
    samp["gemma_scope"] = sorted(gs[gs.kind == "trained"].fid.unique())
    ev = pd.read_csv("results/eval_saes.csv")
    samp["mine_trained"] = sorted(ev[ev.arm == "trained"].fid.unique())

    def wl_of(D, fids, al, k=20):
        fids = np.asarray(fids)
        order = fids[np.argsort(-al[fids])][:k]
        proj = En @ D[torch.tensor(order)].T
        out = []
        for c in range(len(order)):
            idx = torch.topk(proj[:, c], TOPK).indices
            toks = [tok.decode([int(i)]) for i in idx]
            out.append(np.mean([wordlike(x) for x in toks]))
        return np.array(out), order

    print(f"{'dictionary':>14} {'population':>26} {'n':>7} "
          f"{'wordlike frac of top-20 by alignment':>38}")
    print("-" * 92)
    rows = []
    for name, D in dicts.items():
        al = torch.sqrt(torch.clamp((D @ C * D).sum(-1), min=0)).numpy()
        allfids = np.arange(D.shape[0])
        for pop, fids in (("all latents", allfids),
                          ("frequency-stratified 240", samp[name])):
            wl, order = wl_of(D, fids, al)
            print(f"{name:>14} {pop:>26} {len(fids):>7} "
                  f"{'median=%.2f  mean=%.2f' % (np.median(wl), wl.mean()):>38}")
            rows.append((name, pop, float(np.median(wl)), float(wl.mean()),
                         float(al[order].mean())))
        # where do the stratified 240 sit in the dictionary-wide alignment dist?
        pct = np.mean(al[None, :] < al[samp[name]][:, None], axis=1)
        print(f"{'':>14} {'-> alignment percentile of the 240':>26} "
              f"{'':>7} median={np.median(pct):.1%} max={pct.max():.1%}")

    print()
    print("INTERPRETATION")
    for name, pop, med, mean, ala in rows:
        print(f"  {name:>14} / {pop:<26} wordlike med={med:.2f} "
              f"mean alignment of that top-20={ala:.5f}")
    print("\n  If 'all latents' is near 0 and the stratified 240 is high, the")
    print("  alignment confound's size depends on the sampling scheme, and an")
    print("  alignment-adjusted causal number is uninterpretable without one.")


if __name__ == "__main__":
    main()
