# -*- coding: utf-8 -*-
"""Apples-to-apples: lens the TOP-20 BY ALIGNMENT in BOTH dictionaries.

The confound/mechanism decision rule was built on a mismatched comparison:
Gemma Scope's 14119 was its top latent by CAUSAL MASS (alignment only 87th
percentile), while my trained arm's list was top-20 by ALIGNMENT. If Gemma
Scope's top-20 by alignment are also semantic, the rule does not track
dictionaries, 14119 is one unusual latent, and the Gemma Scope adjustment
(2.14 -> 1.42) needs the same scrutiny just applied to the frozen comparison.

OPERATIONAL CRITERION, replacing a judgement call. "Token-identity vs semantic"
is fuzzy -- 'governor'/'Governor'/'gouverneur' is a lexical cluster too. What
actually separates the examples is ORTHOGRAPHIC coherence of the top tokens:
  'k' 'ak' 'nk' 'ik' 'zk' 'ink'                    share characters
  'achievements' 'triumphs' 'glory' 'victories'    share meaning only
So: mean pairwise Jaccard over padded character bigrams+trigrams of the top-k
tokens. High = orthographic, low = semantic. Then the rule is computable:
adjust for alignment when high-alignment latents have high orthographic
coherence.

EFFICIENCY. align(d) = std_v(W_U d) = sqrt(d^T C d) with C = cov(W_U). Computing
C once (2304x2304) makes alignment free for all 16384 latents in both
dictionaries, instead of a 256000 x 16384 product per dictionary.
"""
import glob
import io
import itertools
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

torch.set_num_threads(8)
TOPK = 10


def load_embed():
    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    for fp in sorted(glob.glob(snap + "*.safetensors")):
        t = load_file(fp)
        k = next((k for k in t if "embed_tokens" in k), None)
        if k is not None:
            return t[k].float()
    raise SystemExit("embed_tokens not found")


def cov_of(E, chunk=16384):
    d = E.shape[1]
    s = torch.zeros(d, dtype=torch.float64)
    G = torch.zeros(d, d, dtype=torch.float64)
    n = 0
    for a in range(0, E.shape[0], chunk):
        B = E[a:a + chunk].double()
        s += B.sum(0)
        G += B.T @ B
        n += B.shape[0]
    mu = s / n
    return (G / n - torch.outer(mu, mu)).float()


def ngrams(tok_str, ns=(2, 3)):
    s = "^" + tok_str.strip().lower() + "$"
    out = set()
    for n in ns:
        for i in range(max(0, len(s) - n + 1)):
            out.add(s[i:i + n])
    return out


def ortho_coherence(tokens):
    gs = [ngrams(t) for t in tokens if t.strip()]
    gs = [g for g in gs if g]
    if len(gs) < 2:
        return float("nan")
    vals = []
    for a, b in itertools.combinations(gs, 2):
        u = len(a | b)
        vals.append(len(a & b) / u if u else 0.0)
    return float(np.mean(vals))


def main():
    E = load_embed()
    C = cov_of(E)
    En = E / E.norm(dim=-1, keepdim=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    dicts = {}
    z = np.load(hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        "layer_12/width_16k/average_l0_82/params.npz"))
    W = torch.tensor(z["W_dec"], dtype=torch.float32)
    dicts["gemma_scope"] = W / W.norm(dim=-1, keepdim=True)
    sd = torch.load("saes.pt", map_location="cpu")
    for arm in ("trained", "frozen"):
        w = sd[arm]["W_dec"].float()
        dicts[f"mine_{arm}"] = w / w.norm(dim=-1, keepdim=True)

    summary = {}
    for name, D in dicts.items():
        al = torch.sqrt(torch.clamp((D @ C * D).sum(-1), min=0)).numpy()
        top = np.argsort(-al)[:20]
        proj = En @ D[torch.tensor(top)].T          # [vocab, 20]
        print("=" * 96)
        print(f"{name}: TOP 20 BY ALIGNMENT (of all {D.shape[0]} latents)")
        print("=" * 96)
        cohs = []
        for c, fid in enumerate(top):
            idx = torch.topk(proj[:, c], TOPK).indices
            toks = [tok.decode([int(i)]) for i in idx]
            co = ortho_coherence(toks)
            cohs.append(co)
            print(f"  fid={int(fid):>6} ua={al[fid]:.5f} ortho={co:.3f} | "
                  + " ".join(repr(t) for t in toks[:8]))
        summary[name] = (float(np.nanmedian(cohs)), al)
        print()

    # 14119 in context, and its own coherence
    al_gs = summary["gemma_scope"][1]
    Dgs = dicts["gemma_scope"]
    pct = float((al_gs < al_gs[14119]).mean())
    idx = torch.topk(En @ Dgs[14119], TOPK).indices
    t14 = [tok.decode([int(i)]) for i in idx]
    print("=" * 96)
    print("REFERENCE: gemma_scope 14119 (top by CAUSAL MASS, the original exhibit)")
    print(f"  alignment={al_gs[14119]:.5f} -> {pct:.1%} percentile of all latents")
    print(f"  ortho={ortho_coherence(t14):.3f} | " + " ".join(repr(x) for x in t14))
    print()
    print("=" * 96)
    print("ORTHOGRAPHIC COHERENCE of top-20-by-alignment (median over latents)")
    print("=" * 96)
    for name, (m, _) in summary.items():
        print(f"  {name:>14}: {m:.3f}")
    print(f"  {'14119 alone':>14}: {ortho_coherence(t14):.3f}")
    # null: coherence of random token sets
    rng = np.random.default_rng(0)
    nulls = [ortho_coherence([tok.decode([int(i)])
                              for i in rng.integers(0, E.shape[0], TOPK)])
             for _ in range(200)]
    print(f"  {'random tokens':>14}: {np.nanmedian(nulls):.3f}  <- null level")


if __name__ == "__main__":
    main()
