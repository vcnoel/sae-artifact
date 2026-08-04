# -*- coding: utf-8 -*-
"""Reproduce Cho et al. (2607.20596) single-token detection at Gemma-2-2B L12.

Their Table 14 reports, for our exact model / layer / family / width
(Gemma-2-2B, L12, GemmaScope JumpReLU, 16k): nST = 158, %ST = 0.96.

Three criteria over the top-k=20 activating tokens of each latent:
  gap    = (v1 - v2) / v1  >= 0.30      v1 >= v2 the top two activation values
  purity = max_t |{i in [k]: t_i = t}| / k >= 0.60   after case normalisation
  word   = the dominant token carries a word-boundary prefix
           (U+2581 for SentencePiece, which Gemma uses)

ONE CELL ONLY. This is external validation that their method is implemented
correctly, not a second population: their own Table 18 gives nST per layer for
Gemma-2-2B ranging 13-157 with median 55, so cells that thin cannot support a
variance decomposition. Reproducing the prevalence lets the paper decline the ST
arm on a measurement rather than on their numbers alone.
"""
import argparse
import io
import sys
from collections import Counter

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, LAYER, DEV = "google/gemma-2-2b", 12, "cuda"
SAE = "layer_12/width_16k/average_l0_82/params.npz"
WIKI = ("C:/Users/valno/.cache/huggingface/hub/datasets--wikitext/snapshots/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
        "train-00000-of-00001.parquet")
SEQ, TOPK = 256, 20
GAP, PURITY = 0.30, 0.60
SP_MARK = "\u2581"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=128)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()
    z = np.load(hf_hub_download("google/gemma-scope-2b-pt-res", SAE))
    W_enc = torch.tensor(z["W_enc"], dtype=torch.float16, device=DEV)
    b_enc = torch.tensor(z["b_enc"], dtype=torch.float16, device=DEV)
    thr = torch.tensor(z["threshold"], dtype=torch.float16, device=DEV)
    F = W_enc.shape[1]

    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        i = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if i.shape[1] >= SEQ:
            seqs.append(i[:, :SEQ])
            buf = ""
            if len(seqs) >= a.n_seq:
                break
    ids = torch.cat(seqs, 0)
    S, T = ids.shape
    print(f"corpus {S}x{T} = {S*(T-1)} scored tokens (BOS dropped)", flush=True)

    # running top-k per latent over the whole corpus
    best_v = torch.full((F, TOPK), -1e30, device=DEV, dtype=torch.float32)
    best_t = torch.zeros((F, TOPK), device=DEV, dtype=torch.long)
    with torch.no_grad():
        for i in range(0, S, 4):
            b = ids[i:i + 4].to(DEV)
            h = model(input_ids=b, output_hidden_states=True) \
                .hidden_states[LAYER + 1][:, 1:, :]
            toks = b[:, 1:].reshape(-1)
            x = h.reshape(-1, h.shape[-1]).half()
            pre = x @ W_enc + b_enc
            A = (pre * (pre > thr)).float().T          # [F, n_tok]
            cat_v = torch.cat([best_v, A], dim=1)
            cat_t = torch.cat([best_t, toks.expand(F, -1)], dim=1)
            v, idx = torch.topk(cat_v, TOPK, dim=1)
            best_v = v
            best_t = torch.gather(cat_t, 1, idx)
            del A, cat_v, cat_t, pre, x, h
        torch.cuda.empty_cache()

    bv = best_v.cpu().numpy()
    bt = best_t.cpu().numpy()
    # criterion 1: gap on the top two activation values
    gap = np.where(bv[:, 0] > 0, (bv[:, 0] - bv[:, 1]) / np.maximum(bv[:, 0], 1e-9), 0.0)
    alive = bv[:, 0] > 0
    # criteria 2-3 need surface strings
    vocab_ids = np.unique(bt[alive])
    surf = {int(v): tok.convert_ids_to_tokens(int(v)) for v in vocab_ids}
    purity = np.zeros(F)
    word = np.zeros(F, dtype=bool)
    for f in np.where(alive)[0]:
        row = bt[f]
        strs = [surf.get(int(x), "") for x in row]
        norm = [s.lstrip(SP_MARK).lower() for s in strs]
        c = Counter(norm)
        top, cnt = c.most_common(1)[0]
        purity[f] = cnt / TOPK
        # the dominant token must carry a word-boundary prefix
        word[f] = any(s.startswith(SP_MARK) and s.lstrip(SP_MARK).lower() == top
                      for s in strs)

    st = alive & (gap >= GAP) & (purity >= PURITY) & word
    print()
    print(f"{'criterion':<34}{'count':>8}{'% of F':>9}")
    for lab, m in (("alive (any activation)", alive),
                   (f"gap >= {GAP}", alive & (gap >= GAP)),
                   (f"purity >= {PURITY}", alive & (purity >= PURITY)),
                   ("complete word", alive & word),
                   ("ALL THREE (nST)", st)):
        print(f"{lab:<34}{int(m.sum()):>8}{100*m.sum()/F:>8.2f}%")
    print(f"\n  their Table 14 (Gemma-2-2B, L12, GemmaScope 16k): nST=158, %ST=0.96")
    print(f"  ours: nST={int(st.sum())}, %ST={100*st.sum()/F:.2f}%  "
          f"ratio={st.sum()/158:.2f}x")
    pd.DataFrame(dict(fid=np.arange(F), gap=gap, purity=purity, word=word,
                      alive=alive, st=st)).to_csv(
        "results/st_detect_L12.csv", index=False)
    print("  wrote results/st_detect_L12.csv")


if __name__ == "__main__":
    main()
