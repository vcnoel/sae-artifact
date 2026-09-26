# -*- coding: utf-8 -*-
"""Train one SAE "arm" -- one defensible fitting choice -- sharing seed 0.

WHY. The crossed decomposition had TWO arms, so the arm component carried 1
degree of freedom and Erho^2 = 0.455 was an extrapolation from a single
difference. A G-theory reviewer discounts that on sight, and the reliability
paper's 36 wrappers is why its 0.018 lands. Six arms give the arm and
latent x arm components real degrees of freedom.

EVERY ARM SHARES SEED 0, so W0 is identical and latent i denotes the same
initial direction in all of them. That is what makes the design crossed rather
than nested, and it is the whole reason latent correspondence is exact.

The arms vary things a practitioner actually varies: how tightly the decoder is
constrained, the learning rate, the sparsity k, and the order the data arrives
in. None is a straw man; each is a choice someone would defend.
"""
import argparse
import io
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_saes import DEV, TopKSAE
import model_configs

SEED = 0                      # shared by every arm, non-negotiable


def stream(model, tok, texts, n_tokens, seq, layer, d_model, bs=6):
    buf, seen, batch = "", 0, []
    for t in texts:
        buf += " " + t
        ids = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] < seq:
            continue
        batch.append(ids[:, :seq])
        buf = ""
        if len(batch) == bs:
            b = torch.cat(batch, 0).to(DEV)
            batch = []
            with torch.no_grad():
                h = model(input_ids=b,
                          output_hidden_states=True).hidden_states[layer + 1]
            x = h[:, 1:, :].reshape(-1, d_model).float()
            seen += x.shape[0]
            yield x
            if seen >= n_tokens:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokens", type=int, default=12_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k", type=int, default=82)
    ap.add_argument("--tau", type=float, default=0.0,
                    help="soft-freeze target cosine to init; 0 = decoder free")
    ap.add_argument("--order", type=int, default=0,
                    help="corpus permutation seed; 0 = as-loaded")
    ap.add_argument("--base", default="gemma2-2b",
                    choices=list(model_configs.MODELS),
                    help="which base model/layer/width this arm is fit on")
    a = ap.parse_args()
    cfg = model_configs.get(a.base)
    os.makedirs(cfg["dir"], exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to(DEV).eval()

    paths = [hf_hub_download("wikitext", f, repo_type="dataset")
             for f in ("wikitext-103-raw-v1/train-00000-of-00002.parquet",
                       "wikitext-103-raw-v1/train-00001-of-00002.parquet")]
    texts = []
    for p in paths:
        texts += [str(t).strip() for t in pd.read_parquet(p)["text"].tolist()
                  if len(str(t).strip()) > 200
                  and not str(t).strip().startswith("=")]
    if a.order:
        np.random.default_rng(a.order).shuffle(texts)
    print(f"arm={a.tag} base={a.base} lr={a.lr} k={a.k} tau={a.tau} "
          f"order={a.order} paragraphs={len(texts)}", flush=True)

    m = TopKSAE(cfg["d_model"], cfg["width"], a.k, SEED).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    fired = torch.zeros(cfg["width"], device=DEV)
    step, seen, ema = 0, 0, None
    for x in stream(model, tok, texts, a.tokens,
                    cfg["seq"], cfg["layer"], cfg["d_model"]):
        if step == 0:
            m.b_dec.data = x.mean(0)
        rec, acts = m(x)
        loss = (rec - x).pow(2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        m.renorm()
        if a.tau > 0:
            m.soft_freeze(a.tau)
        fired += (acts > 0).float().sum(0)
        e = float(loss.item())
        ema = e if ema is None else 0.99 * ema + 0.01 * e
        seen += x.shape[0]
        step += 1
        if step % 300 == 0:
            with torch.no_grad():
                v = x.var(0).sum().item()
            print(f"  {seen/1e6:5.1f}M mse={ema:.1f} ev={1-ema/v:.3f} "
                  f"dead={(fired==0).float().mean():.1%}", flush=True)

    with torch.no_grad():
        c = F.cosine_similarity(m.W_dec.data, m.W0, dim=-1)
    print(f"  final cos-to-init: median={c.median():.3f} min={c.min():.3f}")
    out = f"{cfg['dir']}/arm_{a.tag}.pt"
    torch.save({"state": m.state_dict(), "k": a.k, "tag": a.tag,
                "lr": a.lr, "tau": a.tau, "order": a.order, "base": a.base,
                "d_model": cfg["d_model"], "width": cfg["width"]}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
