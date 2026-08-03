# -*- coding: utf-8 -*-
"""Train matched TRAINED and SOFT-FROZEN SAEs on Gemma-2-2B layer 12.

WHY THIS EXISTS. The Sanity Checks paper's headline causal-editing result (0.73
frozen vs 0.72 trained) is against SOFT-FROZEN: decoder randomly initialised and
constrained to stay within cosine 0.8 of its init, encoder trained normally. My
earlier run used an untrained tied random dictionary, which is a harsher control
but a different one -- refuting a claim nobody made. This trains their baseline.

MATCHED BY CONSTRUCTION. Both SAEs see the identical activation stream in the
identical order, same width, same k, same optimiser, same seed. Neither matches
Gemma Scope's training budget; that is fine and stated, because the comparison
that matters is trained-vs-soft-frozen at equal budget, not either against
Gemma Scope.

TopK rather than JumpReLU: L0 is then exactly k by construction, so the sparsity
match between the two arms is exact rather than approximate, and there is no
straight-through estimator to muddy a short run. The paper evaluated TopK too.

SOFT-FREEZE PROJECTION. After each step, any decoder row whose cosine to its
init has fallen below tau is rotated back to exactly tau in the plane spanned by
(row, init), preserving its norm:  w' = ||w|| * (tau*w0 + sqrt(1-tau^2)*u),
where u is the unit component of w orthogonal to w0.
"""
import argparse
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
LAYER, DEV = 12, "cuda"
D_MODEL, WIDTH, K = 2304, 16384, 82
TAU = 0.8
SEQ = 512


class TopKSAE(torch.nn.Module):
    def __init__(self, d, m, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        W = torch.randn(m, d, generator=g)
        W = W / W.norm(dim=-1, keepdim=True)
        self.W_dec = torch.nn.Parameter(W.clone())
        self.W_enc = torch.nn.Parameter(W.t().clone().contiguous())
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.k = k
        self.register_buffer("W0", W.clone())

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        v, i = torch.topk(pre, self.k, dim=-1)
        out = torch.zeros_like(pre)
        return out.scatter_(-1, i, F.relu(v))

    def forward(self, x):
        a = self.encode(x)
        return a @ self.W_dec + self.b_dec, a

    @torch.no_grad()
    def renorm(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def soft_freeze(self, tau=TAU):
        w, w0 = self.W_dec.data, self.W0
        nrm = w.norm(dim=-1, keepdim=True)
        wn = w / nrm
        cos = (wn * w0).sum(-1, keepdim=True)
        bad = (cos < tau).squeeze(-1)
        if not bad.any():
            return
        perp = wn[bad] - cos[bad] * w0[bad]
        pn = perp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u = perp / pn
        new = tau * w0[bad] + np.sqrt(1 - tau ** 2) * u
        self.W_dec.data[bad] = new * nrm[bad]


def stream(model, tok, texts, n_tokens, bs=6):
    """Yield layer-12 residuals, BOS dropped (see sae_rare.py for why)."""
    buf, seen = "", 0
    batch = []
    for t in texts:
        buf += " " + t
        ids = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] < SEQ:
            continue
        batch.append(ids[:, :SEQ])
        buf = ""
        if len(batch) == bs:
            b = torch.cat(batch, 0).to(DEV)
            batch = []
            with torch.no_grad():
                h = model(input_ids=b,
                          output_hidden_states=True).hidden_states[LAYER + 1]
            x = h[:, 1:, :].reshape(-1, D_MODEL).float()
            seen += x.shape[0]
            yield x
            if seen >= n_tokens:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=20_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="data/saes.pt")
    # Arms are trained SEQUENTIALLY by default now. Training both at once put
    # peak GPU memory at 15.96/16.38 GB, which on WDDM spills to host RAM and
    # collapsed throughput (100% reported utilisation at 41 C, no progress for
    # 30 minutes). The activation stream is deterministic -- fixed corpus order,
    # no shuffling -- so separate runs still see identical activations, which is
    # what the matched comparison requires.
    ap.add_argument("--arm", default="both",
                    choices=["both", "trained", "frozen"])
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()

    paths = [hf_hub_download("wikitext", f, repo_type="dataset")
             for f in ("wikitext-103-raw-v1/train-00000-of-00002.parquet",
                       "wikitext-103-raw-v1/train-00001-of-00002.parquet")]
    texts = []
    for p in paths:
        texts += [str(t).strip() for t in pd.read_parquet(p)["text"].tolist()
                  if len(str(t).strip()) > 200
                  and not str(t).strip().startswith("=")]
    print(f"corpus paragraphs: {len(texts)}", flush=True)

    want = ("trained", "frozen") if a.arm == "both" else (a.arm,)
    arms = {n: TopKSAE(D_MODEL, WIDTH, K, 0).to(DEV) for n in want}
    opts = {k: torch.optim.Adam(v.parameters(), lr=a.lr)
            for k, v in arms.items()}
    fired = {k: torch.zeros(WIDTH, device=DEV) for k in arms}

    # b_dec <- mean activation, standard init; estimated from the first chunk
    step, seen, ema = 0, 0, {k: None for k in arms}
    for x in stream(model, tok, texts, a.tokens):
        if step == 0:
            for m in arms.values():
                m.b_dec.data = x.mean(0)
        for name, m in arms.items():
            rec, acts = m(x)
            loss = (rec - x).pow(2).sum(-1).mean()
            opts[name].zero_grad(set_to_none=True)
            loss.backward()
            opts[name].step()
            m.renorm()
            if name == "frozen":
                m.soft_freeze()
            fired[name] += (acts > 0).float().sum(0)
            e = float(loss.item())
            ema[name] = e if ema[name] is None else 0.99 * ema[name] + 0.01 * e
        seen += x.shape[0]
        step += 1
        if step % 200 == 0:
            with torch.no_grad():
                v = x.var(0).sum().item()
                msg = "  ".join(
                    f"{n}: mse={ema[n]:.1f} ev={1-ema[n]/v:.3f} "
                    f"dead={(fired[n]==0).float().mean():.1%}" for n in arms)
            print(f"{seen/1e6:5.1f}M tok | {msg}", flush=True)

    with torch.no_grad():
        for n, m in arms.items():
            c = F.cosine_similarity(m.W_dec.data, m.W0, dim=-1)
            note = f"constraint tau={TAU}" if n == "frozen" else "unconstrained"
            print(f"\n{n} decoder cos-to-init: min={c.min():.3f} "
                  f"mean={c.mean():.3f} median={c.median():.3f}  ({note})")

    torch.save({n: m.state_dict() for n, m in arms.items()}, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
