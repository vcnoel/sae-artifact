# -*- coding: utf-8 -*-
"""Candidate C, Test 1: metric drift under semantically-null reformatting.

For every pair (A, A') the same panel is computed on both members and the
within-pair relative drift |m(A') - m(A)| / |m(A)| recorded.

DESIGN POINTS THAT DECIDE THE ANSWER

fp32, not bf16. The quantity being measured is a small relative change. In bf16
the arithmetic noise floor is itself percents, which would be indistinguishable
from the effect. Everything runs in float32 with a fixed seed, and an identity
control (the same text measured twice) pins the numerical floor.

Raw AND ceiling-normalised. This is the adversarial split. Valentin's
Proposition 1 already predicts that any descriptor with an N-dependent range
moves when N moves; showing that again is not a new paper, it is his own
published result. So every descriptor with an analytic ceiling is reported both
raw and divided by that ceiling, and the scale-free descriptors (HFER, Fiedler,
lambda_max, sink concentration) are reported alongside. C survives only if the
normalised and scale-free descriptors move too.

BOS. Special tokens are dropped before graph construction, as in the sweep. Sink
concentration is the exception by definition -- it is the attention mass on
position 0 -- so it is read off the full attention before the drop.

GSP operators come from the installed spectral-trust package at its canonical
config (uniform / symmetric / rw, hfer_cutoff 0.1, remove_self_loops False).
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import spectral_trust as st  # noqa: E402

MODEL = "meta-llama/Llama-3.2-1B"


def eff_rank(H):
    s = np.linalg.svd(H, compute_uv=False)
    p = s / (s.sum() + 1e-30)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def panel(text, model, tok, gc, sa, device):
    enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        out = model(**enc, output_attentions=True, output_hidden_states=True)

    atts = out.attentions                      # L x [1, H, N, N]
    hids = out.hidden_states[1:]               # L x [1, N, d]
    ids = enc["input_ids"][0]
    special = set(tok.all_special_ids)
    keep = torch.tensor([i for i, t in enumerate(ids.tolist())
                         if t not in special], device=device)

    rows = []
    for li, (A, H) in enumerate(zip(atts, hids)):
        A0 = A[0].to(torch.float32)                       # [H, N, N]
        # sink concentration: mass on position 0, before the special-token drop
        sink = float(A0[:, :, 0].mean().item())

        Ak = A0[:, keep][:, :, keep]
        Hk = H[0].to(torch.float32)[keep]
        N = Ak.shape[-1]
        if N < 8:
            continue

        # --- his operators, his config ---
        Auni = gc.aggregate_heads(Ak.unsqueeze(0))
        Asym = gc.symmetrize_attention(Auni)
        L = gc.construct_laplacian(Asym)
        d = sa.analyze_layer(Hk, L, li)

        # --- plain attention statistics (no Laplacian involved) ---
        P = Ak.mean(0)
        P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
        ent = float(-(P * (P + 1e-30).log()).sum(-1).mean().item())
        idx = torch.arange(N, device=P.device, dtype=torch.float32)
        dist = float((P * (idx[:, None] - idx[None, :]).abs()).sum(-1)
                     .mean().item())
        er = eff_rank(Hk.cpu().numpy())

        rows.append(dict(
            layer=li, N=N,
            attn_entropy=ent, attn_entropy_norm=ent / np.log(N),
            attn_distance=dist, attn_distance_norm=dist / N,
            eff_rank=er, eff_rank_norm=er / N,
            sink_conc=sink,
            hfer=float(d.hfer), fiedler=float(d.fiedler_value),
            lambda_max=float(np.max(d.eigenvalues)),
            smoothness=float(d.smoothness_index),
            spec_entropy=float(d.spectral_entropy),
            spec_entropy_norm=float(d.spectral_entropy) / np.log(N),
            energy=float(d.energy), energy_norm=float(d.energy) / N,
        ))
    t = pd.DataFrame(rows)
    return t.drop(columns=["layer"]).mean().to_dict(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="panel.csv")
    a = ap.parse_args()

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = json.load(open("pairs.json", encoding="utf-8"))
    if a.limit:
        pairs = pairs[:a.limit]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager").to(dev).eval()

    cfg = st.GSPConfig(model_name=MODEL, device=dev, eigen_solver="dense",
                       verbose=False, save_plots=False,
                       save_intermediate=False)
    gc_, sa = st.GraphConstructor(cfg), st.SpectralAnalyzer(cfg)

    recs = []
    for i, p in enumerate(pairs):
        for side in ("a", "b"):
            m, _ = panel(p[side], model, tok, gc_, sa, dev)
            recs.append(dict(pair=i, side=side, stratum=p["stratum"],
                             delta=p["delta"], n_tok=p[f"n_{side}"], **m))
        if i == 0:
            # identity control: the same text twice, to pin the numerical floor
            m2, _ = panel(pairs[0]["a"], model, tok, gc_, sa, dev)
            recs.append(dict(pair=-1, side="identity", stratum="control",
                             delta=0.0, n_tok=pairs[0]["n_a"], **m2))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(pairs)} pairs", flush=True)

    df = pd.DataFrame(recs)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  rows={len(df)}")


if __name__ == "__main__":
    main()
