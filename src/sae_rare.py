# -*- coding: utf-8 -*-
"""Does the trained-vs-random SAE gap open on RARE features?

WHY THIS QUESTION. Sanity Checks for SAEs (2602.14111) reports trained SAEs
barely beating frozen-decoder baselines on aggregate causal editing (RAVEL 0.72
vs 0.73). In their SYNTHETIC setting they also report that SAEs "capture almost
exclusively the highest-frequency features, leaving over 90% of the ground-truth
dictionary, including the entire long tail of less frequent features, unmatched",
and that TopK recovery falls from 99.9% under constant activation probability to
7-43% under heavy tails -- which they state they cannot explain. But on REAL
models they excluded dead features and did NOT stratify by activation frequency.

So the aggregate null is dominated by frequent features. If the trained
dictionary's advantage lives in the tail, an aggregate mean hides it; if there is
no advantage anywhere, that is a much stronger negative than the paper claims.
Either way the frequency-stratified comparison is the missing measurement.

THE RANDOM BASELINE HERE IS STRICTLY HARSHER THAN THEIRS. Their "frozen decoder"
still trains the encoder; their best causal-editing baseline (soft-frozen) trains
the encoder AND lets the decoder move within cosine 0.8 of init. This baseline
trains nothing: random unit decoder directions, tied encoder, top-k gating with k
matched per token to the trained SAE's L0. A trained SAE failing to beat THIS is
a stronger result than failing to beat theirs; a trained SAE beating this is a
weaker result than beating theirs. Stated so the asymmetry is not read the wrong
way.

CAUSAL EFFECT is KL(p_clean || p_ablated) on the next-token distribution, where
ablation subtracts a_f * d_f from the layer-12 residual at the firing position
only. Reported raw AND per unit perturbation norm, because a dictionary whose
directions carry more magnitude would otherwise win trivially.
"""
import argparse
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b"
LAYER = 12                      # SAE is on resid_post of layer 12
SAE_PATH = ("layer_12/width_16k/average_l0_82/params.npz")
WIKI = (os.path.expanduser("~") + "/.cache/huggingface/hub/datasets--wikitext/"
        "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
        "wikitext-2-raw-v1/train-00000-of-00001.parquet")
SEQ_LEN = 256
DEV = "cuda"


def load_sae():
    p = hf_hub_download("google/gemma-scope-2b-pt-res", SAE_PATH)
    z = np.load(p)
    return {k: torch.tensor(z[k], dtype=torch.float32, device=DEV)
            for k in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold")}


def sae_encode(x, s):
    """Gemma Scope JumpReLU."""
    pre = x @ s["W_enc"] + s["b_enc"]
    return pre * (pre > s["threshold"])


def corpus(tok, n_seq):
    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        ids = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] >= SEQ_LEN:
            seqs.append(ids[:, :SEQ_LEN])
            buf = ""
            if len(seqs) >= n_seq:
                break
    return torch.cat(seqs, 0)


@torch.no_grad()
def resid_cache(model, ids, bs=8):
    outs = []
    for i in range(0, len(ids), bs):
        o = model(input_ids=ids[i:i + bs].to(DEV), output_hidden_states=True)
        outs.append(o.hidden_states[LAYER + 1].float().cpu())
    return torch.cat(outs, 0)                      # [S, T, d]


@torch.no_grad()
def kl_under_ablation(model, ids_batch, pos, vecs, bs=8):
    """KL(clean || ablated) at each (row, pos), subtracting vecs[row] at pos."""
    handle = {}

    def mk_hook(active):
        def hook(mod, inp, out):
            if not active["on"]:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            for r, (pp, vv) in enumerate(zip(active["pos"], active["vec"])):
                h[r, pp] = h[r, pp] - vv.to(h.dtype)
            return (h,) + out[1:] if isinstance(out, tuple) else h
        return hook

    active = {"on": False, "pos": None, "vec": None}
    handle = model.model.layers[LAYER].register_forward_hook(mk_hook(active))
    try:
        kls = []
        for i in range(0, len(ids_batch), bs):
            b = ids_batch[i:i + bs].to(DEV)
            p_ = pos[i:i + bs]
            v_ = vecs[i:i + bs]
            active["on"] = False
            lc = model(input_ids=b).logits.float()
            active.update(on=True, pos=p_, vec=v_)
            la = model(input_ids=b).logits.float()
            active["on"] = False
            for r, pp in enumerate(p_):
                lp = torch.log_softmax(lc[r, pp], -1)
                lq = torch.log_softmax(la[r, pp], -1)
                kls.append(float((lp.exp() * (lp - lq)).sum().item()))
        return kls
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=96)
    ap.add_argument("--per_bin", type=int, default=40)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--out", default="sae_rare.csv")
    a = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()
    s = load_sae()

    ids = corpus(tok, a.n_seq)
    print(f"corpus {tuple(ids.shape)}", flush=True)
    R = resid_cache(model, ids)                         # [S,T,d] cpu float32
    S, T, d = R.shape

    # --- sanity check: does the SAE actually reconstruct this hook point? ---
    # Position 0 is excluded throughout. The BOS residual carries a massive
    # activation the SAE does not reconstruct (cos 0.44 there against ~0.93
    # elsewhere) and its norm dominates the pooled variance: including it drives
    # explained variance to -3.5, excluding it gives 0.87. This is ordinary
    # preprocessing, not a finding.
    flat = R[:, 1:, :].reshape(-1, d)[:4096].to(DEV)
    acts = sae_encode(flat, s)
    rec = acts @ s["W_dec"] + s["b_dec"]
    ev = 1 - ((flat - rec).var() / flat.var()).item()
    l0 = (acts > 0).float().sum(-1).mean().item()
    print(f"SANITY  explained_variance={ev:.3f}  mean_L0={l0:.1f} "
          f"(expected ~0.8-0.9 and ~82; if not, the hook point is wrong)",
          flush=True)
    if ev < 0.5:
        raise SystemExit("SAE does not reconstruct this hook point - abort.")

    # --- feature activation frequency over the corpus ---
    freq = torch.zeros(s["W_enc"].shape[1], device=DEV)
    maxact = torch.zeros_like(freq)
    where = {}
    for i in range(0, S):
        A = sae_encode(R[i, 1:].to(DEV), s)             # [T-1,F], BOS dropped
        freq += (A > 0).float().sum(0)
        m, _ = A.max(0)
        maxact = torch.maximum(maxact, m)
    freq = (freq / (S * (T - 1))).cpu().numpy()
    print(f"features with freq>0: {(freq>0).sum()} / {len(freq)}", flush=True)

    # --- stratify live features by log frequency ---
    live = np.where(freq > 1e-5)[0]
    lf = np.log10(freq[live])
    qs = np.quantile(lf, np.linspace(0, 1, 7))
    rng = np.random.default_rng(0)
    picks = []
    for b in range(6):
        m = live[(lf >= qs[b]) & (lf <= qs[b + 1])]
        if len(m) == 0:
            continue
        sel = rng.choice(m, size=min(a.per_bin, len(m)), replace=False)
        picks += [(int(f), b, float(freq[f])) for f in sel]
    print(f"selected {len(picks)} trained features across 6 frequency bins",
          flush=True)

    # --- random dictionary: tied, unit-norm, top-k matched to trained L0 ---
    # Full width, not just the sampled columns: the top-k gate has to face the
    # same amount of competition the trained SAE's JumpReLU does, otherwise the
    # random features are selected against a far smaller pool and fire on
    # inputs a real dictionary would never route to them.
    K = int(round(l0))
    F_ = s["W_enc"].shape[1]
    Wr = torch.randn(d, F_, generator=torch.Generator(device=DEV)
                     .manual_seed(1), device=DEV)
    Wr = Wr / Wr.norm(dim=0, keepdim=True)
    rcols = rng.choice(F_, size=len(picks), replace=False)

    # --- precompute activations for the selected columns, once ---
    # Recomputing per feature was O(n_features * n_seq) full-width matmuls.
    fids = torch.tensor([p[0] for p in picks], device=DEV)
    Wt = s["W_enc"][:, fids]
    bt, tht = s["b_enc"][fids], s["threshold"][fids]
    Wrs = Wr[:, torch.tensor(rcols, device=DEV)]
    act_tr, act_rd = [], []
    for i in range(0, S, 4):
        Ri = R[i:i + 4, 1:].to(DEV).reshape(-1, d)
        pre = Ri @ Wt + bt
        act_tr.append((pre * (pre > tht)).cpu())
        full = Ri @ Wr
        thr = torch.topk(full, K, dim=-1).values[:, -1:]
        sel = Ri @ Wrs
        act_rd.append(torch.where(sel >= thr, sel,
                                  torch.zeros_like(sel)).cpu())
        del full
    ACT = {"trained": torch.cat(act_tr).reshape(S, T - 1, -1),
           "random": torch.cat(act_rd).reshape(S, T - 1, -1)}
    torch.cuda.empty_cache()
    print("precomputed activations", flush=True)

    rows = []
    for kind in ("trained", "random"):
        A_all = ACT[kind]
        for j, (fid, b, fr) in enumerate(picks):
            dvec = (s["W_dec"][fid] if kind == "trained"
                    else Wr[:, int(rcols[j])])
            col = A_all[:, :, j]                        # [S, T-1]
            flat = col.reshape(-1)
            n = min(a.n_pos, int((flat > 0).sum().item()))
            if n == 0:
                continue
            top = torch.topk(flat, n)
            best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                    for v, ix in zip(top.values, top.indices)]
            ib = torch.stack([ids[i] for _, i, _ in best])
            pos = [p for _, _, p in best]
            vecs = [float(v) * dvec for v, _, _ in best]
            nrm = [float((float(v) * dvec).norm().item()) for v, _, _ in best]
            kls = kl_under_ablation(model, ib, pos, vecs)
            for kl, nn in zip(kls, nrm):
                rows.append(dict(kind=kind, fid=fid, bin=b, freq=fr, kl=kl,
                                 pnorm=nn, kl_per_norm=kl / max(nn, 1e-6)))
            if (j + 1) % 40 == 0:
                print(f"  {kind} {j+1}/{len(picks)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(df)}")


if __name__ == "__main__":
    main()
