# -*- coding: utf-8 -*-
"""Is a dominant latent generic across SAEs, or a fact about one dictionary?

REALLOCATED BUDGET. 2040 features from one dictionary cannot tell us whether the
single dominant latent found there is generic. 300 features from each of nine
Gemma Scope dictionaries can: layers 12 and 19, widths 16k/65k/131k, L0 from 22
to 445.

WHY max/median WITHIN A FIXED-SIZE SAMPLE. Finding each dictionary's global top-1
would need all 16k-131k features measured. Instead the statistic is the
max/median ratio within a random sample of fixed size n=300, which IS comparable
across dictionaries and is the same shape of statistic the Chronos paper reports
(max-to-median 30.5x). "Every SAE we examined shows max/median ~10x at n=300" is
a phenomenon; it does not require identifying the true argmax.

THE DIRECT-PATH CONTROL, and it is nearly free. The dominant latent on the
primary dictionary turned out to be an orthographic 'k' feature -- a
token-identity feature that writes almost straight to the unembedding. Next-token
KL therefore rewards direct-path features rather than computationally important
ones. Since the full logits are already computed, KL is read at the firing
position t AND at t+1 and t+3. The t+3 number has no direct contribution from the
ablated token's own identity, so a tail that exists at t and vanishes at t+3 is a
direct-path artifact rather than a causal phenomenon.

Per-arm random-dictionary controls are kept only for the anchor dictionary; the
cross-dictionary question is about within-dictionary tail shape.
"""
import argparse
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, DEV = "google/gemma-2-2b", "cuda"
WIKI = ("C:/Users/valno/.cache/huggingface/hub/datasets--wikitext/snapshots/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
        "train-00000-of-00001.parquet")
SEQ = 256
GRID = [(12, "16k", 22), (12, "16k", 82), (12, "16k", 445),
        (12, "65k", 72), (12, "131k", 67),
        (19, "16k", 23), (19, "16k", 73), (19, "16k", 279),
        (19, "65k", 63)]
OFFSETS = (0, 1, 3)


def load_sae(layer, width, l0):
    p = hf_hub_download(
        "google/gemma-scope-2b-pt-res",
        f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz")
    z = np.load(p)
    return {k: torch.tensor(z[k], dtype=torch.float16, device=DEV)
            for k in ("W_enc", "W_dec", "b_enc", "threshold")}


def encode(x, s):
    pre = x.half() @ s["W_enc"] + s["b_enc"]
    return pre * (pre > s["threshold"])


@torch.no_grad()
def kl_multi(model, layer, ids, pos, vecs, bs=6):
    """Returns dict offset -> list of KL values."""
    st = {"on": False, "pos": None, "vec": None}

    def hook(mod, inp, out):
        if not st["on"]:
            return out
        h = (out[0] if isinstance(out, tuple) else out).clone()
        for r, (pp, vv) in enumerate(zip(st["pos"], st["vec"])):
            h[r, pp] = h[r, pp] - vv.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    hd = model.model.layers[layer].register_forward_hook(hook)
    try:
        # MAGNITUDE and RATE readouts from the same forward passes.
        # The analytic transfer argument (alignment scales the response along
        # d_f identically for ablation / interchange / clamping) holds for
        # MAGNITUDE readouts. RATE readouts are threshold crossings and
        # saturate: once the argmax flips, further alignment buys nothing. So
        # alignment may predict KL strongly and flip weakly, which would
        # support "prefer rate readouts for causal claims about latents".
        res = {o: [] for o in OFFSETS}
        flip = {o: [] for o in OFFSETS}
        dtop = {o: [] for o in OFFSETS}
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs].to(DEV)
            p_, v_ = pos[i:i + bs], vecs[i:i + bs]
            st["on"] = False
            lc = model(input_ids=b).logits.float()
            st.update(on=True, pos=p_, vec=v_)
            la = model(input_ids=b).logits.float()
            st["on"] = False
            for r, pp in enumerate(p_):
                for o in OFFSETS:
                    q = min(pp + o, lc.shape[1] - 1)
                    lp = torch.log_softmax(lc[r, q], -1)
                    lq = torch.log_softmax(la[r, q], -1)
                    res[o].append(float((lp.exp() * (lp - lq)).sum().item()))
                    t0 = int(lp.argmax())
                    flip[o].append(int(t0 != int(lq.argmax())))
                    dtop[o].append(float((lp[t0].exp() - lq[t0].exp()).item()))
        return res, flip, dtop
    finally:
        hd.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--n_feat", type=int, default=300)
    ap.add_argument("--n_pos", type=int, default=4)
    ap.add_argument("--out", default="multi_dict.csv")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()

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
    ids_all = torch.cat(seqs, 0)
    S, T = ids_all.shape
    print(f"corpus {S}x{T}", flush=True)

    rows = []
    for (layer, width, l0) in GRID:
        tag = f"L{layer}_{width}_l0{l0}"
        with torch.no_grad():
            R = []
            for i in range(0, S, 8):
                R.append(model(input_ids=ids_all[i:i + 8].to(DEV),
                               output_hidden_states=True)
                         .hidden_states[layer + 1][:, 1:, :].cpu())
            R = torch.cat(R)
        s = load_sae(layer, width, l0)
        F_ = s["W_enc"].shape[1]
        with torch.no_grad():
            freq = torch.zeros(F_, device=DEV)
            for i in range(0, S, 4):
                A = encode(R[i:i + 4].to(DEV).reshape(-1, R.shape[-1]), s)
                freq += (A > 0).float().sum(0)
            freq = (freq / (S * (T - 1))).float().cpu().numpy()
        live = np.where(freq > 1e-5)[0]
        rng = np.random.default_rng(0)
        pick = rng.choice(live, size=min(a.n_feat, len(live)), replace=False)
        print(f"{tag}: F={F_} live={len(live)} sampled={len(pick)}", flush=True)

        with torch.no_grad():
            Wsub = s["W_enc"][:, torch.tensor(pick, device=DEV)]
            bsub = s["b_enc"][torch.tensor(pick, device=DEV)]
            tsub = s["threshold"][torch.tensor(pick, device=DEV)]
            ACT = []
            for i in range(0, S, 4):
                x = R[i:i + 4].to(DEV).reshape(-1, R.shape[-1]).half()
                pre = x @ Wsub + bsub
                ACT.append((pre * (pre > tsub)).float().cpu())
            ACT = torch.cat(ACT).reshape(S, T - 1, len(pick))

        for j, fid in enumerate(pick):
            col = ACT[:, :, j].reshape(-1)
            n = min(a.n_pos, int((col > 0).sum()))
            if n == 0:
                continue
            top = torch.topk(col, n)
            best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                    for v, ix in zip(top.values, top.indices)]
            dvec = s["W_dec"][int(fid)].float()
            ib = torch.stack([ids_all[i] for _, i, _ in best])
            pos = [p for _, _, p in best]
            vecs = [v * dvec for v, _, _ in best]
            res, flip, dtop = kl_multi(model, layer, ib, pos, vecs)
            for m in range(len(pos)):
                pn = float((best[m][0] * dvec).norm().item())
                # position recorded as a candidate FIFTH artifact: later
                # positions have more context and a sharper next-token
                # distribution, so KL is not comparable across them. Omitted
                # from sae_rare.csv, which is why that check could not be run
                # retrospectively.
                rows.append(dict(dict_tag=tag, layer=layer, width=width, l0=l0,
                                 fid=int(fid), freq=float(freq[fid]),
                                 pos=int(pos[m]), rel_pos=pos[m] / (T - 1),
                                 act=float(best[m][0]), pnorm=pn,
                                 **{f"kl_t{o}": res[o][m] for o in OFFSETS},
                                 **{f"kpn_t{o}": res[o][m] / max(pn, 1e-6)
                                    for o in OFFSETS},
                                 **{f"flip_t{o}": flip[o][m] for o in OFFSETS},
                                 **{f"dtop_t{o}": dtop[o][m]
                                    for o in OFFSETS}))
        del s, ACT, R
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"  {tag} done, rows={len(rows)}", flush=True)

    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
