# -*- coding: utf-8 -*-
"""Crossed latent x arm measurement on a SHARED feature list.

WHY THIS EXISTS. eval_saes.py samples uniformly inside the per-arm loop with an
advancing RNG, so each arm received a different feature list and only 6 latents
overlapped -- enough for nothing. The trained and soft-frozen arms were seeded
identically, so W0 is shared and latent i denotes the same initial direction in
both; measuring the SAME ids in both arms is what makes a latent x arm variance
decomposition possible, and it is the direct analogue of model x wrapper.

Design: intersect the live sets across arms, draw one uniform sample from the
intersection, measure every arm on that sample at several positions each. The
untrained arm is included for the endpoint comparison but is NOT part of the
crossed analysis, since it was seeded differently and its latent i is unrelated.
"""
import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_saes import D_MODEL, DEV, K, LAYER, MODEL, SEQ, WIDTH, TopKSAE
from eval_saes import WIKI, kl_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/saes.pt")
    ap.add_argument("--n_seq", type=int, default=96)
    ap.add_argument("--n_feat", type=int, default=240)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--out", default="results/eval_shared.csv")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()
    sd = torch.load(a.ckpt, map_location=DEV)
    arms = {}
    for n in ("trained", "frozen"):
        m = TopKSAE(D_MODEL, WIDTH, K, 0).to(DEV)
        m.load_state_dict(sd[n])
        arms[n] = m.eval()
    m = TopKSAE(D_MODEL, WIDTH, K, 999).to(DEV)
    m.b_dec.data = arms["trained"].b_dec.data.clone()
    arms["random"] = m.eval()

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

    R = []
    with torch.no_grad():
        for i in range(0, S, 8):
            R.append(model(input_ids=ids_all[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[LAYER + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    print(f"resid {tuple(R.shape)}", flush=True)

    # pass 1: per-arm activation frequency, accumulated without storing A
    live = None
    for name, mm in arms.items():
        cnt = torch.zeros(WIDTH, device=DEV)
        with torch.no_grad():
            for i in range(0, S, 8):
                A = mm.encode(R[i:i + 8].to(DEV).reshape(-1, D_MODEL))
                cnt += (A > 0).float().sum(0)
                del A
        f = (cnt / (S * (T - 1))).cpu().numpy()
        alive = set(np.where(f > 1e-5)[0].tolist())
        live = alive if live is None else (live & alive)
        print(f"  {name}: {len(alive)} live", flush=True)
    live = np.array(sorted(live))
    rng = np.random.default_rng(0)
    fids = np.sort(rng.choice(live, size=min(a.n_feat, len(live)),
                              replace=False))
    print(f"shared live intersection {len(live)}; sampled {len(fids)}",
          flush=True)

    rows = []
    for name, mm in arms.items():
        sel = torch.tensor(fids, dtype=torch.long, device=DEV)
        with torch.no_grad():
            Wsub = mm.W_enc[:, sel]
            ACT = []
            for i in range(0, S, 8):
                x = R[i:i + 8].to(DEV).reshape(-1, D_MODEL) - mm.b_dec
                pre = x @ Wsub + mm.b_enc[sel]
                # TopK gating is over the FULL width in training, so the k-th
                # largest must come from the full pre-activation vector, not from
                # the sampled subset -- otherwise a 240-column subset would gate
                # as if the dictionary were 240 wide.
                full = (R[i:i + 8].to(DEV).reshape(-1, D_MODEL) - mm.b_dec) \
                    @ mm.W_enc + mm.b_enc
                kth = torch.topk(full, mm.k, dim=-1).values[:, -1:]
                gate = (pre >= kth).float()
                ACT.append((torch.relu(pre) * gate).cpu())
                del full, pre
            ACT = torch.cat(ACT).reshape(S, T - 1, len(fids))
        for j, fid in enumerate(fids):
            col = ACT[:, :, j].reshape(-1)
            n = min(a.n_pos, int((col > 0).sum()))
            if n == 0:
                continue
            top = torch.topk(col, n)
            best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                    for v, ix in zip(top.values, top.indices)]
            dvec = mm.W_dec.data[int(fid)]
            ib = torch.stack([ids_all[i] for _, i, _ in best])
            pos = [p for _, _, p in best]
            vecs = [v * dvec for v, _, _ in best]
            kls = kl_batch(model, ib, pos, vecs)
            for kl, (v, _, p) in zip(kls, best):
                pn = float((v * dvec).norm().item())
                rows.append(dict(arm=name, fid=int(fid), pos=p,
                                 rel_pos=p / (T - 1), act=v, kl=kl, pnorm=pn,
                                 kl_per_norm=kl / max(pn, 1e-6)))
        print(f"  {name} measured, rows={len(rows)}", flush=True)
        del ACT
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ov = df.groupby("fid").arm.nunique()
    print(f"wrote {a.out}  rows={len(df)}  "
          f"latents in all 3 arms={int((ov == 3).sum())}")


if __name__ == "__main__":
    main()
