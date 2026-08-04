# -*- coding: utf-8 -*-
"""Measure every arm on ONE shared latent list -- the six-level crossed design.

All arms were trained from seed 0, so W0 is identical and latent i denotes the
same initial direction in each. The live sets are intersected, one uniform sample
is drawn from the intersection, and every arm is measured on exactly that sample.
Anything else makes the design nested and the interaction inestimable.

Each arm carries its own k, so the TopK gate is taken from that arm's full-width
pre-activation -- gating a 240-column subset would behave as if the dictionary
were 240 wide.
"""
import argparse
import glob
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch

from train_saes import DEV, TopKSAE
from eval_saes import WIKI, kl_batch
from transformers import AutoModelForCausalLM, AutoTokenizer
import model_configs


def load_arms(arm_dir, cfg):
    """Each checkpoint carries its own d_model/width when present (arms
    trained after model_configs was introduced); older Gemma-2-2B checkpoints
    predate that field, so fall back to the requested base's config -- they
    were only ever trained on gemma2-2b anyway."""
    arms = {}
    for p in sorted(glob.glob(f"{arm_dir}/arm_*.pt")):
        d = torch.load(p, map_location=DEV)
        d_model = int(d.get("d_model", cfg["d_model"]))
        width = int(d.get("width", cfg["width"]))
        m = TopKSAE(d_model, width, int(d["k"]), 0).to(DEV)
        m.load_state_dict(d["state"])
        arms[d["tag"]] = m.eval()
    return arms


@torch.no_grad()
def gated(m, X, sel):
    """Activation of the selected latents under this arm's full-width TopK."""
    x = X - m.b_dec
    pre_sel = x @ m.W_enc[:, sel] + m.b_enc[sel]
    full = x @ m.W_enc + m.b_enc
    kth = torch.topk(full, m.k, dim=-1).values[:, -1:]
    del full
    return torch.relu(pre_sel) * (pre_sel >= kth).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seq", type=int, default=96)
    ap.add_argument("--n_feat", type=int, default=240)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--out", default="results/eval_arms.csv")
    ap.add_argument("--base", default="gemma2-2b",
                    choices=list(model_configs.MODELS),
                    help="which base model the arms in --arm_dir were fit on")
    ap.add_argument("--arm_dir", default=None,
                    help="defaults to the base's own arm directory")
    a = ap.parse_args()
    cfg = model_configs.get(a.base)
    arm_dir = a.arm_dir or cfg["dir"]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to(DEV).eval()
    arms = load_arms(arm_dir, cfg)
    print(f"arms: {list(arms)}  (k = "
          f"{ {t: m.k for t, m in arms.items()} })", flush=True)
    if len(arms) < 3:
        raise SystemExit(f"need at least 3 arms in {arm_dir}; "
                         f"run src/chain6.sh (or the --base {a.base} "
                         f"equivalent) first")

    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        i = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if i.shape[1] >= cfg["seq"]:
            seqs.append(i[:, :cfg["seq"]])
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
                     .hidden_states[cfg["layer"] + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    print(f"resid {tuple(R.shape)}", flush=True)

    live = None
    for tag, m in arms.items():
        cnt = torch.zeros(cfg["width"], device=DEV)
        with torch.no_grad():
            for i in range(0, S, 8):
                A = m.encode(R[i:i + 8].to(DEV).reshape(-1, cfg["d_model"]))
                cnt += (A > 0).float().sum(0)
                del A
        al = set(np.where((cnt / (S * (T - 1))).cpu().numpy() > 1e-5)[0].tolist())
        live = al if live is None else (live & al)
        print(f"  {tag}: {len(al)} live", flush=True)
    live = np.array(sorted(live))
    fids = np.sort(np.random.default_rng(0).choice(
        live, size=min(a.n_feat, len(live)), replace=False))
    print(f"shared intersection {len(live)}; sampled {len(fids)}", flush=True)

    sel = torch.tensor(fids, dtype=torch.long, device=DEV)
    rows = []
    for tag, m in arms.items():
        ACT = []
        with torch.no_grad():
            for i in range(0, S, 8):
                ACT.append(gated(m, R[i:i + 8].to(DEV).reshape(-1, cfg["d_model"]),
                                 sel).cpu())
        ACT = torch.cat(ACT).reshape(S, T - 1, len(fids))
        for j, fid in enumerate(fids):
            col = ACT[:, :, j].reshape(-1)
            n = min(a.n_pos, int((col > 0).sum()))
            if n == 0:
                continue
            top = torch.topk(col, n)
            best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                    for v, ix in zip(top.values, top.indices)]
            dvec = m.W_dec.data[int(fid)]
            ib = torch.stack([ids_all[i] for _, i, _ in best])
            kls = kl_batch(model, ib, [p for _, _, p in best],
                           [v * dvec for v, _, _ in best])
            for kl, (v, _, p) in zip(kls, best):
                pn = float((v * dvec).norm().item())
                rows.append(dict(arm=tag, fid=int(fid), pos=p,
                                 rel_pos=p / (T - 1), act=v, kl=kl, pnorm=pn,
                                 kl_per_norm=kl / max(pn, 1e-6)))
        print(f"  {tag} measured, rows={len(rows)}", flush=True)
        del ACT
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ov = df.groupby("fid").arm.nunique()
    print(f"wrote {a.out} rows={len(df)}  latents in all "
          f"{len(arms)} arms={int((ov == len(arms)).sum())}")


if __name__ == "__main__":
    main()
