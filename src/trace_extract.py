# -*- coding: utf-8 -*-
"""Real per-token activation traces for one latent under two arms.

WHY THIS EXISTS. The evaluation CSVs record only each arm's top-n positions,
so they cannot supply a trace across a sequence. A first version of the
mechanism figure filled the unrecorded positions with random baseline noise
while its docstring claimed the traces were real. In a paper about undisclosed
measurement conventions that is not a defensible figure, so the traces are
computed here instead: one forward pass, both arms encoded at every position.

Writes results/trace_<fid>.csv with one row per (arm, position).

NOT AN INDEPENDENT CHECK. This script imports `gated` and `load_arms` from the
evaluation path and reads the same checkpoints, so reproducing an argmax
confirms that the same function was called on the same weights. It is a
consistency check, not independent validation.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import model_configs
from eval_arms import load_arms, gated
from eval_saes import WIKI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma2-2b")
    ap.add_argument("--fid", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True,
                    help="index of the sequence, matching the eval run")
    ap.add_argument("--n_seq", type=int, default=384)
    ap.add_argument("--arms", default="free,tau090")
    a = ap.parse_args()
    cfg = model_configs.get(a.base)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to("cuda").eval()
    arms = load_arms(cfg["dir"], cfg)
    want = a.arms.split(",")

    # rebuild the identical sequence set the evaluation used
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
    ids = seqs[a.seq]
    with torch.no_grad():
        R = model(input_ids=ids.to("cuda"), output_hidden_states=True) \
            .hidden_states[cfg["layer"] + 1][:, 1:, :].float()

    sel = torch.tensor([a.fid], dtype=torch.long, device="cuda")
    rows = []
    toks = [tok.decode([t]) for t in ids[0, 1:].tolist()]
    for tag in want:
        m = arms[tag]
        with torch.no_grad():
            act = gated(m, R.reshape(-1, cfg["d_model"]), sel).cpu().numpy()
        for p, v in enumerate(act[:, 0], start=1):
            rows.append(dict(arm=tag, pos=p, act=float(v),
                             token=toks[p - 1]))
    d = pd.DataFrame(rows)
    out = f"results/trace_{a.fid}.csv"
    d.to_csv(out, index=False)
    for tag in want:
        s = d[d.arm == tag]
        print(f"  {tag}: argmax at pos {int(s.loc[s.act.idxmax()].pos)}, "
              f"nonzero at {int((s.act > 0).sum())}/{len(s)} positions")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
