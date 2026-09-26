# -*- coding: utf-8 -*-
"""How sparse is the latent in Figure 1, relative to the retained population?

Figure 1's latent fires at 5 and 4 of 511 positions. An earlier candidate fired
at 55 and 58. Those are opposite ends of some distribution, and a figure drawn
from either tail is unrepresentative in a way the caption must state. This
samples retained latents, extracts true traces on one sequence with the model
loaded once, and reports where the figure's latent sits.
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

ARMS = ["free", "tau090"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_lat", type=int, default=40)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--n_seq", type=int, default=384)
    a = ap.parse_args()
    cfg = model_configs.get("gemma2-2b")

    sh = pd.read_csv("results/eval_arms_g2_s384_shared.csv")
    ret = np.array(sorted(sh.fid.unique()))
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(ret, min(a.n_lat, len(ret)), replace=False))

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to("cuda").eval()
    arms = load_arms(cfg["dir"], cfg)

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
    with torch.no_grad():
        R = model(input_ids=seqs[a.seq].to("cuda"), output_hidden_states=True) \
            .hidden_states[cfg["layer"] + 1][:, 1:, :].float()

    sel = torch.tensor(pick, dtype=torch.long, device="cuda")
    counts = {}
    for tag in ARMS:
        with torch.no_grad():
            A = gated(arms[tag], R.reshape(-1, cfg["d_model"]), sel).cpu().numpy()
        counts[tag] = (A > 0).sum(0)

    tot = counts[ARMS[0]] + counts[ARMS[1]]
    d = pd.DataFrame({"fid": pick, ARMS[0]: counts[ARMS[0]],
                      ARMS[1]: counts[ARMS[1]], "total": tot})
    d.to_csv("results/density_sample.csv", index=False)
    print(f"sample of {len(pick)} retained latents, sequence {a.seq}, "
          f"{R.shape[1]} positions")
    print(d["total"].describe()[["min", "25%", "50%", "75%", "max"]].to_string())
    for f in (14405, 15765):
        if f in set(pick):
            v = int(d[d.fid == f]["total"].iloc[0])
            pct = 100.0 * float((d["total"] < v).mean())
            print(f"  fid {f}: total {v} -> {pct:.0f}th percentile")
        else:
            print(f"  fid {f}: not in sample; compare against quartiles above")


if __name__ == "__main__":
    main()
