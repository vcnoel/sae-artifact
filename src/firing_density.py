# -*- coding: utf-8 -*-
"""E29c: how many positions does a latent fire at, at each model scale?

The other half of the comparability question. Position agreement between two
dictionaries is bounded by how many candidate positions exist: a latent firing
at two tokens will be measured at the same one far more often, by chance alone,
than a latent firing at two hundred. So a difference in agreement between 2B and
9B is only a difference in the CONVENTION's behaviour if firing density is
comparable.

The direction matters and is not obvious in advance:
  9B latents SPARSER (fewer firing positions) -> chance agreement is HIGHER at
      9B, so a lower measured agreement there is more surprising, not less;
  9B latents DENSER -> part of any drop is mechanical and must be discounted.

Reports, per dictionary, the distribution over latents of the number of distinct
positions at which the latent fires, plus the implied chance agreement 1/k for a
uniform pick among those k. Needs one residual pass per model, which is why it
is worth running while the large GPU is available and expensive to want afterwards.
"""
import argparse
import io
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus import WIKI
from released_agreement import JumpReLU
from scope_matrix import SUITES as GRIDS

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="2b", choices=list(GRIDS))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n_seq", type=int, default=384)
    ap.add_argument("--n_lat", type=int, default=2048,
                    help="latents sampled per dictionary; seed fixed at 0")
    ap.add_argument("--paths", default=None,
                    help="comma-separated dictionary paths, overriding the "
                         "suite grid; needed at layers whose published L0 "
                         "values differ from the grid's")
    ap.add_argument("--out", default=None,
                    help="output CSV; defaults to the per-suite path. A depth "
                         "run MUST pass this, or it overwrites the headline "
                         "firing density for that suite.")
    ap.add_argument("--device_map", default=None)
    a = ap.parse_args()
    cfg = GRIDS[a.suite]
    layer = a.layer if a.layer is not None else cfg["layer"]
    out = a.out or f"results/firing_density_{a.suite}.csv"
    assert not os.path.exists(out) or a.out, (
        f"{out} exists; pass --out to write elsewhere. Refusing so a depth run "
        f"cannot overwrite the headline firing density for this suite.")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16,
        **({"device_map": a.device_map} if a.device_map else {})).eval()
    if not a.device_map:
        model = model.to(DEV)

    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        i = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if i.shape[1] >= 512:
            seqs.append(i[:, :512])
            buf = ""
            if len(seqs) >= a.n_seq:
                break
    ids = torch.cat(seqs, 0)
    S = ids.shape[0]
    R = []
    with torch.no_grad():
        for i in range(0, S, 8):
            R.append(model(input_ids=ids[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[layer + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    T = R.shape[1]
    del model
    torch.cuda.empty_cache()
    print(f"{a.suite}: residual {tuple(R.shape)}  d_model {R.shape[-1]}",
          flush=True)

    # The suite grids are layer-specific: layer 12's width_16k/average_l0_82
    # does not exist at layer 20, which publishes L0 22/38/71/139/294. --paths
    # lets a depth run measure density for the exact pair it evaluated, which
    # matters because firing density is the denominator of position agreement:
    # a denser pair has more candidate positions and mechanically lower
    # agreement, so the density belongs beside the agreement number.
    grid = cfg["grid"]
    if a.paths:
        grid = [(p.split("/")[-1], p) for p in a.paths.split(",") if p.strip()]

    rows = []
    for name, path in grid:
        sae = JumpReLU(hf_hub_download(cfg["repo"],
                                       f"layer_{layer}/{path}/params.npz"))
        rng = np.random.default_rng(0)
        sel_ids = np.sort(rng.choice(sae.width, min(a.n_lat, sae.width),
                                     replace=False))
        sel = torch.tensor(sel_ids, dtype=torch.long, device=DEV)
        fires = torch.zeros(len(sel_ids), dtype=torch.long)
        with torch.no_grad():
            for i in range(0, S, 8):
                x = R[i:i + 8].to(DEV).reshape(-1, R.shape[-1])
                fires += (sae.encode(x, sel) > 0).sum(0).cpu().long()
        k = fires.numpy().astype(float)
        live = k[k > 0]
        rows.append(dict(
            suite=a.suite, dictionary=name, width=int(sae.width),
            d_model=int(R.shape[-1]), n_sampled=len(sel_ids),
            frac_live=float((k > 0).mean()),
            fire_median=float(np.median(live)) if len(live) else 0.0,
            fire_q25=float(np.percentile(live, 25)) if len(live) else 0.0,
            fire_q75=float(np.percentile(live, 75)) if len(live) else 0.0,
            # chance that two independent picks among a latent's own firing
            # positions land on the same one, per latent, then medianed
            chance_agree_pct=float(np.median(100.0 / live)) if len(live) else 0.0,
            positions_total=int(S * T)))
        pd.DataFrame(rows).to_csv(out, index=False)
        print("  %-8s width %6d  live %4.1f%%  fires median %7.0f "
              "[q25 %.0f, q75 %.0f]  chance agree %.2f%%"
              % (name, sae.width, 100 * rows[-1]["frac_live"],
                 rows[-1]["fire_median"], rows[-1]["fire_q25"],
                 rows[-1]["fire_q75"], rows[-1]["chance_agree_pct"]),
              flush=True)
        del sae
        torch.cuda.empty_cache()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
