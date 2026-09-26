# -*- coding: utf-8 -*-
"""E26b: position agreement between EVERY pair of a set of released dictionaries.

released_agreement.py compares two hand-chosen Gemma Scope pairs. That invites
the question of whether those two were the convenient ones. This computes the
full pairwise matrix over a grid of released layer-12 dictionaries spanning
width and sparsity, so the claim becomes "no pair of Google's own dictionaries
agrees about where to measure" rather than "these two do not".

Every pair is matched by mutual-nearest-neighbour decoder cosine at a fixed
threshold and a fixed sample size, so cells are comparable to each other. The
median cosine of each pair's matched set is reported alongside, because a pair
of very similar dictionaries should agree more -- that is the gradient E26
measured, and it must not be mistaken for an artifact of the matrix.
"""
import argparse
import io
import itertools
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
from released_agreement import JumpReLU, match, positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Six released dictionaries per suite, spanning the axes a practitioner varies.
# 2B: 131k and above are left out deliberately -- their decoders are >1GB and
# the pairwise cosine against another does not fit alongside the residual cache
# on a 16GB card. 27B has no such choice to make: 131k is the ONLY width Google
# published, at three layers, so its grid varies sparsity alone and is a
# structurally weaker test than the other two. That is a property of the
# release, not of this script, and the write-up must say so.
SUITES = {
    "2b": dict(repo="google/gemma-scope-2b-pt-res", model="google/gemma-2-2b",
               layer=12, grid=[
                   ("16k", "width_16k/average_l0_22"),
                   ("16k*", "width_16k/average_l0_82"),
                   ("32k", "width_32k/average_l0_22"),
                   ("32k*", "width_32k/average_l0_76"),
                   ("65k", "width_65k/average_l0_21"),
                   ("65k*", "width_65k/average_l0_72")]),
    "9b": dict(repo="google/gemma-scope-9b-pt-res", model="google/gemma-2-9b",
               layer=20, grid=[
                   ("16k", "width_16k/average_l0_20"),
                   ("16k*", "width_16k/average_l0_68"),
                   ("32k", "width_32k/average_l0_11"),
                   ("32k*", "width_32k/average_l0_57"),
                   ("65k", "width_65k/average_l0_11"),
                   ("65k*", "width_65k/average_l0_55")]),
    "27b": dict(repo="google/gemma-scope-27b-pt-res",
                model="google/gemma-2-27b", layer=10, grid=[
                    ("L0 15", "width_131k/average_l0_15"),
                    ("L0 24", "width_131k/average_l0_24"),
                    ("L0 37", "width_131k/average_l0_37"),
                    ("L0 64", "width_131k/average_l0_64"),
                    ("L0 106", "width_131k/average_l0_106"),
                    ("L0 200", "width_131k/average_l0_200")]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="2b", choices=list(SUITES))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--n_seq", type=int, default=384)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--n_lat", type=int, default=384)
    ap.add_argument("--cos", type=float, default=0.5,
                    help="matched-pair floor; the same for every cell so the "
                         "cells are comparable")
    ap.add_argument("--shuffle_pairs", type=int, default=None,
                    help="seed for randomising pair order, so a run cut short "
                         "yields an unbiased subsample rather than every "
                         "narrow-vs-X pair and no wide-vs-wide one")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = SUITES[a.suite]
    REPO, MODEL = cfg["repo"], cfg["model"]
    LAYER = a.layer if a.layer is not None else cfg["layer"]
    GRID = cfg["grid"]
    out = a.out or (f"results/scope_matrix_{a.suite}.csv" if a.suite != "2b"
                    else "results/scope_matrix.csv")
    done = (pd.read_csv(out) if os.path.exists(out)
            else pd.DataFrame(columns=["a", "b"]))
    seen = {(r.a, r.b) for r in done.itertuples()}

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16,
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
                     .hidden_states[LAYER + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    T = R.shape[1]
    del model
    torch.cuda.empty_cache()
    print(f"residual {tuple(R.shape)}; {len(GRID)} dictionaries, "
          f"{len(list(itertools.combinations(GRID, 2)))} pairs", flush=True)

    rows = list(done.to_dict("records"))

    # PAIR ORDER DETERMINES WHETHER A PARTIAL RUN IS INTERPRETABLE.
    # combinations() walks the grid in width order, so the first five pairs are
    # all 16k-vs-X and a run cut short at teardown would contain no wide-vs-wide
    # pair at all -- its agreement range would look like the full range while
    # being a systematic subsample. Shuffling with a fixed seed makes every
    # prefix an unbiased sample of the 15 pairs, so a partial can be reported as
    # a random subsample with the pairs named. The seed is recorded in the CSV.
    order = list(itertools.combinations(GRID, 2))
    if a.shuffle_pairs is not None:
        np.random.default_rng(a.shuffle_pairs).shuffle(order)
        print(f"pair order shuffled with seed {a.shuffle_pairs}: any prefix is "
              f"an unbiased subsample", flush=True)
    for (na, pa), (nb, pb) in order:
        if (na, nb) in seen:
            print(f"  SKIP {na} vs {nb}", flush=True)
            continue
        A = JumpReLU(hf_hub_download(REPO, f"layer_{LAYER}/{pa}/params.npz"))
        B = JumpReLU(hf_hub_download(REPO, f"layer_{LAYER}/{pb}/params.npz"))
        m, _, _ = match(A, B, (a.cos,))
        ia, ib, cs = m[a.cos]
        if len(ia) < 50:
            print(f"  {na} vs {nb}: only {len(ia)} matched; skipped", flush=True)
            del A, B
            torch.cuda.empty_cache()
            continue
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(len(ia), min(a.n_lat, len(ia)), replace=False))
        ia, ib, cs = ia[take], ib[take], cs[take]
        ta, ma_ = positions(A, R, ia, a.n_pos, S, T)
        tb, mb_ = positions(B, R, ib, a.n_pos, S, T)
        jac, agr, live = [], [], 0
        for sa, sb, xa, xb in zip(ta, tb, ma_, mb_):
            if not sa or not sb:
                continue
            live += 1
            jac.append(len(sa & sb) / len(sa | sb))
            agr.append(1.0 if xa == xb else 0.0)
        rows.append(dict(a=na, b=nb, width_a=A.width, width_b=B.width,
                         n_matched=len(ia), n_live=live,
                         median_cos=float(np.median(cs)),
                         jaccard=float(np.mean(jac)),
                         top_agree=100.0 * float(np.mean(agr)),
                         pair_order_seed=a.shuffle_pairs,
                         pairs_total=len(order)))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {na:<5} vs {nb:<5}  matched {len(ia):>4}  "
              f"cos {np.median(cs):.3f}  agree {100 * np.mean(agr):5.1f}%  "
              f"jac {np.mean(jac):.3f}", flush=True)
        del A, B
        torch.cuda.empty_cache()

    d = pd.DataFrame(rows)
    print(f"\nwrote {out}: {len(d)} pairs")
    print(f"  agreement range {d.top_agree.min():.1f}% to {d.top_agree.max():.1f}%")


if __name__ == "__main__":
    main()
