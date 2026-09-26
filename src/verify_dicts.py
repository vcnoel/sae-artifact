# -*- coding: utf-8 -*-
"""Check every released dictionary against the sparsity printed in its own path.

Every guard in this repo verifies that a number matches its source. None
verified that the source was computed correctly, and that is exactly how the
decoder-bias bug survived: check_numbers.py was satisfied because the macros
equalled what the pipeline produced, and the pipeline was wrong.

The fix was available the whole time. `average_l0_82` is a published spec. This
measures mean L0 under our encode and fails loudly when it disagrees, so a
reader implementation cannot be silently wrong again. Explained variance is
reported alongside as a second, softer signal (the released figure depends on
the evaluation distribution, so it is printed, not asserted).

Runs standalone against the same dictionaries the reruns touch, so their output
is validated on arrival rather than trusted.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from corpus import WIKI
from released_agreement import JumpReLU
from scope_matrix import SUITES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 0.15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="2b", choices=list(SUITES))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n_seq", type=int, default=32)
    ap.add_argument("--paths", default=None,
                    help="comma-separated dictionary paths, overriding the "
                         "suite grid. Required with --layer when that layer "
                         "publishes different L0 values from the grid's, which "
                         "is the usual case: layer 12 has L0 82 at width 16k "
                         "and layer 20 has 22/38/71/139/294.")
    a = ap.parse_args()
    cfg = SUITES[a.suite]
    layer = a.layer if a.layer is not None else cfg["layer"]

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to(DEV).eval()
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
    R = []
    with torch.no_grad():
        for i in range(0, ids.shape[0], 8):
            R.append(model(input_ids=ids[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[layer + 1][:, 1:, :].float().cpu())
    x = torch.cat(R).reshape(-1, cfg.get("d_model") or R[0].shape[-1]).to(DEV)
    del model, R
    torch.cuda.empty_cache()

    print(f"{a.suite} layer {layer}: {x.shape[0]:,} activation vectors\n")
    print("%-26s %10s %10s %8s %10s" % ("dictionary", "spec L0", "measured",
                                        "verdict", "expl.var"))
    bad = []
    grid = cfg["grid"]
    if a.paths:
        grid = [(p.split("/")[-1], p) for p in a.paths.split(",") if p.strip()]
    for name, path in grid:
        sae = JumpReLU(hf_hub_download(cfg["repo"],
                                       f"layer_{layer}/{path}/params.npz"))
        meas, spec, ok = sae.verify_l0(x, tol=TOL)
        with torch.no_grad():
            acts = sae.encode(x[:8192])
            rec = acts @ sae.W_dec + sae.b_dec
            ev = 1 - ((x[:8192] - rec).var(0).sum()
                      / x[:8192].var(0).sum()).item()
        print("%-26s %10s %10.1f %8s %10.3f"
              % (path.split("/")[-1], spec, meas, "OK" if ok else "FAIL", ev))
        if not ok:
            bad.append((path, meas, spec))
        del sae
        torch.cuda.empty_cache()

    if bad:
        print("\nSPEC MISMATCH: the reader does not reproduce the released "
              "sparsity.", file=sys.stderr)
        for p, m, s in bad:
            print("  %s: measured L0 %.1f against specified %d" % (p, m, s),
                  file=sys.stderr)
        raise SystemExit(1)
    print(f"\nall dictionaries within {TOL:.0%} of their specified L0")


if __name__ == "__main__":
    main()
