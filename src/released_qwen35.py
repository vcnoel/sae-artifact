# -*- coding: utf-8 -*-
"""E26q: do two RELEASED Qwen-Scope dictionaries agree on where to measure?

released_agreement.py measures position agreement between released Gemma Scope
pairs. Every one of those is a JumpReLU dictionary trained by one lab on one
model family, which leaves the premise open to "a Gemma Scope property". Qwen
has since released production dictionaries on Qwen3.5 (Qwen-Scope), trained
with a different recipe and a different activation, so this is the same
measurement on a second lab's release. No training, forward passes only.

WHAT DIFFERS FROM GEMMA SCOPE, AND WHY EACH DIFFERENCE IS HANDLED HERE.

  Activation. Qwen-Scope is TopK (k = 50 or 100), not JumpReLU. A TopK latent's
  activation depends on every other latent's pre-activation, so the per-latent
  column shortcut released_agreement.JumpReLU.encode takes is WRONG here: the
  full pre-activation is computed, the top k kept, and only then are the
  selected columns read out. The released app.py applies ReLU before the top-k;
  the README snippet does not. ReLU-then-topk is used because it is what the
  release's own tool runs; with k << width the two differ only where fewer than
  k pre-activations are positive.

  Layout. W_enc is (d_sae, d_model) and W_dec is (d_model, d_sae), the
  transpose of Gemma Scope's. Both are stored here in Gemma Scope's orientation
  so match() and positions() are reused unchanged.

  The grid. Each size ships exactly two dictionaries per layer, same width and
  k = 50 vs 100. So the only pair is a SPARSITY pair: the analogue of the
  paper's "same width, L0 82 vs 22" and never of the width pair. Like Gemma
  Scope 27B this is a strictly weaker test than the 2B grid and must be
  reported as one.

  L0 is not a check. The JumpReLU reader caught its decoder-bias bug because
  the measured L0 disagreed with the L0 in the path. A TopK encode returns k
  actives under ANY convention, so the equivalent check here is reconstruction:
  fraction of variance unexplained under the released convention and under
  the (x - b_dec) alternative, both printed. The released one must be the
  better of the two and must be plausible, or the run stops before the
  position stage.

  Hook point. resid_post = output of model.layers[L], which is what the
  release's app.py hooks and what hidden_states[L + 1] is for L below the last
  layer. The residual is captured by a forward hook that stops the forward
  there, so layers above L never run -- and, with --truncate, are never
  loaded, which is what lets the 9B fit a 16GB card.

Layer: nearest to the 46% depth used for Gemma (layer 12 of 26). 2B: 11 of 24
(also model_configs.py's qwen35-2b layer). 9B: 15 of 32.
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from corpus import WIKI
from released_agreement import match, positions

DEV = "cuda" if torch.cuda.is_available() else "cpu"

SUITES = {
    "2b": dict(model="Qwen/Qwen3.5-2B-Base", layer=11, d_model=2048,
               left="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100",
               right="Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50"),
    # 19.3GB of bf16 weights, which does not fit 16GB whole. --truncate loads
    # layers 0..15 only (about half the stack plus both embeddings).
    "9b": dict(model="Qwen/Qwen3.5-9B-Base", layer=15, d_model=4096,
               left="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100",
               right="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50"),
}


def fetch(repo, fname):
    """hf_hub_download, then the file's sha256 against the Hub's LFS record.

    This machine has corrupted large HTTPS transfers before. A truncated or
    bit-flipped .pt either fails to load -- fine -- or loads with wrong values,
    which nothing downstream would notice. Hashing 0.5-2GB costs seconds.
    """
    path = hf_hub_download(repo, fname)
    info = HfApi().model_info(repo, files_metadata=True)
    want = next((s.lfs.sha256 if hasattr(s.lfs, "sha256") else s.lfs["sha256"])
                for s in info.siblings if s.rfilename == fname and s.lfs)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    assert h.hexdigest() == want, (
        f"{repo}/{fname}: sha256 {h.hexdigest()} != Hub {want}. Delete the "
        f"cached blob and re-download; do not run on a corrupted dictionary.")
    return path


class TopK:
    """Qwen-Scope's released parameterisation, as published."""

    def __init__(self, repo, layer):
        cfg = json.load(open(hf_hub_download(repo, "config.json")))
        assert cfg["model_type"] == "topk_sae" and cfg["hook_point"] == \
            "resid_post", f"{repo}: unexpected config {cfg}"
        assert layer in cfg["layers"], f"{repo} does not cover layer {layer}"
        z = torch.load(fetch(repo, f"layer{layer}.sae.pt"), map_location="cpu",
                       weights_only=True)
        # stored in Gemma Scope's orientation: W_enc (d_model, width),
        # W_dec (width, d_model), so match() and positions() apply unchanged
        self.W_enc = z["W_enc"].T.contiguous().float().to(DEV)
        self.b_enc = z["b_enc"].float().to(DEV)
        self.W_dec = z["W_dec"].T.contiguous().float().to(DEV)
        self.b_dec = z["b_dec"].float().to(DEV)
        self.width = self.W_dec.shape[0]
        self.k = int(cfg["k"])
        assert self.width == cfg["d_sae"] and \
            self.W_enc.shape[0] == cfg["d_model"], f"{repo}: shape mismatch"
        self.path = f"{repo}/layer{layer}.sae.pt"

    @torch.no_grad()
    def encode(self, x, sel=None, centre=False):
        # The top-k is over ALL latents; `sel` is applied afterwards. Slicing
        # W_enc first, as the JumpReLU reader does, would take the top k of the
        # selected columns alone and every latent would fire at every token.
        pre = (x - self.b_dec if centre else x) @ self.W_enc + self.b_enc
        pre = torch.relu(pre)
        v, i = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter_(-1, i, v)
        return acts if sel is None else acts[:, sel]

    @torch.no_grad()
    def fvu(self, x, centre=False):
        rec = self.encode(x, centre=centre) @ self.W_dec + self.b_dec
        return float(((x - rec) ** 2).sum() /
                     ((x - x.mean(0)) ** 2).sum())

    @torch.no_grad()
    def verify(self, R, sample=8192, max_fvu=0.5):
        """FVU under the released encode and the (x - b_dec) alternative.

        Returns (released, alternative, ok). The convention is decided by which
        reconstructs, not by recollection -- the lesson of the Gemma Scope
        decoder-bias bug, restated for an activation whose L0 cannot fail.
        """
        flat = R.reshape(-1, R.shape[-1])
        g = torch.Generator().manual_seed(0)
        x = flat[torch.randperm(flat.shape[0], generator=g)[:sample]].to(DEV)
        rel, alt = self.fvu(x), self.fvu(x, centre=True)
        return rel, alt, (rel <= max_fvu and rel <= alt)


def decoder_layers(model):
    for path in ("model.layers", "model.language_model.layers",
                 "language_model.model.layers"):
        m = model
        try:
            for p in path.split("."):
                m = getattr(m, p)
            return m
        except AttributeError:
            continue
    raise SystemExit("could not find the decoder layers on this model")


class _Stop(Exception):
    pass


@torch.no_grad()
def capture(model, ids, layer, bs):
    """resid_post at `layer`, first position dropped, forward stopped there.

    The first position is dropped as every other script in this repo does. The
    Qwen tokenizer adds no BOS, so it is a real token -- but it is also the
    attention sink with a residual norm far above the rest, which would take the
    argmax of many latents for reasons unrelated to what they represent.
    """
    buf = {}

    def hook(_m, _i, out):
        buf["h"] = (out[0] if isinstance(out, tuple) else out)
        raise _Stop

    h = decoder_layers(model)[layer].register_forward_hook(hook)
    dev = next(model.parameters()).device
    R = []
    try:
        for i in range(0, ids.shape[0], bs):
            try:
                model(input_ids=ids[i:i + bs].to(dev))
            except _Stop:
                pass
            R.append(buf.pop("h")[:, 1:, :].float().cpu())
    finally:
        h.remove()
    return torch.cat(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="2b", choices=list(SUITES))
    ap.add_argument("--n_seq", type=int, default=384)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--n_lat", type=int, default=512,
                    help="matched pairs carried to the position stage")
    ap.add_argument("--cos", type=float, default=0.7,
                    help="primary cosine threshold; others are also reported")
    ap.add_argument("--layer", type=int, default=None,
                    help="override the suite's default layer")
    ap.add_argument("--bs", type=int, default=8,
                    help="forward batch; the linear-attention layers fall back "
                         "to a torch implementation without fla installed, "
                         "which is memory-hungry -- drop to 4 on the 9B")
    ap.add_argument("--truncate", action="store_true",
                    help="load only layers 0..layer. The layers above never run "
                         "(the forward stops at the hook), so their weights "
                         "are dead memory; this is what fits the 9B on 16GB")
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--skip_checks", action="store_true",
                    help="do not stop on a failed hook or FVU check")
    ap.add_argument("--out", default=None,
                    help="path for the .txt; the .csv follows the same stem")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing results CSV")
    a = ap.parse_args()
    cfg = dict(SUITES[a.suite])
    if a.layer is not None:
        cfg["layer"] = a.layer
    out_path = a.out or f"results/released_qwen35_{a.suite}.txt"
    csv_path = re.sub(r"\.txt$", "", out_path) + ".csv"
    assert not os.path.exists(csv_path) or a.overwrite, (
        f"{csv_path} exists; pass --overwrite to replace it. Refusing so a "
        f"completed run cannot be clobbered by a later one.")
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    kw = dict(dtype=torch.bfloat16)
    if a.truncate:
        mc = AutoConfig.from_pretrained(cfg["model"])
        tc = getattr(mc, "text_config", mc)
        n = cfg["layer"] + 1
        tc.num_hidden_layers = n
        if getattr(tc, "layer_types", None):
            tc.layer_types = tc.layer_types[:n]
        kw["config"] = tc
    if a.device_map:
        kw["device_map"] = a.device_map
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], **kw).eval()
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
    ids_all = torch.cat(seqs, 0)
    S = ids_all.shape[0]

    # The hook must read what the paper's other scripts read. On the full
    # model, one batch through output_hidden_states is compared against it;
    # truncated, hidden_states[L + 1] would be the final-normed state, so the
    # comparison is only meaningful untruncated and is skipped otherwise.
    # Truncation itself was checked on the 2B at layer 11 (CPU, bf16): all
    # 161 retained tensors identical and the hooked residual bit-identical to
    # the full model's. Note dtype= is honoured only in bf16 here: asking the
    # composite checkpoint for float32 silently loads bf16 untruncated and
    # float32 truncated, which differs by ~1% and looks like a truncation bug.
    if not a.truncate:
        with torch.no_grad():
            dev = next(model.parameters()).device
            ref = model(input_ids=ids_all[:2].to(dev), output_hidden_states=True
                        ).hidden_states[cfg["layer"] + 1][:, 1:, :].float().cpu()
        got = capture(model, ids_all[:2], cfg["layer"], 2)
        ok = torch.allclose(ref, got, rtol=1e-3, atol=1e-3)
        p(f"hook vs hidden_states[{cfg['layer'] + 1}]: max abs diff "
          f"{float((ref - got).abs().max()):.2e} ({'ok' if ok else 'MISMATCH'})")
        assert ok or a.skip_checks, "hook does not read hidden_states[L + 1]"
    R = capture(model, ids_all, cfg["layer"], a.bs)
    T = R.shape[1]
    del model
    torch.cuda.empty_cache()
    p(f"suite {a.suite}: {cfg['model']} layer {cfg['layer']}, "
      f"residual {tuple(R.shape)}")
    p("")

    THR = (0.5, 0.6, 0.7, 0.8, 0.9)
    rows = []
    A = TopK(cfg["left"], cfg["layer"])
    B = TopK(cfg["right"], cfg["layer"])
    label = f"same width, k {A.k} vs {B.k}"
    p("=" * 84)
    p(f"{label}   ({cfg['left']}  vs  {cfg['right']})")
    p("=" * 84)
    p(f"  widths {A.width} and {B.width}")
    fv = {}
    for nm, D in (("left", A), ("right", B)):
        rel, alt, ok = D.verify(R)
        fv[nm] = rel
        p(f"  {nm:<5} k {D.k:>3}  FVU released {rel:.3f}  "
          f"(x - b_dec) {alt:.3f}  {'ok' if ok else 'CHECK FAILED'}")
        assert ok or a.skip_checks, (
            f"{D.path}: released encode does not reconstruct (FVU {rel:.3f}, "
            f"alternative {alt:.3f}); wrong hook, layer or convention")
    m, best, mutual = match(A, B, THR)
    c = best[mutual]
    p(f"  mutual nearest neighbours: {int(mutual.sum())} of {A.width}, "
      f"median cosine {np.median(c):.3f}")
    p(f"{'cosine >=':>12}{'matched pairs':>16}{'% of left dict':>16}")
    for t in THR:
        n = len(m[t][0])
        p(f"{t:>12.2f}{n:>16}{100.0 * n / A.width:>15.1f}%")

    # Same five disjoint bands and two cumulative thresholds as the Gemma run,
    # so every row has a Gemma counterpart. share_mutual is each band's share
    # of ALL mutual matches, the cosine_dist.py definition, which is what the
    # population-weighted figure needs; it is computed here because this pair
    # has no row in cosine_dist_2b.csv.
    p("")
    p(f"{'matching':>12}{'matched':>9}{'median cos':>12}{'firing both':>13}"
      f"{'Jaccard':>10}{'top-1 agree':>13}{'% mutual':>10}")
    bands = [("0.50-0.60", 0.50, 0.60), ("0.60-0.70", 0.60, 0.70),
             ("0.70-0.80", 0.70, 0.80), ("0.80-0.90", 0.80, 0.90),
             ("0.90-1.00", 0.90, 1.01),
             (">= 0.50", 0.50, 1.01), (">= %.2f" % a.cos, a.cos, 1.01)]
    for lab, lo_c, hi_c in bands:
        share = 100.0 * float(((c >= lo_c) & (c < hi_c)).mean())
        ia, ib, cs = m[lo_c]
        keep = cs < hi_c
        ia, ib, cs = ia[keep], ib[keep], cs[keep]
        if len(ia) == 0:
            p(f"{lab:>12}   no pairs")
            continue
        rng = np.random.default_rng(0)
        take = np.sort(rng.choice(len(ia), min(a.n_lat, len(ia)),
                                  replace=False))
        ia, ib, cs = ia[take], ib[take], cs[take]
        ta, ma_ = positions(A, R, ia, a.n_pos, S, T)
        tb, mb_ = positions(B, R, ib, a.n_pos, S, T)
        jac, agree, live = [], [], 0
        for sa, sb, xa, xb in zip(ta, tb, ma_, mb_):
            if not sa or not sb:
                continue
            live += 1
            jac.append(len(sa & sb) / len(sa | sb))
            agree.append(1.0 if xa == xb else 0.0)
        p(f"{lab:>12}{len(ia):>9}{np.median(cs):>12.3f}{live:>13}"
          f"{np.mean(jac):>10.3f}{100.0 * np.mean(agree):>12.1f}%"
          f"{share:>9.1f}%")
        # the released_agreement_2b.csv columns first, in its order, so the
        # macro code reads both files the same way; Qwen-only columns after
        rows.append(dict(suite=a.suite, pair=label, left=cfg["left"],
                         right=cfg["right"], width_left=A.width,
                         width_right=B.width, matching=lab, cos_lo=lo_c,
                         cos_hi=hi_c, n_matched=len(ia), n_live=live,
                         median_cos=float(np.median(cs)),
                         jaccard=float(np.mean(jac)),
                         top_agree=100.0 * float(np.mean(agree)),
                         layer=cfg["layer"], k_left=A.k, k_right=B.k,
                         fvu_left=fv["left"], fvu_right=fv["right"],
                         n_mutual=int(mutual.sum()),
                         cos_median_mutual=float(np.median(c)),
                         share_mutual=share))
    p("")
    del A, B
    torch.cuda.empty_cache()

    p("=" * 84)
    p("Matched pairs are the latents the two dictionaries agree MOST about --")
    p("near-identical decoder directions -- so these figures are an UPPER bound")
    p("on agreement across the full dictionaries, not a typical value.")
    p("One pair only: Qwen-Scope ships one width per size, so this is a")
    p("sparsity contrast and has no width counterpart.")
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    with io.open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
