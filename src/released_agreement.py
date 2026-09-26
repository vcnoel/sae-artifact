# -*- coding: utf-8 -*-
"""E26: do two RELEASED production dictionaries agree on where to measure?

The paper's premise -- that the measurement position is selected by the
dictionary, so two dictionaries under comparison are read at different tokens --
is stated on six self-trained arms at 12M tokens. That invites the obvious
objection: a toy artifact of an undertrained dictionary.

The premise does not need shared initialisation. It needs two dictionaries a
practitioner might actually compare, and Gemma Scope ships several per layer at
different widths and L0 targets. This measures position agreement between
released pairs at full training scale. No training, forward passes only.

WHY MATCHING IS NEEDED, AND WHAT IT COSTS. Two released dictionaries have no
index correspondence -- latent 500 of one is unrelated to latent 500 of the
other -- so latents are matched by decoder cosine, as Paulo and Belrose do for
the same reason. Matching is MUTUAL nearest neighbour above a stated threshold,
and the survivor count is reported, because a permissive threshold would admit
pairs that are not the same feature and a strict one selects the most
duplicated directions. Both are reported at several thresholds so the reader
can see the sensitivity rather than take one number.

DIRECTION OF THE SELECTION BIAS. Matched pairs are the latents the two
dictionaries agree MOST about -- they are the ones with a near-identical
decoder direction. If anything, they should agree about position more than a
random pair would, so agreement measured here is an UPPER bound on agreement
across the full dictionaries. That runs in the paper's favour and is stated so
it cannot be read the other way.
"""
import argparse
import io
import os
import re
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (repo, base model, layer, d_model). Layer 12 for the 2B to match the paper.
SUITES = {
    "2b": dict(repo="google/gemma-scope-2b-pt-res", model="google/gemma-2-2b",
               layer=12, d_model=2304),
    "9b": dict(repo="google/gemma-scope-9b-pt-res", model="google/gemma-2-9b",
               layer=20, d_model=3584),
    # 27B publishes only three layers (10, 22, 34) and only width 131k, so a
    # 27B pair can differ in SPARSITY but never in width. That is a strictly
    # weaker test than 2B or 9B and must be reported as one.
    "27b": dict(repo="google/gemma-scope-27b-pt-res",
                model="google/gemma-2-27b", layer=10, d_model=4608),
}

# Pairs chosen to mirror two of the paper's own arm contrasts: same width /
# different sparsity, and different width / comparable sparsity. The second is
# the situation a reader comparing two published numbers is actually in.
PAIRS = {
    "2b": [("width_16k/average_l0_82", "width_16k/average_l0_22", "same width, L0 82 vs 22"),
           ("width_16k/average_l0_82", "width_65k/average_l0_72", "16k vs 65k, L0 82 vs 72")],
    # 9B L0 targets differ from the 2B suite's; these are the released values
    # for layer 20, checked against the repo file list rather than assumed.
    "9b": [("width_16k/average_l0_68", "width_16k/average_l0_20",
            "same width, L0 68 vs 20"),
           ("width_16k/average_l0_58", "width_65k/average_l0_55",
            "16k vs 65k, L0 58 vs 55")],
    # 27B: sparsity-only, because 131k is the only width published.
    "27b": [("width_131k/average_l0_64", "width_131k/average_l0_15",
             "same width, L0 64 vs 15"),
            ("width_131k/average_l0_106", "width_131k/average_l0_24",
             "same width, L0 106 vs 24")],
}


class JumpReLU:
    """Gemma Scope's released parameterisation, as published."""

    def __init__(self, path):
        z = np.load(path)
        self.W_enc = torch.tensor(z["W_enc"], dtype=torch.float32, device=DEV)
        self.b_enc = torch.tensor(z["b_enc"], dtype=torch.float32, device=DEV)
        self.W_dec = torch.tensor(z["W_dec"], dtype=torch.float32, device=DEV)
        self.b_dec = torch.tensor(z["b_dec"], dtype=torch.float32, device=DEV)
        self.thr = torch.tensor(z["threshold"], dtype=torch.float32, device=DEV)
        self.width = self.W_dec.shape[0]
        # The released sparsity is printed in the dictionary's own path
        # (average_l0_82). It is a published spec and it is the check that
        # would have caught the decoder-bias bug immediately: the wrong encode
        # gave mean L0 301.8 for a dictionary whose filename says 82.
        m = re.search(r"average_l0_(\d+)", str(path))
        self.spec_l0 = int(m.group(1)) if m else None
        self.path = str(path)

    @torch.no_grad()
    def verify_l0(self, x, tol=0.15, sample=8192):
        """Measure mean L0 and compare against the spec in the path.

        Returns (measured, spec, ok). Callers print it so the pair is in the
        run log: a check whose result is not recorded is a check nobody can
        confirm ran.
        """
        if self.spec_l0 is None:
            return None, None, True
        n = min(sample, x.shape[0])
        idx = torch.randperm(x.shape[0], device=x.device)[:n]
        acts = self.encode(x[idx])
        measured = float((acts > 0).float().sum(-1).mean())
        ok = abs(measured - self.spec_l0) <= tol * self.spec_l0
        return measured, self.spec_l0, ok

    @torch.no_grad()
    def encode(self, x, sel=None):
        # NO decoder-bias subtraction. Gemma Scope's published encode is
        # pre = x @ W_enc + b_enc, and an earlier version of this reader
        # computed (x - b_dec) @ W_enc + b_enc instead. Measured on layer 12
        # width 16k L0 82, that produced mean L0 301.8 and explained variance
        # 0.158, against 83.8 and 0.766 for the released convention -- and the
        # released spec for that dictionary is L0 = 82. Every released-
        # dictionary number computed before this fix is wrong: 16% of latents
        # took a different argmax and the live fraction was 0.95 rather than
        # 0.78. The convention is decided by the L0 match, not by recollection.
        pre = x @ (self.W_enc if sel is None else self.W_enc[:, sel])
        pre = pre + (self.b_enc if sel is None else self.b_enc[sel])
        thr = self.thr if sel is None else self.thr[sel]
        return torch.relu(pre) * (pre > thr).float()


@torch.no_grad()
def match(a, b, thresholds, chunk=2048):
    """Mutual nearest neighbour by decoder cosine, at several thresholds.

    Computed in chunks: 1M x 1M would not fit, and the 16k x 65k case is
    already 1e9 entries at float32.
    """
    A = a.W_dec / a.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    B = b.W_dec / b.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    best_ab = torch.zeros(A.shape[0], device=DEV)
    arg_ab = torch.zeros(A.shape[0], dtype=torch.long, device=DEV)
    best_ba = torch.full((B.shape[0],), -2.0, device=DEV)
    arg_ba = torch.zeros(B.shape[0], dtype=torch.long, device=DEV)
    for i in range(0, A.shape[0], chunk):
        S = A[i:i + chunk] @ B.T
        v, j = S.max(dim=1)
        best_ab[i:i + chunk], arg_ab[i:i + chunk] = v, j
        cv, ci = S.max(dim=0)
        upd = cv > best_ba
        best_ba[upd] = cv[upd]
        arg_ba[upd] = ci[upd] + i
        del S
    mutual = arg_ba[arg_ab] == torch.arange(A.shape[0], device=DEV)
    out = {}
    for t in thresholds:
        keep = mutual & (best_ab >= t)
        idx = torch.nonzero(keep).flatten()
        out[t] = (idx.cpu().numpy(), arg_ab[idx].cpu().numpy(),
                  best_ab[idx].cpu().numpy())
    return out, best_ab.cpu().numpy(), mutual.cpu().numpy()


@torch.no_grad()
def positions(sae, R, ids, n_pos, S, T):
    """Each dictionary's own top-n activating (sequence, position) pairs."""
    sel = torch.tensor(ids, dtype=torch.long, device=DEV)
    chunks = []
    for i in range(0, S, 8):
        x = R[i:i + 8].to(DEV).reshape(-1, R.shape[-1])
        chunks.append(sae.encode(x, sel).cpu())
    A = torch.cat(chunks).reshape(S, T, len(ids))
    tops, argmax = [], []
    for j in range(len(ids)):
        col = A[:, :, j].reshape(-1)
        n = min(n_pos, int((col > 0).sum()))
        if n == 0:
            tops.append(frozenset())
            argmax.append(None)
            continue
        tk = torch.topk(col, n)
        cells = [(int(ix) // T, int(ix) % T) for ix in tk.indices]
        tops.append(frozenset(cells))
        argmax.append(cells[0])
    return tops, argmax


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
    ap.add_argument("--pairs", default=None,
                    help="override the suite's pairs, as "
                         "'left|right|label;left|right|label'. Needed for the "
                         "depth sweep: only layer 12 (2B) and layer 20 (9B) "
                         "publish the 16k/32k/65k grid, so other layers are "
                         "compared on same-width 16k pairs at matched L0 "
                         "ratios instead.")
    ap.add_argument("--device_map", default=None,
                    help="pass 'auto' for models too large for one device; "
                         "left unset the model is placed on DEV as before, "
                         "which is what reproduces the 2B numbers exactly")
    ap.add_argument("--out", default=None,
                    help="path for the .txt; the .csv follows the same stem")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing an existing results CSV")
    a = ap.parse_args()
    cfg = dict(SUITES[a.suite])
    if a.layer is not None:
        cfg["layer"] = a.layer
    out_path = a.out or f"results/released_agreement_{a.suite}.txt"
    csv_path = re.sub(r"\.txt$", "", out_path) + ".csv"
    # Checked BEFORE the model loads, not after the run: a refusal that fires at
    # the end costs the whole run. Refusing at all because the CSV path used to
    # be hardcoded per suite, so a depth leg silently overwrote the headline
    # band run's output and left the caller to move it afterwards.
    assert not os.path.exists(csv_path) or a.overwrite, (
        f"{csv_path} exists; pass --overwrite to replace it. Refusing so a "
        f"completed run cannot be clobbered by a later one.")
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if a.device_map:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], dtype=torch.bfloat16,
            device_map=a.device_map).eval()
    else:
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
    ids_all = torch.cat(seqs, 0)
    S = ids_all.shape[0]
    R = []
    with torch.no_grad():
        for i in range(0, S, 8):
            R.append(model(input_ids=ids_all[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[cfg["layer"] + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)
    T = R.shape[1]
    del model
    torch.cuda.empty_cache()
    p(f"suite {a.suite}: {cfg['model']} layer {cfg['layer']}, "
      f"residual {tuple(R.shape)}")
    p("")

    THR = (0.5, 0.6, 0.7, 0.8, 0.9)
    rows = []
    pairs = PAIRS[a.suite]
    if a.pairs:
        pairs = [tuple(x.split("|")) for x in a.pairs.split(";") if x.strip()]
        assert all(len(x) == 3 for x in pairs), "pairs need left|right|label"
    for left, right, label in pairs:
        p("=" * 84)
        p(f"{label}   ({left}  vs  {right})")
        p("=" * 84)
        A = JumpReLU(hf_hub_download(cfg["repo"], f"layer_{cfg['layer']}/{left}/params.npz"))
        B = JumpReLU(hf_hub_download(cfg["repo"], f"layer_{cfg['layer']}/{right}/params.npz"))
        p(f"  widths {A.width} and {B.width}")
        m, best, mutual = match(A, B, THR)
        p(f"  mutual nearest neighbours: {int(mutual.sum())} of {A.width}")
        p(f"{'cosine >=':>12}{'matched pairs':>16}{'% of left dict':>16}")
        for t in THR:
            n = len(m[t][0])
            p(f"{t:>12.2f}{n:>16}{100.0 * n / A.width:>15.1f}%")

        # Two thresholds, not one. The upper-bound claim -- that matched pairs
        # agree about position MORE than typical pairs, because they are the
        # pairs whose decoders already agree -- is an assertion until the
        # matching is loosened and agreement is seen to fall. 0.5 admits pairs
        # that are similar rather than near-identical; if agreement drops, the
        # bound is demonstrated rather than asserted.
        # A BAND, not only thresholds. Lowering a threshold enlarges the
        # eligible pool but the sample stays dominated by near-identical pairs:
        # dropping from 0.70 to 0.50 moved the median cosine only 0.896 -> 0.865,
        # so agreement falling across that step is a direction, not a bound.
        # Sampling INSIDE 0.5-0.6 puts the median where the loosening is
        # supposed to be and makes the upper-bound claim quantitative.
        p("")
        p(f"{'matching':>12}{'matched':>9}{'median cos':>12}{'firing both':>13}"
          f"{'Jaccard':>10}{'top-1 agree':>13}")
        # Five DISJOINT bands plus the two cumulative thresholds. Disjoint is
        # what makes this a gradient rather than three overlapping samples:
        # each row's median cosine sits inside its own band, so agreement can
        # be read against similarity instead of against a cut-off.
        bands = [("0.50-0.60", 0.50, 0.60), ("0.60-0.70", 0.60, 0.70),
                 ("0.70-0.80", 0.70, 0.80), ("0.80-0.90", 0.80, 0.90),
                 ("0.90-1.00", 0.90, 1.01),
                 (">= 0.50", 0.50, 1.01), (">= %.2f" % a.cos, a.cos, 1.01)]
        for lab, lo_c, hi_c in bands:
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
              f"{np.mean(jac):>10.3f}{100.0 * np.mean(agree):>12.1f}%")
            rows.append(dict(suite=a.suite, pair=label, left=left, right=right,
                             width_left=A.width, width_right=B.width,
                             matching=lab, cos_lo=lo_c, cos_hi=hi_c,
                             n_matched=len(ia), n_live=live,
                             median_cos=float(np.median(cs)),
                             jaccard=float(np.mean(jac)),
                             top_agree=100.0 * float(np.mean(agree))))
        p("")
        del A, B
        torch.cuda.empty_cache()

    p("=" * 84)
    p("Matched pairs are the latents the two dictionaries agree MOST about --")
    p("near-identical decoder directions -- so these figures are an UPPER bound")
    p("on agreement across the full dictionaries, not a typical value.")
    if rows:
        # The CSV path follows --out. It used to be hardcoded per suite while
        # --out controlled only the .txt, so every depth leg (which is this
        # script at --suite 2b, one layer at a time) wrote over the headline
        # band run's CSV and relied on the caller moving it afterwards. The
        # headline file is the night's most important output and it does not
        # belong at a path another job writes to.
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")
    with io.open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
