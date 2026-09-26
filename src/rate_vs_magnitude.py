# -*- coding: utf-8 -*-
"""E14: does a RATE readout resist the alignment sensitivity that KL does not?

THE CLAIM UNDER TEST (OUTLINE, "MAGNITUDE vs RATE"). Direct-path unembedding
alignment scales a MAGNITUDE readout (KL, logit difference) by construction: the
response grows with how much of d_f the unembedding sees. A RATE readout (did
the argmax change?) is a threshold crossing, and thresholds saturate -- once the
intervention flips the token, more alignment buys nothing. So alignment should
predict KL more strongly than it predicts flip rate.

  rho(align, KL) >> rho(align, flip)  -> the protocol can RECOMMEND a readout
                                         that resists the confound, not merely
                                         diagnose one that does not
  rho(align, KL) ~= rho(align, flip)  -> the recommendation dies; say so, and
                                         the catalogue stands on its own

PRE-REGISTERED BEFORE RUNNING. multi_dict.csv already carried flip_t/dtop_t
columns computed from the same forward passes as kl_t -- the test was set up
and never run. Nothing about the measurement changes here; this only joins the
existing rate columns to alignment, which was never computed for these nine
dictionaries.

Note the direction of the prediction is about MAGNITUDE of correlation, and
dtop (probability drop of the original top token) is deliberately included as an
intermediate case: it is a magnitude, but a bounded one, so it should sit
between KL and flip if saturation is the mechanism.
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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy import stats

torch.set_num_threads(8)


def load_embed():
    """The embedding matrix, once. unembed_align.py restreams every shard per
    call; here there are nine dictionaries, so re-reading ~5GB of shards nine
    times is both slow and enough memory churn to get the process killed while
    a training job holds the rest of RAM."""
    import glob
    import os
    p = hf_hub_download("google/gemma-2-2b", "model.safetensors.index.json")
    for f in sorted(glob.glob(os.path.join(os.path.dirname(p),
                                           "*.safetensors"))):
        t = load_file(f)
        key = next((k for k in t if "embed_tokens" in k), None)
        if key is not None:
            E = t[key].float().contiguous()
            del t
            return E
        del t
    raise SystemExit("embed_tokens not found in any shard")


def align_of(E, D, chunk=16384):
    """std over vocab of (E @ d) for each column of D. Same measure as
    unembed_align.py -- spread, not norm, since a uniform logit shift leaves
    the softmax unchanged."""
    n = D.shape[1]
    s1 = torch.zeros(n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    cnt = 0
    for a in range(0, E.shape[0], chunk):
        V = E[a:a + chunk] @ D
        s1 += V.sum(0).double()
        s2 += (V ** 2).sum(0).double()
        cnt += V.shape[0]
    mean = s1 / cnt
    var = s2 / cnt - mean ** 2
    return torch.sqrt(var.clamp_min(0)).float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/multi_dict.csv")
    ap.add_argument("--out", default="results/rate_vs_magnitude.txt")
    ap.add_argument("--csv", default="results/rate_vs_magnitude.csv")
    A = ap.parse_args()

    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    s = pd.read_csv(A.src)
    E = load_embed()
    print(f"  embed {tuple(E.shape)}", flush=True)

    # dtop_t0 is a SIGNED difference p_clean[t0] - p_ablated[t0], and 46% of
    # values are negative -- ablating a latent often RAISES the original top
    # token's probability. A rank correlation against a signed, near-symmetric
    # quantity measures direction, not magnitude, and its per-latent median
    # sits at ~0. The bounded-magnitude readout is |dtop|; the signed version
    # is kept alongside so the distinction is visible rather than assumed.
    s["adtop_t0"] = s["dtop_t0"].abs()

    # per-latent aggregate: median magnitude readouts, MEAN of the rate readout
    # (a rate is a proportion over the measured positions, not a median)
    g = (s.groupby(["dict_tag", "layer", "width", "l0", "fid"])
          .agg(kpn=("kpn_t0", "median"), kl=("kl_t0", "median"),
               flip=("flip_t0", "mean"), dtop=("dtop_t0", "median"),
               adtop=("adtop_t0", "median"), n=("kl_t0", "size"))
          .reset_index())

    rows = []
    for (tag, layer, width, l0), sub in g.groupby(
            ["dict_tag", "layer", "width", "l0"], sort=False):
        z = np.load(hf_hub_download(
            "google/gemma-scope-2b-pt-res",
            f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"))
        # keep only the sampled rows: the 131k dictionary is 1.2GB in full and
        # only 300 of its directions are ever needed here
        W = torch.tensor(z["W_dec"][sub.fid.values], dtype=torch.float32)
        del z
        W = W / W.norm(dim=-1, keepdim=True)
        D = W.T.contiguous()
        sub = sub.copy()
        sub["align"] = align_of(E, D)
        rows.append(sub)
        print(f"  {tag}: aligned {len(sub)} latents", flush=True)
        del W, D

    a = pd.concat(rows, ignore_index=True)
    a.to_csv(A.csv, index=False)

    p("=" * 84)
    p("E14  RATE vs MAGNITUDE READOUTS UNDER THE ALIGNMENT SENSITIVITY")
    p("=" * 84)
    p(f"{'dictionary':<16}{'n':>5}{'rho(al,KL/nrm)':>16}{'rho(al,KLraw)':>15}"
      f"{'rho(al,|dtop|)':>16}{'rho(al,dtop)':>14}{'rho(al,flip)':>14}"
      f"{'fliprate':>10}")
    # NB: bracket access throughout -- `df.align` is a DataFrame METHOD, so
    # attribute access on this column silently yields a bound method.
    for tag, sub in a.groupby("dict_tag", sort=False):
        vals = [stats.spearmanr(sub["align"], sub[c]).statistic
                for c in ("kpn", "kl", "adtop", "dtop", "flip")]
        p(f"{tag:<16}{len(sub):>5}" + "".join(
            f"{v:>+16.3f}" if i == 0 else
            f"{v:>+15.3f}" if i == 1 else
            f"{v:>+16.3f}" if i == 2 else
            f"{v:>+14.3f}" for i, v in enumerate(vals))
          + f"{sub['flip'].mean():>10.3f}")

    # pooled, within-dictionary (ranks taken inside each dictionary so the
    # comparison is not driven by between-dictionary differences)
    a["z_al"] = a.groupby("dict_tag")["align"].rank(pct=True)
    READOUTS = (("kpn", "KL per unit norm (MAG)"),
                ("kl", "KL raw (MAG)"),
                ("adtop", "|prob change of top tok| (MAG, bounded)"),
                ("dtop", "signed prob change of top tok (NOT a magnitude)"),
                ("flip", "argmax flipped (RATE)"))
    for c, _ in READOUTS:
        a[f"z_{c}"] = a.groupby("dict_tag")[c].rank(pct=True)
    p("")
    pooled = {}
    for c, lab in READOUTS:
        r = stats.spearmanr(a["z_al"], a[f"z_{c}"])
        pooled[c] = r.statistic
        p(f"  pooled within-dict rho(align, {lab:<48}) = "
          f"{r.statistic:+.3f}  (p={r.pvalue:.2g}, n={len(a)})")

    # The alignment sensitivity is itself depth-dependent (shallow rho ~0.35,
    # deep ~0.09), so pooling over layer 19 -- where alignment predicts nothing
    # for EITHER readout -- inflates the KL/flip ratio by diluting only the
    # numerator. The layer-restricted comparison is the honest one.
    p("")
    p("  restricted to layer 12, where the alignment sensitivity actually "
      "exists:")
    sub12 = a[a.layer == 12]
    pooled12 = {}
    z_al = sub12.groupby("dict_tag")["align"].rank(pct=True)
    for c, lab in READOUTS:
        z_c = sub12.groupby("dict_tag")[c].rank(pct=True)
        r = stats.spearmanr(z_al, z_c)
        pooled12[c] = r.statistic
        p(f"    rho(align, {lab:<48}) = {r.statistic:+.3f}  "
          f"(p={r.pvalue:.2g}, n={len(sub12)})")
    p(f"    |rho_KL| / |rho_flip| (layer 12 only) = "
      f"{abs(pooled12['kpn']) / max(abs(pooled12['flip']), 1e-9):.2f}")

    p("")
    ratio = abs(pooled["kpn"]) / max(abs(pooled["flip"]), 1e-9)
    p(f"  |rho_KL| / |rho_flip| = {ratio:.2f}")
    if abs(pooled["flip"]) < 0.5 * abs(pooled["kpn"]):
        p("  VERDICT: the rate readout is substantially less sensitive to")
        p("  alignment than the magnitude readout. The protocol can recommend a")
        p("  readout, not just flag a confound.")
    else:
        p("  VERDICT: the rate readout is NOT meaningfully less sensitive.")
        p("  The magnitude-vs-rate recommendation does not survive; report the")
        p("  catalogue without it. (Pre-registered as a possible outcome.)")
    p("")
    p("  Caveat 1: flip rate is bounded and its variance collapses when the")
    p("  mean rate is near 0 or 1. Measured flip rates run 0.017-0.110, i.e.")
    p("  the rate readout is a RARE event here, so it has genuinely less power")
    p("  than KL and part of the smaller rho is that, not saturation. This is")
    p("  the main threat to the recommendation and is not resolved by the data")
    p("  in hand -- a design with flip rates near 0.5 would settle it.")
    p("")
    p("  Caveat 2: the pre-registered ORDERING was KL > bounded-magnitude >")
    p("  flip. Read against |dtop| -- the actual bounded magnitude -- see the")
    p("  table above. The SIGNED dtop column is reported only to show why it")
    p("  must not be used: 46% of its values are negative, so a rank")
    p("  correlation against it measures direction rather than size and its")
    p("  per-latent median is ~0. An earlier version of this analysis used the")
    p("  signed column and read its near-zero correlation as evidence about")
    p("  saturation; it was evidence about signs.")
    p("")
    p("  Caveat 3: this comparison is not fully like-for-like on normalisation.")
    p("  kpn is per unit perturbation norm; flip and |dtop| are not normalised,")
    p("  and a rate cannot be. Raw KL is shown alongside kpn so the size of that")
    p("  gap is visible rather than hidden -- note rho(pnorm, KL_raw)=0.70,")
    p("  i.e. raw KL partly tracks intervention size (Artifact 2).")

    with open(A.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out} and {A.csv}")


if __name__ == "__main__":
    main()
