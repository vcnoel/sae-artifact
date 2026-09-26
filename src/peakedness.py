# -*- coding: utf-8 -*-
"""E27: are our arms' position-activation profiles flatter than a released one?

The steelman of the undertraining objection is not "your SAEs are bad". It is
mechanical: an undertrained dictionary has a flatter activation profile across
positions, so its argmax is less determined, so two such dictionaries disagree
about where to measure for a reason that would not apply at production scale.
That predicts something measurable, and this measures it.

For each latent, over the positions where it fires, we report two shape
statistics of the profile:

  peak ratio      max / second-max, per (latent, sequence). At 1.0 the top two
                  positions are tied and the argmax is arbitrary; large values
                  mean one position dominates.
  entropy         Shannon entropy of the activation profile over positions,
                  normalised by log(n_firing), so 1.0 is perfectly flat.

Both are computed per (latent, sequence) and then reduced per latent, because a
latent that fires on many sequences would otherwise be compared against one
that fires on few.

MATCHED SPARSITY. Our arms are TopK at k=82; the released comparator is Gemma
Scope width 16k at average L0 82, the closest released sparsity. A comparison
against a much sparser or denser release would confound peakedness with L0.

WHAT WOULD FALSIFY THE PAPER'S POSITION. If our arms are flatter -- lower peak
ratio, higher entropy -- the undertraining objection has a mechanism and the
measured disagreement is partly ours. If the released dictionary's profiles are
equally flat, the objection is closed: unstable argmax is a property of SAE
latents on real text, not of a 12M-token budget.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import model_configs
from eval_arms import load_arms, gated
from eval_saes import WIKI
from released_agreement import JumpReLU

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GS = dict(repo="google/gemma-scope-2b-pt-res",
          path="layer_12/width_16k/average_l0_82/params.npz")


def shape_stats(A, min_fire=3):
    """A is (S, T, F) gated activations. Returns per-latent medians.

    A (latent, sequence) row is used only when the latent fires at min_fire or
    more positions there: max-over-second-max is undefined at one firing and
    degenerate at two, and including those would report the sparsity of the
    dictionary rather than the shape of its profile.
    """
    S, T, F = A.shape
    peak, ent, rows_used = [], [], []
    for j in range(F):
        M = A[:, :, j]                                   # S x T
        pk, en = [], []
        for s in range(S):
            v = M[s]
            v = v[v > 0]
            if v.numel() < min_fire:
                continue
            top2 = torch.topk(v, 2).values
            pk.append(float(top2[0] / top2[1].clamp_min(1e-12)))
            q = (v / v.sum()).clamp_min(1e-12)
            en.append(float(-(q * q.log()).sum() / np.log(v.numel())))
        if not pk:
            continue
        peak.append(float(np.median(pk)))
        ent.append(float(np.median(en)))
        rows_used.append(len(pk))
    return np.array(peak), np.array(ent), np.array(rows_used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma2-2b")
    ap.add_argument("--n_seq", type=int, default=384)
    ap.add_argument("--n_feat", type=int, default=240)
    ap.add_argument("--out", default="results/peakedness.txt")
    a = ap.parse_args()
    cfg = model_configs.get(a.base)
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16).to(DEV).eval()
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
    p(f"{cfg['model']} layer {cfg['layer']}, residual {tuple(R.shape)}")
    p("")

    def live_sample(enc):
        cnt = torch.zeros(cfg["width"], device=DEV)
        with torch.no_grad():
            for i in range(0, S, 8):
                A = enc(R[i:i + 8].to(DEV).reshape(-1, cfg["d_model"]))
                cnt += (A > 0).float().sum(0)
                del A
        rate = (cnt / (S * T)).cpu().numpy()
        live = np.where(rate > 1e-5)[0]
        order = np.random.default_rng(0).permutation(cfg["width"])
        ls = set(live.tolist())
        return np.sort(np.array([i for i in order if i in ls][:a.n_feat]))

    def acts_for(enc_sel, ids):
        sel = torch.tensor(ids, dtype=torch.long, device=DEV)
        ch = []
        with torch.no_grad():
            for i in range(0, S, 8):
                x = R[i:i + 8].to(DEV).reshape(-1, cfg["d_model"])
                ch.append(enc_sel(x, sel).cpu())
        return torch.cat(ch).reshape(S, T, len(ids))

    rows = []
    p(f"{'dictionary':<22}{'lat':>5}{'peak ratio (max/2nd)':>24}"
      f"{'entropy (1=flat)':>20}")
    p(f"{'':<22}{'':>5}{'median':>9}{'q25':>7}{'q75':>8}"
      f"{'median':>9}{'q25':>6}{'q75':>6}")

    todo = [(f"ours: {t}", (lambda m: (lambda x: m.encode(x)))(m),
             (lambda m: (lambda x, s: gated(m, x, s)))(m))
            for t, m in arms.items()]
    gs = JumpReLU(hf_hub_download(GS["repo"], GS["path"]))
    todo.append(("gemma-scope 16k L0 82", lambda x: gs.encode(x),
                 lambda x, s: gs.encode(x, s)))

    for label, enc_all, enc_sel in todo:
        ids = live_sample(enc_all)
        A = acts_for(enc_sel, ids)
        pk, en, used = shape_stats(A)
        del A
        torch.cuda.empty_cache()
        p(f"{label:<22}{len(pk):>5}"
          f"{np.median(pk):>9.2f}{np.percentile(pk, 25):>7.2f}"
          f"{np.percentile(pk, 75):>8.2f}"
          f"{np.median(en):>9.3f}{np.percentile(en, 25):>6.3f}"
          f"{np.percentile(en, 75):>6.3f}")
        rows.append(dict(dictionary=label, n_latents=len(pk),
                         peak_median=float(np.median(pk)),
                         peak_q25=float(np.percentile(pk, 25)),
                         peak_q75=float(np.percentile(pk, 75)),
                         ent_median=float(np.median(en)),
                         ent_q25=float(np.percentile(en, 25)),
                         ent_q75=float(np.percentile(en, 75)),
                         seq_rows_median=float(np.median(used))))

    d = pd.DataFrame(rows)
    ours = d[d.dictionary.str.startswith("ours")]
    rel = d[~d.dictionary.str.startswith("ours")]
    p("")
    p(f"  our arms, peak ratio median across arms = "
      f"{ours.peak_median.median():.2f} "
      f"(range {ours.peak_median.min():.2f}-{ours.peak_median.max():.2f})")
    p(f"  released comparator                     = "
      f"{float(rel.peak_median.iloc[0]):.2f}")
    p(f"  our arms, entropy median across arms    = "
      f"{ours.ent_median.median():.3f} "
      f"(range {ours.ent_median.min():.3f}-{ours.ent_median.max():.3f})")
    p(f"  released comparator                     = "
      f"{float(rel.ent_median.iloc[0]):.3f}")
    p("")
    flatter = (ours.peak_median.median() < float(rel.peak_median.iloc[0])
               and ours.ent_median.median() > float(rel.ent_median.iloc[0]))
    if flatter:
        p("  OUR ARMS ARE FLATTER on both statistics. The undertraining")
        p("  mechanism has support and the measured disagreement is partly a")
        p("  property of our training budget. Say so in the limitation.")
    else:
        p("  Our arms are NOT flatter on both statistics. The undertraining")
        p("  mechanism does not have the profile-shape support it needs, and")
        p("  unstable argmax is a property of SAE latents on real text rather")
        p("  than of a 12M-token budget.")
    d.to_csv("results/peakedness.csv", index=False)
    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
