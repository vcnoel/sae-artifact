# -*- coding: utf-8 -*-
"""PREREG E7 remainders, on the nine-dictionary sweep. CPU only.

(A) SATURATION / rate-vs-magnitude, with the marginal-matched control.
    Prediction: alignment predicts KL at rho ~ 0.375 and the argmax-flip rate at
    rho ~ 0.1, because a rate is a threshold crossing and saturates.
    REQUIRED CONTROL: flip has a low positive rate (~2-11%), so binarising KL at
    its median would compare two binary readouts with different marginals and any
    rho difference would be information loss, not saturation. So KL is thresholded
    at whatever cutoff reproduces flip's OWN positive rate, per dictionary. Only
    then is a rho difference attributable to saturation.

(B) ORTHOGRAPHY GENERALITY. Logit-lens each dictionary's top latent by causal
    mass. If all nine are orthographic / token-identity, the feature-14119 exhibit
    generalises; if heterogeneous, it is an anecdote and the direct-path argument
    rests on the correlation alone. Wordlike fraction (alphabetic, len>=3) is the
    discriminator -- n-gram overlap missed the punctuation class.
"""
import glob
import io
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from scipy import stats
from transformers import AutoTokenizer

torch.set_num_threads(8)
TOPK = 10


def wordlike(s):
    t = s.strip()
    return 1.0 if (len(t) >= 3 and t.isalpha()) else 0.0


def main():
    d = pd.read_csv("results/multi_dict.csv")
    # embedding + covariance once: alignment is then cheap for any decoder
    snap = glob.glob("C:/Users/valno/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    E = None
    for fp in sorted(glob.glob(snap + "*.safetensors")):
        t = load_file(fp)
        k = next((k for k in t if "embed_tokens" in k), None)
        if k is not None:
            E = t[k].float()
            break
    dm = E.shape[1]
    s1 = torch.zeros(dm, dtype=torch.float64)
    G = torch.zeros(dm, dm, dtype=torch.float64)
    n = 0
    for a in range(0, E.shape[0], 16384):
        B = E[a:a + 16384].double()
        s1 += B.sum(0)
        G += B.T @ B
        n += B.shape[0]
    mu = s1 / n
    C = (G / n - torch.outer(mu, mu)).float()
    En = E / E.norm(dim=-1, keepdim=True)
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    rowsA, rowsB = [], []
    for tag, g in d.groupby("dict_tag"):
        layer = int(g.layer.iloc[0])
        width = g.width.iloc[0]
        l0 = int(g.l0.iloc[0])
        z = np.load(hf_hub_download(
            "google/gemma-scope-2b-pt-res",
            f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz"))
        W = torch.tensor(z["W_dec"], dtype=torch.float32)
        W = W / W.norm(dim=-1, keepdim=True)

        f = g.groupby("fid").agg(kl=("kl_t0", "median"),
                                 kpn=("kpn_t0", "median"),
                                 flip=("flip_t0", "mean")).reset_index()
        ids = torch.tensor(f.fid.values, dtype=torch.long)
        Dsub = W[ids]
        f["ua"] = torch.sqrt(torch.clamp((Dsub @ C * Dsub).sum(-1),
                                         min=0)).numpy()

        # (A) rho against a MAGNITUDE readout and a marginal-matched RATE readout
        rate = float((g.flip_t0 > 0).mean())
        thr = np.quantile(f.kpn.values, 1 - rate) if 0 < rate < 1 else np.inf
        f["kl_bin"] = (f.kpn.values >= thr).astype(float)
        r_mag = stats.spearmanr(f.kpn, f["ua"])
        r_rate = stats.spearmanr(f.flip, f["ua"])
        r_binm = stats.spearmanr(f["kl_bin"], f["ua"])
        rowsA.append(dict(tag=tag, rate=rate, rho_kl=r_mag.statistic,
                          p_kl=r_mag.pvalue, rho_flip=r_rate.statistic,
                          p_flip=r_rate.pvalue, rho_klbin=r_binm.statistic))

        # (B) top latent by causal mass, logit-lensed
        top = int(f.sort_values("kpn", ascending=False).fid.iloc[0])
        proj = En @ W[top]
        toks = [tok.decode([int(i)]) for i in torch.topk(proj, TOPK).indices]
        rowsB.append(dict(tag=tag, fid=top,
                          wl=float(np.mean([wordlike(x) for x in toks])),
                          toks=" ".join(repr(x) for x in toks[:8])))
        del W, z

    A = pd.DataFrame(rowsA)
    print("=" * 100)
    print("(A) SATURATION: does alignment predict a MAGNITUDE readout more than "
          "a marginal-matched RATE readout?")
    print("=" * 100)
    print(f"{'dictionary':>16} {'flip rate':>10} {'rho(KL)':>9} {'rho(flip)':>10} "
          f"{'rho(KL binarised at same rate)':>32}")
    for _, r in A.iterrows():
        print(f"{r.tag:>16} {r.rate:>10.3f} {r.rho_kl:>+9.3f} "
              f"{r.rho_flip:>+10.3f} {r.rho_klbin:>+32.3f}")
    print(f"\n  MEDIAN  rho(KL)={A.rho_kl.median():+.3f}  "
          f"rho(flip)={A.rho_flip.median():+.3f}  "
          f"rho(KL binarised)={A.rho_klbin.median():+.3f}")
    print("  Saturation is supported only if rho(flip) < rho(KL binarised),")
    print("  i.e. the drop is not explained by coarsening at the same marginal.")

    B = pd.DataFrame(rowsB)
    print()
    print("=" * 100)
    print("(B) ORTHOGRAPHY: top latent by causal mass, per dictionary")
    print("=" * 100)
    for _, r in B.iterrows():
        print(f"{r.tag:>16} fid={r.fid:>6} wordlike={r.wl:.2f} | {r.toks}")
    print(f"\n  wordlike fraction across the nine top latents: "
          f"median={B.wl.median():.2f} mean={B.wl.mean():.2f}  "
          f"(low = orthographic/token-identity, as feature 14119 was)")
    A.to_csv("results/sweep_saturation.csv", index=False)
    B.to_csv("results/sweep_orthography.csv", index=False)


if __name__ == "__main__":
    main()
