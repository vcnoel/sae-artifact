# -*- coding: utf-8 -*-
"""The headline number: Hodges-Lehmann on the ALIGNMENT-CONTROLLED trained-vs-
soft-frozen comparison. CPU only.

Why this and not what I had. HL=1.209 [1.032,1.424] was computed on raw kpn, but
alignment predicts causal mass at rho=+0.264 in the trained arm and +0.004 in
frozen -- the same asymmetry that licensed one-sided adjustment everywhere else.
The matched-support figure I reported (1.13, [0.97,1.42]) was a ratio of medians,
not HL, so the paper's headline comparison has never been computed with the
estimator that matches its own test.

Matching is nearly vacuous here (237/240 trained latents already lie inside
frozen's alignment range, because that range is wide), so ADJUSTMENT is the real
control. Both are reported: matched HL, and HL after adjusting trained latents to
frozen's median alignment.
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
from safetensors.torch import load_file
from scipy import stats

torch.set_num_threads(8)
rng = np.random.default_rng(0)
B = 2000


def align_of(files, D, chunk=16384):
    n = D.shape[1]
    s1 = torch.zeros(n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    cnt = 0
    for fp in files:
        t = load_file(fp)
        key = next((k for k in t if "embed_tokens" in k), None)
        if key is None:
            continue
        E = t[key]
        for a in range(0, E.shape[0], chunk):
            V = E[a:a + chunk].float() @ D
            s1 += V.sum(0).double()
            s2 += (V ** 2).sum(0).double()
            cnt += V.shape[0]
        del t, E
    mu = s1 / cnt
    return torch.sqrt((s2 / cnt - mu ** 2).clamp_min(0)).float().numpy()


def hl_ratio(a, b):
    return float(10 ** np.median(np.log10(a)[:, None] - np.log10(b)[None, :]))


def report(t, r, label):
    pt = hl_ratio(t, r)
    boot = np.array([hl_ratio(rng.choice(t, len(t), True),
                              rng.choice(r, len(r), True)) for _ in range(B)])
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    p = stats.mannwhitneyu(t, r).pvalue
    flag = "excludes 1" if lo > 1 else "INCLUDES 1"
    print(f"  {label:<34} n_t={len(t):4d}  HL={pt:.3f}  "
          f"95% CI [{lo:.3f}, {hi:.3f}]  {flag}   MW p={p:.4f}")
    return pt, lo, hi


def main():
    df = pd.read_csv("results/eval_saes.csv")
    f = (df.groupby(["arm", "fid"])
         .agg(kpn=("kl_per_norm", "median")).reset_index())
    sd = torch.load("data/saes.pt", map_location="cpu")
    Ds, parts = [], []
    for arm in ("trained", "frozen"):
        w = sd[arm]["W_dec"].float()
        w = w / w.norm(dim=-1, keepdim=True)
        s = f[f.arm == arm].copy()
        Ds.append(w[torch.tensor(s.fid.values, dtype=torch.long)].T.contiguous())
        parts.append(s)
    snap = glob.glob("~/.cache/huggingface/hub/"
                     "models--google--gemma-2-2b/snapshots/*/")[0]
    al = align_of(sorted(glob.glob(snap + "*.safetensors")),
                  torch.cat(Ds, 1))
    n0 = Ds[0].shape[1]
    parts[0]["ua"], parts[1]["ua"] = al[:n0], al[n0:]
    T, R = parts[0], parts[1]
    r = R.kpn.to_numpy()

    print("alignment: trained med=%.5f [%.5f, %.5f]   frozen med=%.5f "
          "[%.5f, %.5f]" % (T['ua'].median(), T['ua'].min(), T['ua'].max(),
                            R['ua'].median(), R['ua'].min(), R['ua'].max()))
    a = stats.spearmanr(T.kpn, T['ua'])
    b = stats.spearmanr(R.kpn, R['ua'])
    print(f"rho(kpn, alignment): trained {a.statistic:+.3f} (p={a.pvalue:.4f})"
          f"  frozen {b.statistic:+.3f} (p={b.pvalue:.4f})")
    print()
    print("=" * 92)
    print("TRAINED vs SOFT-FROZEN -- Hodges-Lehmann, raw and alignment-controlled")
    print("=" * 92)
    report(T.kpn.to_numpy(), r, "raw")

    lo, hi = R['ua'].min(), R['ua'].max()
    ins = T[(T['ua'] >= lo) & (T['ua'] <= hi)]
    report(ins.kpn.to_numpy(), r, "matched on alignment support")

    y = np.log10(T.kpn.values)
    X = np.column_stack([np.ones(len(T)), T['ua'].values])
    bta = np.linalg.lstsq(X, y, rcond=None)[0]
    ref = float(R['ua'].median())
    adj = 10 ** (y - bta[1] * (T['ua'].values - ref))
    print(f"  (adjustment slope={bta[1]:+.2f}, trained moved to frozen median "
          f"alignment {ref:.5f})")
    report(adj, r, "alignment-ADJUSTED (primary)")

    # jackknife the adjusted headline
    print("\n  jackknife of the adjusted HL:")
    v = np.sort(adj)[::-1]
    for k in (1, 3, 10):
        print(f"    drop top-{k:<2d} HL={hl_ratio(v[k:], r):.3f}")


if __name__ == "__main__":
    main()
