# -*- coding: utf-8 -*-
"""Causal-effect comparison: trained vs SOFT-FROZEN SAE, matched budget.

The question is not whether trained beats frozen on average -- the Sanity Checks
paper already reports that it barely does (RAVEL 0.72 vs 0.73). The question is
whether that near-tie is a property of the dictionaries or of the statistic used
to compare them. So the same aggregate is reported alongside the spread, the
tail mass, and the fraction of features exceeding the other arm's 99th
percentile.

Metric: KL(clean || ablated) on the next-token distribution, ablating one
feature's contribution at its own firing position. Reported raw and per unit
perturbation norm, because the two arms' activations differ in scale and the raw
number would otherwise reward magnitude rather than structure. That control
already overturned one of my own conclusions on the Gemma Scope run.
"""
import argparse
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_saes import (D_MODEL, DEV, K, LAYER, MODEL, SEQ, WIDTH, TopKSAE)

WIKI = ("C:/Users/valno/.cache/huggingface/hub/datasets--wikitext/snapshots/"
        "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
        "train-00000-of-00001.parquet")


@torch.no_grad()
def kl_batch(model, ids, pos, vecs, bs=6):
    state = {"on": False, "pos": None, "vec": None}

    def hook(mod, inp, out):
        if not state["on"]:
            return out
        h = (out[0] if isinstance(out, tuple) else out).clone()
        for r, (pp, vv) in enumerate(zip(state["pos"], state["vec"])):
            h[r, pp] = h[r, pp] - vv.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    hd = model.model.layers[LAYER].register_forward_hook(hook)
    try:
        out = []
        for i in range(0, len(ids), bs):
            b = ids[i:i + bs].to(DEV)
            p_, v_ = pos[i:i + bs], vecs[i:i + bs]
            state["on"] = False
            lc = model(input_ids=b).logits.float()
            state.update(on=True, pos=p_, vec=v_)
            la = model(input_ids=b).logits.float()
            state["on"] = False
            for r, pp in enumerate(p_):
                lp = torch.log_softmax(lc[r, pp], -1)
                lq = torch.log_softmax(la[r, pp], -1)
                out.append(float((lp.exp() * (lp - lq)).sum().item()))
        return out
    finally:
        hd.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="saes.pt")
    ap.add_argument("--n_seq", type=int, default=96)
    ap.add_argument("--per_bin", type=int, default=40)
    ap.add_argument("--n_pos", type=int, default=6)
    ap.add_argument("--out", default="eval_saes.csv")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(DEV).eval()
    sd = torch.load(a.ckpt, map_location=DEV)
    arms = {}
    for n in ("trained", "frozen"):
        m = TopKSAE(D_MODEL, WIDTH, K, 0).to(DEV)
        m.load_state_dict(sd[n])
        arms[n] = m.eval()

    txt = [str(t).strip() for t in pd.read_parquet(WIKI)["text"].tolist()
           if len(str(t).strip()) > 400 and not str(t).strip().startswith("=")]
    seqs, buf = [], ""
    for t in txt:
        buf += " " + t
        ids = tok(buf, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if ids.shape[1] >= SEQ:
            seqs.append(ids[:, :SEQ])
            buf = ""
            if len(seqs) >= a.n_seq:
                break
    ids_all = torch.cat(seqs, 0)
    S, T = ids_all.shape

    R = []
    with torch.no_grad():
        for i in range(0, S, 8):
            R.append(model(input_ids=ids_all[i:i + 8].to(DEV),
                           output_hidden_states=True)
                     .hidden_states[LAYER + 1][:, 1:, :].float().cpu())
    R = torch.cat(R)                                        # [S, T-1, d]
    print(f"resid {tuple(R.shape)}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for name, m in arms.items():
        with torch.no_grad():
            A = []
            for i in range(0, S, 8):
                A.append(m.encode(R[i:i + 8].to(DEV).reshape(-1, D_MODEL))
                         .cpu())
            A = torch.cat(A).reshape(S, T - 1, WIDTH)
            ev = None
            xb = R[:8].to(DEV).reshape(-1, D_MODEL)
            rec, _ = m(xb)
            ev = 1 - float((xb - rec).var() / xb.var())
        freq = (A > 0).float().mean((0, 1)).numpy()
        live = np.where(freq > 1e-5)[0]
        print(f"{name}: EV={ev:.3f} live={len(live)}/{WIDTH}", flush=True)
        lf = np.log10(freq[live])
        qs = np.quantile(lf, np.linspace(0, 1, 7))
        for b in range(6):
            pool = live[(lf >= qs[b]) & (lf <= qs[b + 1])]
            if len(pool) == 0:
                continue
            for fid in rng.choice(pool, size=min(a.per_bin, len(pool)),
                                  replace=False):
                col = A[:, :, int(fid)].reshape(-1)
                n = min(a.n_pos, int((col > 0).sum()))
                if n == 0:
                    continue
                top = torch.topk(col, n)
                best = [(float(v), int(ix) // (T - 1), int(ix) % (T - 1) + 1)
                        for v, ix in zip(top.values, top.indices)]
                dvec = m.W_dec.data[int(fid)]
                ib = torch.stack([ids_all[i] for _, i, _ in best])
                pos = [p for _, _, p in best]
                vecs = [float(v) * dvec for v, _, _ in best]
                kls = kl_batch(model, ib, pos, vecs)
                for kl, (v, _, _) in zip(kls, best):
                    pn = float((v * dvec).norm().item())
                    rows.append(dict(arm=name, fid=int(fid), bin=b,
                                     freq=float(freq[fid]), kl=kl, pnorm=pn,
                                     kl_per_norm=kl / max(pn, 1e-6)))
        print(f"  {name} done, rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    f = df.groupby(["arm", "fid", "bin"]).agg(
        kl=("kl", "median"), kpn=("kl_per_norm", "median"),
        pn=("pnorm", "median")).reset_index()
    print("\n" + "=" * 74)
    print("TRAINED vs SOFT-FROZEN (their baseline), feature-level")
    print("=" * 74)
    for col in ("kl", "kpn"):
        t, r = f[f.arm == "trained"][col], f[f.arm == "frozen"][col]
        q99 = r.quantile(.99)
        print(f"\n[{col}]  n_t={len(t)} n_f={len(r)}")
        print(f"  median  trained={t.median():.6f}  frozen={r.median():.6f}"
              f"   ratio={t.median()/max(r.median(),1e-12):.2f}")
        print(f"  sd      trained={t.std():.6f}  frozen={r.std():.6f}"
              f"   ratio={t.std()/max(r.std(),1e-12):.1f}x")
        print(f"  max     trained={t.max():.5f}  frozen={r.max():.5f}")
        print(f"  trained above frozen p99: {(t>q99).mean():.1%}  (1% by "
              f"construction)")
        s = t.sort_values(ascending=False)
        print(f"  top-5% of trained features hold "
              f"{s.head(max(1,len(s)//20)).sum()/s.sum():.1%} of total mass")
        # Brown-Forsythe (median-centred) rather than mean-centred Levene: with
        # a ~40x SD ratio driven by a handful of extreme features, the
        # mean-centred test is not trustworthy. The tail statistic above is the
        # instrument to lead with; this is reported as a secondary check only.
        print(f"  Mann-Whitney p={stats.mannwhitneyu(t,r).pvalue:.2e}   "
              f"Brown-Forsythe p="
              f"{stats.levene(t, r, center='median').pvalue:.2e}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
