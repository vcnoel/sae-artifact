# -*- coding: utf-8 -*-
"""E25: what are the five latents carrying 58% of the msa rise?

Three mechanisms for the v_a/msa rise have been tested and failed (E20.3
selection-on-outcome, E21c compression, E22 msab coupling). The jackknife (E24c)
showed the rise is concentrated: 5 latents of 70 carry 58%, 10 carry 79%. If
those five share an identifiable property, that is the last mechanism candidate
standing. If they share nothing, the effect is concentrated in latents with no
common property, which is itself reportable.

Compares the top-5 and top-10 carriers against the rest on: activation
percentile, causal effect, how far their measured positions moved between
designs, and decoder cosine to the rest of the sample. Also checks whether the
carrier sets overlap across the two base models -- they cannot share latent ids
meaningfully (different dictionaries), so the comparison is on properties.
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
from scipy import stats

from boundary_check import comp_cube
from moderator import MODELS, build

ARM_DIR = {"gemma2-2b": "data", "gemma3-1b": "data/g3"}


def decoder_cos(base, fids, carriers):
    """Mean |cos| of each carrier's decoder row to the rest of the sample."""
    try:
        import glob
        from train_saes import TopKSAE, DEV  # noqa: F401
        p = sorted(glob.glob(f"{ARM_DIR[base]}/arm_free.pt"))
        if not p:
            return None
        d = torch.load(p[0], map_location="cpu")
        W = d["state"]["W_dec"] if "W_dec" in d["state"] else None
        if W is None:
            return None
        W = W.float()
        W = W / W.norm(dim=-1, keepdim=True)
        idx = torch.tensor(list(fids), dtype=torch.long)
        M = W[idx]
        C = (M @ M.T).abs()
        C.fill_diagonal_(0.0)
        per = C.max(dim=1).values.numpy()
        return pd.Series(per, index=list(fids))
    except Exception as e:  # noqa: BLE001
        print(f"  [decoder cosine unavailable: {type(e).__name__}]")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/carriers.txt")
    A = ap.parse_args()
    out = []

    def p(s=""):
        print(s, flush=True)
        out.append(s)

    p("=" * 90)
    p("E25  WHAT ARE THE LATENTS CARRYING THE msa RISE?")
    p("=" * 90)

    for name, pap, shp in MODELS:
        pa, sh, ca, cs, fids = build(pap, shp)
        fids = np.array(fids)
        chg = cs.mean(axis=(1, 2)) - ca.mean(axis=(1, 2))
        order = np.argsort(-np.abs(chg))
        full = comp_cube(cs)["msa"] - comp_cube(ca)["msa"]
        p("")
        p(f"--- {name}  ({len(fids)} latents, full d msa = {full:+.3f}) ---")
        top5 = set(fids[order[:5]].tolist())
        top10 = set(fids[order[:10]].tolist())
        rest = set(fids.tolist()) - top10

        # per-latent descriptors from the per-arm eval
        g = (pa[pa.fid.isin(fids)].groupby("fid")
             .agg(act_med=("act", "median"), act_max=("act", "max"),
                  kpn=("kl_per_norm", "median"), pos_med=("pos", "median")))
        g["act_pct"] = g.act_med.rank(pct=True)
        # how far the measured positions moved between designs
        ppa = pa[pa.fid.isin(fids)].groupby("fid")["pos"].apply(set)
        psh = sh[sh.fid.isin(fids)].groupby("fid")["pos"].apply(set)
        jac = {f: len(ppa[f] & psh[f]) / len(ppa[f] | psh[f])
               for f in fids if f in ppa.index and f in psh.index}
        g["jac"] = pd.Series(jac)
        g["dchg"] = pd.Series(chg, index=fids)

        dc = decoder_cos(name, fids, top5)
        if dc is not None:
            g["maxcos"] = dc

        cols = ["act_pct", "act_max", "kpn", "jac", "dchg"]
        if "maxcos" in g:
            cols.append("maxcos")
        p(f"{'group':<10}{'n':>4}" + "".join(f"{c:>11}" for c in cols))
        for lab, s in (("TOP 5", top5), ("TOP 10", top10), ("REST", rest)):
            sub = g.loc[sorted(s)]
            p(f"{lab:<10}{len(sub):>4}" +
              "".join(f"{sub[c].median():>11.3f}" for c in cols))
        p("")
        for c in cols:
            a_ = g.loc[sorted(top5)][c]
            b_ = g.loc[sorted(rest)][c]
            try:
                pv = stats.mannwhitneyu(a_, b_).pvalue
                flag = "  <-- differs" if pv < 0.05 else ""
                p(f"    Mann-Whitney top5 vs rest, {c:<9} p={pv:.3g}{flag}")
            except ValueError:
                pass
        p(f"    carrier fids: {sorted(top5)}")

    p("")
    p("  If no descriptor separates the carriers, the msa rise is concentrated")
    p("  in latents with no common property -- report that, and stop looking")
    p("  for a mechanism.")

    with open(A.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
