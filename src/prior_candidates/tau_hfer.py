# -*- coding: utf-8 -*-
"""Does an EIGENVALUE-THRESHOLD HFER survive node-set refinement?

The proposal under test: replace HFER's mode-INDEX cutoff (top rho fraction of
modes) with an eigenvalue threshold (modes with lambda > tau). The claim is that
eigenvalues are graphon-level while indices are not, so HFER_tau should be
refinement-invariant where HFER_rho drifts 114%.

Reuses A2's operator and generators unchanged. Same seeds, so the HFER_rho
column must reproduce results/A2_refinement/refinement.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\valno\Dev\spectral-multilingual")
from refinement import (D, GENERATORS, K_MAX, M, RHO, SEED, coarse_grain,
                        refine, refine_signal)

TAUS = [0.5, 0.8, 1.0, 1.2, 1.5]


def spec(W, X):
    W = 0.5 * (W + W.T)
    d = np.maximum(W.sum(axis=1), 1e-12)
    Dm = np.diag(d ** -0.5)
    L = np.eye(len(W)) - Dm @ W @ Dm
    L = 0.5 * (L + L.T)
    ev, U = np.linalg.eigh(L)
    mass = ((U.T @ X) ** 2).sum(axis=1)
    return ev, mass


def run(signal_mode):
    """signal_mode: 'inherit' = children copy parent exactly (the graphon-signal
    case); 'jitter' = A2's default, children carry their own residual state."""
    X0 = np.random.default_rng(SEED).standard_normal((M, D))
    rows = []
    for gname, gen in GENERATORS.items():
        W0 = gen(np.random.default_rng(SEED))
        for regime in ("uniform", "chain", "hetero"):
            for k in range(1, K_MAX + 1):
                r = np.random.default_rng(SEED + k)
                W = refine(W0, k, regime, r)
                assert np.abs(coarse_grain(W, k) - W0).max() < 1e-10
                if signal_mode == "inherit":
                    X = np.repeat(X0, k, axis=0)
                else:
                    X = refine_signal(X0, k, regime, r)
                ev, mass = spec(W, X)
                N = len(ev)
                tot = mass.sum() + 1e-30
                cut = int((1.0 - RHO) * N)
                rec = dict(generator=gname, regime=regime, k=k, N=N,
                           hfer_rho=float(mass[cut:].sum() / tot))
                for t in TAUS:
                    rec[f"hfer_tau{t}"] = float(mass[ev > t].sum() / tot)
                    rec[f"frac_modes_above{t}"] = float((ev > t).mean())
                rows.append(rec)
    return pd.DataFrame(rows)


def drift(df, col):
    out = {}
    for (g, rg), sub in df.groupby(["generator", "regime"]):
        a = float(sub.loc[sub.k == 1, col].iloc[0])
        b = float(sub.loc[sub.k == K_MAX, col].iloc[0])
        out[(g, rg)] = 0.0 if abs(a) < 1e-12 else (b - a) / abs(a)
    return out


for mode in ("jitter", "inherit"):
    df = run(mode)
    print(f"\n{'='*72}\nSIGNAL = {mode}"
          f"  ({'A2 default: children carry own state' if mode=='jitter' else 'children copy parent exactly'})"
          f"\n{'='*72}")
    cols = ["hfer_rho"] + [f"hfer_tau{t}" for t in TAUS]
    for col in cols:
        d = drift(df, col)
        het = {k[0]: v for k, v in d.items() if k[1] == "hetero"}
        print(f"{col:>14}  hetero drift k=1->8: " +
              "  ".join(f"{g}={v:+.3f}" for g, v in het.items()) +
              f"   | max|drift| all regimes = {max(abs(v) for v in d.values()):.3f}")
    h = df[(df.regime == "hetero") & (df.generator == "block")]
    print("\n  block/hetero, fraction of modes above tau (does the threshold "
          "track a moving spectrum?):")
    print(h[["k", "N", "frac_modes_above0.8", "frac_modes_above1.0",
             "frac_modes_above1.2"]].to_string(index=False))
