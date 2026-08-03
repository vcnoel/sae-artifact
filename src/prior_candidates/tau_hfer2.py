# -*- coding: utf-8 -*-
"""Two stress tests the first run did not cover.

(1) SIGNAL REALISM. A2's X0 is isotropic Gaussian, so energy is near-uniform
    across modes and "energy above tau" degenerates into "count of modes above
    tau" -- a quantity that is invariant for trivial ESD reasons. Real token
    signals are strongly low-pass (the paper reports HFER ~0.02-0.05). Redo with
    a low-pass X0 synthesised on the base graph's own eigenbasis.

(2) UNEVEN REFINEMENT. Real tokenizers fragment different words by different
    amounts. That is not a measure-preserving refinement of the same graphon; it
    reweights it. This is the case the proposed paper claims as its contribution,
    so it is the case the repair has to survive.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\valno\Dev\spectral-multilingual")
from refinement import (D, GENERATORS, K_MAX, M, RHO, SEED, coarse_grain,
                        refine, within_parent_profile)

TAUS = [0.5, 0.8, 1.0, 1.2]


def lap(W):
    W = 0.5 * (W + W.T)
    d = np.maximum(W.sum(axis=1), 1e-12)
    Dm = np.diag(d ** -0.5)
    L = np.eye(len(W)) - Dm @ W @ Dm
    return 0.5 * (L + L.T)


def lowpass_signal(W0, rng, d_sig=64, alpha=3.0):
    """X0 with energy decaying as (1+lambda)^-alpha on the BASE graph's modes:
    a smooth graph signal, which is what a low-pass token signal looks like."""
    ev, U = np.linalg.eigh(lap(W0))
    w = (1.0 + ev) ** (-alpha)
    coef = rng.standard_normal((len(ev), d_sig)) * w[:, None]
    return U @ coef


def measure(W, X, rho=RHO):
    ev, U = np.linalg.eigh(lap(W))
    mass = ((U.T @ X) ** 2).sum(axis=1)
    tot = mass.sum() + 1e-30
    N = len(ev)
    out = {"hfer_rho": float(mass[int((1 - rho) * N):].sum() / tot)}
    for t in TAUS:
        out[f"tau{t}"] = float(mass[ev > t].sum() / tot)
    return out


def refine_uneven(W0, ks, rng):
    """Parent i splits into ks[i] children; edge mass W0[i,j] is spread over the
    ks[i] x ks[j] child block with row sums preserved. Coarse-graining by SUM
    (not by /k, since k now varies) returns W0 exactly."""
    m = len(W0)
    off = np.concatenate([[0], np.cumsum(ks)])
    N = int(off[-1])
    W = np.empty((N, N))
    prof = [within_parent_profile(int(k), "hetero", rng) if k > 1
            else np.ones((1, 1)) for k in ks]
    for i in range(m):
        for j in range(m):
            ki, kj = int(ks[i]), int(ks[j])
            # rows sum to 1 over the kj columns, then scaled by 1/ki so the
            # block sums to W0[i,j] exactly.
            P = np.resize(prof[i], (ki, kj))
            P = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-12) / ki
            W[off[i]:off[i+1], off[j]:off[j+1]] = W0[i, j] * P
    return W, off


def coarse_uneven(W, off):
    m = len(off) - 1
    return np.array([[W[off[i]:off[i+1], off[j]:off[j+1]].sum()
                      for j in range(m)] for i in range(m)])


def drift(vals):
    a, b = vals[0], vals[-1]
    return 0.0 if abs(a) < 1e-12 else (b - a) / abs(a)


# --- (1) low-pass signal, even refinement --------------------------------
print("=" * 74)
print("(1) LOW-PASS SIGNAL (alpha=3), even hetero refinement k=1..8")
print("=" * 74)
rows = []
for gname, gen in GENERATORS.items():
    W0 = gen(np.random.default_rng(SEED))
    X0 = lowpass_signal(W0, np.random.default_rng(SEED))
    series = {}
    for k in range(1, K_MAX + 1):
        r = np.random.default_rng(SEED + k)
        W = refine(W0, k, "hetero", r)
        assert np.abs(coarse_grain(W, k) - W0).max() < 1e-10
        X = np.repeat(X0, k, axis=0)
        X = X + 0.05 * r.standard_normal(X.shape)
        m = measure(W, X)
        for key, v in m.items():
            series.setdefault(key, []).append(v)
    rows.append(dict(gen=gname, **{f"{key}_k1": v[0] for key, v in series.items()},
                     **{f"{key}_drift": drift(v) for key, v in series.items()}))
t = pd.DataFrame(rows)
for key in ["hfer_rho"] + [f"tau{x}" for x in TAUS]:
    print(f"{key:>10}  k=1 value: " +
          "  ".join(f"{g}={t.loc[i, key+'_k1']:.4f}" for i, g in enumerate(t['gen'])) +
          "   |  drift: " +
          "  ".join(f"{g}={t.loc[i, key+'_drift']:+.3f}" for i, g in enumerate(t['gen'])))

# --- (2) uneven refinement, the real tokenizer case ----------------------
print()
print("=" * 74)
print("(2) UNEVEN REFINEMENT: parent i splits into k_i ~ realistic fertility mix")
print("    (some words stay whole, some shatter into byte fragments)")
print("=" * 74)
for gname, gen in GENERATORS.items():
    W0 = gen(np.random.default_rng(SEED))
    X0 = lowpass_signal(W0, np.random.default_rng(SEED))
    print(f"\n  generator = {gname}")
    print(f"  {'mean k':>7} {'N':>5} {'cg err':>9} {'hfer_rho':>9} " +
          " ".join(f"{'tau'+str(x):>8}" for x in TAUS))
    base = None
    for kmax in (1, 2, 4, 6, 8):
        r = np.random.default_rng(SEED + kmax)
        # heavy-tailed fertility: most parents split a little, a few a lot
        ks = 1 + r.binomial(kmax - 1, 0.35, size=M) if kmax > 1 else np.ones(M, int)
        W, off = refine_uneven(W0, ks, r)
        cg = float(np.abs(coarse_uneven(W, off) - W0).max())
        X = np.concatenate([np.repeat(X0[i:i+1], int(ks[i]), axis=0)
                            for i in range(M)], axis=0)
        X = X + 0.05 * r.standard_normal(X.shape)
        m = measure(W, X)
        if base is None:
            base = m
        print(f"  {ks.mean():7.2f} {len(W):5d} {cg:9.1e} {m['hfer_rho']:9.4f} " +
              " ".join(f"{m['tau'+str(x)]:8.4f}" for x in TAUS))
    print(f"  {'drift':>7} {'':5} {'':9} "
          f"{(m['hfer_rho']-base['hfer_rho'])/abs(base['hfer_rho']):+9.3f} " +
          " ".join(f"{(m['tau'+str(x)]-base['tau'+str(x)])/max(abs(base['tau'+str(x)]),1e-12):+8.3f}"
                   for x in TAUS))
