# -*- coding: utf-8 -*-
"""Reliability of the latent ranking against the number of positions per latent.

Decision study on the shared-position cubes at 384 sequences: the variance
components of eq. (erho2) are estimated once on the full balanced cube, and
E rho^2(k) = v_a / (v_a + v_ab + v_e / k) projects the coefficient to k
positions per latent. An empirical check subsamples k of the cube's positions
(k >= 2, 200 draws) and re-estimates the coefficient directly.

Writes results/position_curve.csv (one row per model and k) and
results/position_curve_kmin.csv (smallest k reaching 0.8 and 0.9 per model).
Run from the repository root: PYTHONPATH=src python analysis/position_curve.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from boundary_check import comp_cube          # noqa: E402
from moderator import build                   # noqa: E402

MODELS = [("QThreeFiveNineB", "q359b"), ("QThreeFiveFourB", "q354b"),
          ("QThreeFive", "q35"), ("SmolThree", "s3"), ("OlmoTwo", "o2"),
          ("GThree", "g3"), ("GTwo", "g2")]
KMAX = 64
TARGETS = (0.8, 0.9)


def erho(c, k):
    den = c["v_a"] + c["v_ab"] + c["v_e"] / k
    return c["v_a"] / den if den > 0 else float("nan")


def main():
    rows, kmin = [], []
    for tag, ft in MODELS:
        pa = f"results/eval_arms_{ft}_s384.csv"
        sh = f"results/eval_arms_{ft}_s384_shared.csv"
        if not (os.path.exists(pa) and os.path.exists(sh)):
            continue
        _, _, _, cs, fids = build(pa, sh)
        c = comp_cube(cs)
        n = cs.shape[2]
        rng = np.random.default_rng(0)
        for k in range(1, KMAX + 1):
            emp = float("nan")
            if 2 <= k <= n:
                draws = []
                for _ in range(200):
                    sel = rng.choice(n, k, replace=False)
                    draws.append(comp_cube(cs[:, :, sel])["erho2"])
                emp = float(np.nanmean(draws))
            rows.append(dict(model=tag, k=k, n_latents=cs.shape[0], n_pos=n,
                             erho2_proj=erho(c, k), erho2_emp=emp))
        e_inf = c["v_a"] / (c["v_a"] + c["v_ab"])
        rec = dict(model=tag, n_pos=n, erho2_at_1=erho(c, 1),
                   erho2_at_n=erho(c, n), erho2_limit=e_inf)
        for t in TARGETS:
            ks = [k for k in range(1, 10001) if erho(c, k) >= t]
            rec[f"k_for_{int(t * 100):03d}"] = ks[0] if ks else None
        kmin.append(rec)
        print(rec, flush=True)
    pd.DataFrame(rows).to_csv("results/position_curve.csv", index=False)
    pd.DataFrame(kmin).to_csv("results/position_curve_kmin.csv", index=False)


if __name__ == "__main__":
    main()
