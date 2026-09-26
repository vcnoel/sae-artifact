# -*- coding: utf-8 -*-
"""Reliability of the latent ranking against the number of positions per latent.

CPU only. For each base model at 384 evaluation sequences, the shared-position
(latent, arm, position) cube of the crossed design is built exactly as in
make_macros.py (moderator.build, then boundary_check.comp_cube). Two curves are
written for k = 1..K positions per latent:

  erho2_dstudy  v_a / (v_a + v_ab + v_e / k), from the components estimated on
                the full cube (the generalizability D-study of eq. erho2)
  erho2_sub     the same coefficient re-estimated on cubes that keep k of the
                n positions of every cell, mean over R random subsets (k >= 2)
  pos_share     (v_e / k) / (v_a + v_ab + v_e / k), the share of the variance of
                a latent's mean score over k positions that position carries

Output: results/position_count_curve.csv
Run from the repository root:  PYTHONPATH=src python src/position_count_curve.py
"""
import io
import os
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

from boundary_check import comp_cube
from moderator import build as build_cubes

MODELS = {"QThreeFiveNineB": "q359b", "QThreeFiveFourB": "q354b",
          "QThreeFive": "q35", "SmolThree": "s3", "OlmoTwo": "o2",
          "GThree": "g3", "GTwo": "g2"}
KMAX = 12
R = 50


def main():
    rows = []
    for tag, ft in MODELS.items():
        pa = f"results/eval_arms_{ft}_s384.csv"
        sh = f"results/eval_arms_{ft}_s384_shared.csv"
        if not (os.path.exists(pa) and os.path.exists(sh)):
            continue
        _, _, _ca, cs, fids = build_cubes(pa, sh)
        c = comp_cube(cs)
        I, J, n = cs.shape
        rng = np.random.default_rng(0)
        for k in range(1, KMAX + 1):
            den = c["v_a"] + c["v_ab"] + c["v_e"] / k
            sub = float("nan")
            if 2 <= k <= n:
                vals = []
                for _ in range(R):
                    idx = np.stack([rng.permutation(n)[:k]
                                    for _ in range(I * J)]).reshape(I, J, k)
                    cube_k = np.take_along_axis(cs, idx, axis=2)
                    vals.append(comp_cube(cube_k)["erho2"])
                sub = float(np.mean(vals))
            rows.append(dict(model=tag, corpus=384, latents=I, arms=J,
                             n_positions=n, k=k,
                             erho2_dstudy=c["v_a"] / den,
                             erho2_sub=sub,
                             pos_share=100 * (c["v_e"] / k) / den))
        print(tag, I, J, n, " ".join(f"{r['erho2_dstudy']:.3f}"
                                     for r in rows if r["model"] == tag))
    out = pd.DataFrame(rows)
    out.to_csv("results/position_count_curve.csv", index=False)
    print("wrote results/position_count_curve.csv", len(out))


if __name__ == "__main__":
    main()
