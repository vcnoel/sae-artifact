# -*- coding: utf-8 -*-
"""Observed vs null displacement between the tokens two arms choose.

Figure 1(a) plots these; this writes them so the prose can cite them through
make_macros rather than reading them off a picture. The null shuffles among the
positions the latent ACTUALLY fires at, not all 511 tokens: the arms can only
choose where the latent fires, so an unconstrained null would flatter us.
"""
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np
import pandas as pd

from fig_one import displacements

FILES = {"gemma2-2b": "results/eval_arms_g2_s384.csv",
         "gemma3-1b": "results/eval_arms_g3_s384.csv"}

rows = []
for model, f in FILES.items():
    obs, null = displacements(pd.read_csv(f))
    # "shuffled", not "null": pandas' default na_values contains the string
    # "null", so read_csv turned that label into NaN and every baseline row
    # silently disappeared downstream. The macros for the null were simply
    # never emitted, and an unused macro is not an error, so nothing complained.
    for lab, v in (("observed", obs), ("shuffled", null)):
        rows.append(dict(model=model, kind=lab, n=len(v),
                         pct_same=100.0 * float((v == 0).mean()),
                         median=float(np.median(v)),
                         pct_far=100.0 * float((v > 128).mean())))
d = pd.DataFrame(rows)
d.to_csv("results/displacement.csv", index=False)
print(d.to_string(index=False))
