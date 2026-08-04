# -*- coding: utf-8 -*-
"""Base-model configs for the arm trainer/evaluator.

Gemma-2-2B and Gemma-3-1B both have 26 transformer blocks, so layer 12 sits at
the same RELATIVE depth (46%) in both -- picked for that reason, not tuned per
model. Width and k are held at the values used for the Gemma-2-2B six-arm run
so the two replications differ only in base model, not in dictionary shape.
"""

MODELS = {
    "gemma2-2b": dict(
        model="google/gemma-2-2b", layer=12, d_model=2304,
        width=16384, k=82, seq=512, dir="data",
    ),
    "gemma3-1b": dict(
        model="google/gemma-3-1b-pt", layer=12, d_model=1152,
        width=16384, k=82, seq=512, dir="data/g3",
    ),
}


def get(base):
    if base not in MODELS:
        raise SystemExit(f"unknown --base {base!r}; choices: {list(MODELS)}")
    return MODELS[base]
