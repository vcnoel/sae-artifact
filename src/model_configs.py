# -*- coding: utf-8 -*-
"""Base-model configs for the arm trainer/evaluator.

Gemma-2-2B and Gemma-3-1B both have 26 transformer blocks, so layer 12 sits at
the same RELATIVE depth (46%) in both -- picked for that reason, not tuned per
model. Width and k are held at the values used for the Gemma-2-2B six-arm run
so the two replications differ only in base model, not in dictionary shape.

Qwen3.5-2B-Base is the second architecture family. It has 24 blocks, so layer 11
sits at 45.8% relative depth, the nearest match to the Gemma 46%. Its hidden size
is 2048, read from the published config. Width and k are held at the same values
as the Gemma arms for the same reason. It is a hybrid stack, six of its 24 blocks
carrying full attention, which does not affect an autoencoder fit on the residual
stream but does mean any per head index taken from it must say which it counts.

OLMo-2-1B (the April 2025 base checkpoint) is the third family. It has 16 blocks, so
layer 7 sits at 43.8% relative depth, the nearest match to the Gemma 46% (layer 8
would be 50%). Hidden size 2048, from the published config. Like Qwen3.5, its
tokenizer prepends no BOS, so position 0 is the first real token.

SmolLM3-3B (the July 2025 base checkpoint) is the fourth family. It has 36 blocks, so
layer 17 sits at 47.2% relative depth, the nearest match to the Gemma 46% (layer 16
would be 44.4%). Hidden size 2048, from the published config. Its tokenizer has no
BOS token either.

Qwen3.5-4B-Base and Qwen3.5-9B-Base extend the Qwen3.5 family upward, so a trend in
scale is separable from a family effect. Both have 32 blocks, so layer 15 sits at
46.9% relative depth, the nearest match to the Gemma 46%. Hidden sizes 2560 and 4096,
from the published configs. Width and k are held at the values of every other arm,
so the dictionary shape does not change with the base model.

qwen35-2b-full is the Qwen3.5-2B-Base configuration unchanged, trained on 48M tokens
per arm (four times the 12M budget), to test whether
the position effect depends on the training budget. Same shared initialisation, same six fitting choices, its own folder.
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
    "qwen35-2b": dict(
        model="Qwen/Qwen3.5-2B-Base", layer=11, d_model=2048,
        width=16384, k=82, seq=512, dir="data/q35",
    ),
    "olmo2-1b": dict(
        model="allenai/OLMo-2-0425-1B", layer=7, d_model=2048,
        width=16384, k=82, seq=512, dir="data/o2",
    ),
    "smollm3-3b": dict(
        model="HuggingFaceTB/SmolLM3-3B-Base", layer=17, d_model=2048,
        width=16384, k=82, seq=512, dir="data/s3",
    ),
    "qwen35-2b-full": dict(
        model="Qwen/Qwen3.5-2B-Base", layer=11, d_model=2048,
        width=16384, k=82, seq=512, dir="data/q35_full",
    ),
    "qwen35-4b": dict(
        model="Qwen/Qwen3.5-4B-Base", layer=15, d_model=2560,
        width=16384, k=82, seq=512, dir="data/q35_4b",
    ),
    "qwen35-9b": dict(
        model="Qwen/Qwen3.5-9B-Base", layer=15, d_model=4096,
        width=16384, k=82, seq=512, dir="data/q35_9b",
    ),
}


def get(base):
    if base not in MODELS:
        raise SystemExit(f"unknown --base {base!r}; choices: {list(MODELS)}")
    return MODELS[base]
