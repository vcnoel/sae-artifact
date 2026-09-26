# -*- coding: utf-8 -*-
"""Resolve the evaluation corpus without importing the training stack.

released_agreement.py and scope_matrix.py took WIKI from eval_saes, which
imports train_saes, which pulls in a base-model config and CUDA constants. On a
fresh machine that chain fails for reasons that have nothing to do with the corpus.
This resolves the same file and nothing else, so the scripts are portable.

The path is unchanged where the old cache exists, so local runs reproduce
exactly; elsewhere the same parquet is fetched from the Hub.
"""
import os

_CACHED = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--wikitext/snapshots/"
    "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/"
    "train-00000-of-00001.parquet")


def _resolve():
    if os.path.exists(_CACHED):
        return _CACHED
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        "wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        repo_type="dataset")


WIKI = _resolve()
