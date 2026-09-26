# -*- coding: utf-8 -*-
"""Record what the remote runs actually used, before the machine is released.

"384 sequences of WikiText-103" is not reproducible: it does not say which 384.
This captures the corpus file and its hash, the exact sequence-selection rule,
the resolved Gemma Scope and base-model commit SHAs, the package versions, and
the GPU, into a file that ships with the repo. The question this answers arrives
in November from a reviewer, long after the GPU is gone.
"""
import hashlib
import io
import json
import os
import platform
import subprocess
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

REPOS = ["google/gemma-scope-2b-pt-res", "google/gemma-scope-9b-pt-res",
         "google/gemma-2-2b", "google/gemma-2-9b"]
OUT = "results/provenance.json"


def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
            if cap and h.name and fh.tell() > cap:
                break
    return h.hexdigest()


def main():
    rec = {"host": platform.node(), "python": sys.version.split()[0],
           "platform": platform.platform()}

    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:                                    # noqa: BLE001
        rec["torch_error"] = str(e)
    for mod in ("transformers", "huggingface_hub", "numpy", "pandas"):
        try:
            rec[mod] = __import__(mod).__version__
        except Exception:                                     # noqa: BLE001
            rec[mod] = "absent"

    # corpus: which file, and its hash
    try:
        sys.path.insert(0, "src")
        from corpus import WIKI
        rec["corpus_path"] = WIKI
        rec["corpus_exists"] = os.path.exists(WIKI)
        if os.path.exists(WIKI):
            rec["corpus_bytes"] = os.path.getsize(WIKI)
            rec["corpus_sha256"] = sha256(WIKI)
    except Exception as e:                                    # noqa: BLE001
        rec["corpus_error"] = str(e)

    # the selection rule, stated rather than left implicit in the code
    rec["sequence_selection"] = (
        "WikiText-103 train parquet; paragraphs with >400 stripped chars and "
        "not starting with '='; concatenated in file order into a rolling "
        "buffer; each time the buffer tokenises to >=512 tokens the first 512 "
        "are taken as one sequence and the buffer is reset; position 0 (BOS) "
        "is dropped, leaving 511 measured positions per sequence. Deterministic "
        "given the file, so n_seq=384 denotes the first 384 such sequences.")
    rec["n_seq"] = 384
    rec["n_pos_per_latent"] = 6
    rec["positions_per_sequence"] = 511

    # TWO corpora, and an earlier version of the paper named only one of them,
    # incorrectly. The arms are TRAINED on WikiText-103 (12M tokens will not
    # fit in WikiText-2 without repetition); every EVALUATION runs on
    # WikiText-2 raw. Recording them as separate fields so the two cannot be
    # conflated again.
    rec["train_corpus"] = {
        "repo": "wikitext", "config": "wikitext-103-raw-v1",
        "files": ["train-00000-of-00002.parquet",
                  "train-00001-of-00002.parquet"],
        "tokens_consumed": 12_000_000,
        "used_by": "src/train_saes.py"}
    rec["eval_corpus"] = {
        "repo": "wikitext", "config": "wikitext-2-raw-v1",
        "file": "train-00000-of-00001.parquet",
        "paragraphs_past_filter": 11843,
        "distinct_512_token_sequences_available": 3432,
        "tokens_available": 3432 * 512,
        "ladder_rungs": [96, 384, 1536],
        "max_rung_fraction_of_available": round(1536 / 3432, 3),
        "sequences_reused": False,
        "used_by": "src/eval_arms.py, src/released_agreement.py, "
                   "src/scope_matrix.py, src/firing_density.py"}

    # resolved commit SHAs for every repo used
    revs = {}
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for r in REPOS:
            try:
                kind = "dataset" if r == "wikitext" else "model"
                revs[r] = api.repo_info(r, repo_type=kind).sha
            except Exception as e:                            # noqa: BLE001
                revs[r] = "unresolved: %s" % type(e).__name__
    except Exception as e:                                    # noqa: BLE001
        revs["error"] = str(e)
    rec["hf_revisions"] = revs

    try:
        rec["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or "unavailable"
    except Exception:                                         # noqa: BLE001
        rec["git_commit"] = "unavailable"

    os.makedirs("results", exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    print(json.dumps(rec, indent=2, sort_keys=True))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
