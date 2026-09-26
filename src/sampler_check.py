# -*- coding: utf-8 -*-
"""Refuse to compare evaluation files that do not share a latent sample.

R6: results/eval_arms_rerun.csv predates the sampler fix and shares 7 of 240
latents with the 384 and 1536 runs, which share 232 with each other. The paper
put its numbers side by side under a corpus label for weeks. Nothing in the
pipeline objected, because every file was individually valid.

The bug had already been found, fixed, and written up in the past tense in the
reproducibility statement. What had not happened was repointing make_macros.py
at the fixed output, which sat on disk under a _fixsamp suffix the whole time.
Fixing a defect and adopting the fix are separate actions; this checks the
second one.

Run directly, or via src/check_numbers.py which calls it.
"""
import io
import itertools
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import os

import pandas as pd

MIN_OVERLAP = 0.80

# Files that are compared against each other anywhere in the paper. Each group
# must be on one sampler; membership is what a cross-corpus claim asserts.
GROUPS = {
    "gemma2-2b corpus curve": [
        "results/eval_arms_rerun_fixsamp.csv",
        "results/eval_arms_g2_s384.csv",
        "results/eval_arms_g2_s1536.csv",
    ],
    "gemma3-1b corpus curve": [
        "results/eval_arms_g3.csv",
        "results/eval_arms_g3_s384.csv",
        "results/eval_arms_g3_s1536.csv",
    ],
}

# Known-bad files, kept on disk so R6 stays inspectable. Referencing one from
# an analysis script is itself the defect.
QUARANTINE = {
    "results/eval_arms_rerun.csv": "pre-sampler-fix; use *_fixsamp (R6)",
    "results/eval_arms_shared.csv": "pre-sampler-fix; use *_fixsamp (R6)",
}


def fids(path):
    return set(pd.read_csv(path, usecols=["fid"]).fid.unique())


def main():
    bad, checked = [], 0
    for label, files in GROUPS.items():
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            print(f"{label}: SKIP (missing {', '.join(missing)})")
            continue
        S = {f: fids(f) for f in files}
        print(f"{label}")
        for x, y in itertools.combinations(files, 2):
            inter = len(S[x] & S[y])
            frac = inter / min(len(S[x]), len(S[y]))
            checked += 1
            flag = "ok" if frac >= MIN_OVERLAP else "FAIL"
            print(f"  {os.path.basename(x):<34} vs {os.path.basename(y):<34}"
                  f" {inter:>4} shared ({frac:5.1%}) {flag}")
            if frac < MIN_OVERLAP:
                bad.append(f"{label}: {x} vs {y} share {frac:.1%} of latents, "
                           f"below {MIN_OVERLAP:.0%}")

    # a checker that compares nothing is worse than no checker
    if checked == 0:
        print("ERROR: no file pairs were compared; the groups above resolve to "
              "nothing on disk.", file=sys.stderr)
        raise SystemExit(1)

    # Per file, with an allowlist. repro_check.py exists precisely to compare
    # the pre-fix run against its rerun -- it is what measured the 8-of-240
    # overlap -- so forbidding it there would delete the evidence for R6.
    ALLOWED = {"sampler_check.py", "repro_check.py"}
    for f in sorted(os.listdir("src")):
        if not f.endswith((".py", ".sh")) or f in ALLOWED:
            continue
        text = io.open(os.path.join("src", f), encoding="utf-8").read()
        text = "\n".join(ln for ln in text.split("\n")
                         if not ln.lstrip().startswith("#"))
        for q, why in QUARANTINE.items():
            if q in text:
                bad.append(f"src/{f} references quarantined {q}: {why}")

    if bad:
        print("\nSAMPLER MISMATCH:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        raise SystemExit(1)
    print(f"\n{checked} file pairs checked; all share >= {MIN_OVERLAP:.0%} "
          f"of their latents")


if __name__ == "__main__":
    main()
