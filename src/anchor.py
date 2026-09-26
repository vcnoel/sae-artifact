# -*- coding: utf-8 -*-
"""The anchor model of the body: one rule, read by make_macros.py and figures_v2.py.

The body quotes one base model as its worked example (the Anchor* macros) and
Figure 1 is drawn on one base model. Both read choose() below, so the prose and
the figure cannot name different models.

Rule: the largest Qwen3.5 rung whose inputs are all on disk. A rung qualifies
when every file the Anchor* macros and Figure 1 read is present:

  results/eval_arms_<tag>{,_s384,_s1536}{,_shared}.csv   the three corpus rungs,
      per-arm and shared (Figure 1 draws a model only when all three exist, and
      the 384 pair gives the full-sample interaction and the sampled count)
  results/displacement.csv, observed and shuffled rows    the chance comparison
  results/union_check.csv, union and shared rows          the retained count
  results/position_structure_<tag>.txt                    the residual regression

9B is taken when it qualifies, else 4B, else 2B. A rung that has landed only in
part is never the anchor, so a half-finished evaluation cannot move the prose.

    python src/anchor.py      # prints the slug chosen now and what each rung lacks
"""
import os

import pandas as pd

# largest first: (macro slug, model name in the result files, file tag)
LADDER = [("QThreeFiveNineB", "qwen35-9b", "q359b"),
          ("QThreeFiveFourB", "qwen35-4b", "q354b"),
          ("QThreeFive", "qwen35-2b", "q35")]


def _rows(path, model, col, need):
    if not os.path.exists(path):
        return False
    d = pd.read_csv(path)
    have = set(d.loc[d.model == model, col])
    return all(k in have for k in need)


def missing(model, tag, root="."):
    """Inputs a rung still lacks, as a list of short descriptions."""
    j = lambda p: os.path.join(root, p)
    out = []
    for sfx in ("", "_s384", "_s1536"):
        for sh in ("", "_shared"):
            p = f"results/eval_arms_{tag}{sfx}{sh}.csv"
            if not os.path.exists(j(p)):
                out.append(p)
    if not _rows(j("results/displacement.csv"), model, "kind",
                 ("observed", "shuffled")):
        out.append(f"displacement.csv rows for {model}")
    if not _rows(j("results/union_check.csv"), model, "rule",
                 ("union", "shared")):
        out.append(f"union_check.csv rows for {model}")
    p = f"results/position_structure_{tag}.txt"
    if not os.path.exists(j(p)):
        out.append(p)
    return out


def choose(root="."):
    """(slug, model, tag) of the anchor: the largest rung with no input missing."""
    for slug, model, tag in LADDER:
        if not missing(model, tag, root):
            return slug, model, tag
    raise SystemExit("anchor: no Qwen3.5 rung has all of its inputs; "
                     + "; ".join(f"{m}: {', '.join(missing(m, t, root))}"
                                 for _, m, t in LADDER))


if __name__ == "__main__":
    for slug, model, tag in LADDER:
        miss = missing(model, tag)
        print(f"{model:10s} {'complete' if not miss else 'lacks ' + ', '.join(miss)}")
    print("anchor:", choose()[1])
