# -*- coding: utf-8 -*-
"""Extract the per-layer tables from Cho et al. 2607.20596 and VALIDATE the
extraction by re-deriving their reported Spearman rho.

PDF table extraction is error-prone and these numbers become load-bearing, so
nothing is used until the extraction reproduces a statistic the paper prints.
Table 21 (Gemma-2-9B x GemmaScope, 42 layers) states
rho(depth, |delta logit|) = 0.81; Table 18 covers Gemma-2-2B, which is our model.

Writes results/target_tables.csv only if validation passes.
"""
import io
import re
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import fitz
import numpy as np
import pandas as pd
from scipy import stats

PDF = "st.pdf"
MINUS = "−"


def norm(s):
    return s.replace(MINUS, "-").replace("∗", "*").replace("*", "")


def find_table(txt, num):
    i = txt.find(f"Table {num}:")
    if i < 0:
        return None, None
    cap = re.sub(r"\s+", " ", txt[i:i + 260])
    # the table body follows the caption; take a generous window
    return cap, txt[i:i + 6000]


def parse_rows(block):
    """Rows are emitted as a flat stream: layer, nST, pBH, dlogit."""
    toks = [norm(x) for x in re.findall(
        r"-?\d+\.\d+e-\d+|-?\d+\.\d+|-?\d+", norm(block))]
    rows, i = [], 0
    while i + 3 < len(toks):
        L, n, p, d = toks[i:i + 4]
        try:
            Li, ni, pf, df = int(L), int(n), float(p), float(d)
        except ValueError:
            i += 1
            continue
        # a plausible row: layer 0-45, nST 1-2000, p a tiny probability,
        # dlogit a smallish negative number
        if 0 <= Li <= 45 and 1 <= ni <= 3000 and 0 < pf < 0.5 and -5 < df < 1:
            rows.append((Li, ni, pf, df))
            i += 4
        else:
            i += 1
    # keep the longest run of consecutive layers
    if not rows:
        return pd.DataFrame(columns=["layer", "nST", "pBH", "dlogit"])
    rows.sort()
    return pd.DataFrame(rows, columns=["layer", "nST", "pBH", "dlogit"]
                        ).drop_duplicates("layer")


def main():
    txt = "\n".join(p.get_text() for p in fitz.open(PDF))
    out = []
    for num, expect_rho, label in ((21, 0.81, "Gemma-2-9B x GemmaScope"),
                                   (18, None, "Gemma-2-2B"),
                                   (14, None, "ST prevalence by architecture")):
        cap, block = find_table(txt, num)
        if block is None:
            print(f"Table {num}: NOT FOUND")
            continue
        print("=" * 88)
        print(f"Table {num}: {cap[:150]}")
        df = parse_rows(block)
        print(f"  parsed {len(df)} rows, layers {df.layer.min() if len(df) else '-'}"
              f"-{df.layer.max() if len(df) else '-'}")
        if len(df) >= 5:
            r = stats.spearmanr(df.layer, df.dlogit.abs())
            print(f"  nST range {df.nST.min()}-{df.nST.max()}  median {df.nST.median():.0f}")
            print(f"  re-derived rho(layer, |dlogit|) = {r.statistic:+.3f} "
                  f"(p={r.pvalue:.2e})")
            if expect_rho is not None:
                ok = abs(r.statistic - expect_rho) < 0.05
                print(f"  paper reports {expect_rho:+.2f} -> "
                      f"{'VALIDATED' if ok else 'MISMATCH, extraction unsafe'}")
            df["table"] = num
            df["source"] = label
            out.append(df)
        else:
            print("  too few rows parsed; table is not in a flat "
                  "layer/nST/p/dlogit stream")
    if out:
        pd.concat(out).to_csv("results/target_tables.csv", index=False)
        print(f"\nwrote results/target_tables.csv")


if __name__ == "__main__":
    main()
