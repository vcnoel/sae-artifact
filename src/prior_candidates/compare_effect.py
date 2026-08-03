# -*- coding: utf-8 -*-
"""Candidate C, Test 2, done on data rather than by literature estimate.

The question is whether the formatting drift from Test 1 is large or small
relative to the effects these metrics are actually used to report. Rather than
estimating effect sizes out of published figures, this compares against the
best-documented instance available: the cross-language contrasts in Valentin's
own multilingual sweep, same metric family, same model, same pipeline.

Two reference quantities per metric, both in the same relative units as the
drift table:
  cross-language spread -- the interquartile spread of per-language means about
      the grand median, in the MATCHED scope (N fixed at 96, so this is the
      contrast the paper actually claims, with graph size already held fixed);
  within-language spread -- the same statistic computed over items inside one
      language, which is the noise these contrasts are read against.

If reformatting moves a metric as much as changing language does, the confound
is material. If it is an order of magnitude smaller, C is a footnote.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = r"C:\Users\valno\Dev\spectral-multilingual\data"
PQ = os.path.join(REPO, "spec__llama32_1b__matched.parquet")

# metric name in his parquet -> name in the drift table
MAP = {"hfer": "hfer", "fiedler": "fiedler", "lambda_max": "lambda_max",
       "smoothness": "smoothness", "entropy": "spec_entropy",
       "attn_entropy": "attn_entropy", "attn_distance": "attn_distance",
       "eff_rank": "eff_rank", "energy": "energy"}

d = pd.read_parquet(PQ)
mid = d[(d.depth >= 1 / 3) & (d.depth < 2 / 3)]

out = []
w = out.append
w("=" * 88)
w("REFERENCE EFFECT SIZES from the multilingual sweep (llama32_1b, matched "
  "scope, mid depth)")
w(f"  languages={mid.code.nunique()}  items={mid.item_id.nunique()}  "
  f"rows={len(mid)}")
w("=" * 88)
w(f"{'metric':>16} {'cross-language':>16} {'within-language':>17} "
  f"{'lang range':>12}")
w(f"{'':>16} {'(rel. IQR)':>16} {'(rel. IQR)':>17} {'(p95/p5)':>12}")
w("-" * 88)

ref = {}
for src, name in MAP.items():
    if src not in mid.columns:
        continue
    per_lang = mid.groupby("code")[src].mean()
    gm = per_lang.median()
    if abs(gm) < 1e-12:
        continue
    x_iqr = float((per_lang.quantile(.75) - per_lang.quantile(.25)) / abs(gm))
    # within-language: spread of item means inside each language, then median
    wi = mid.groupby(["code", "item_id"])[src].mean().reset_index()
    per = wi.groupby("code")[src].apply(
        lambda s: (s.quantile(.75) - s.quantile(.25)) / max(abs(s.median()),
                                                            1e-12))
    w_iqr = float(per.median())
    rng = float(per_lang.quantile(.95) / max(per_lang.quantile(.05), 1e-12))
    ref[name] = dict(cross=x_iqr, within=w_iqr)
    w(f"{name:>16} {x_iqr:>15.2%} {w_iqr:>16.2%} {rng:>11.2f}x")

# --- join against the measured drift -------------------------------------
if os.path.exists("drift_summary.csv"):
    dr = pd.read_csv("drift_summary.csv").set_index("metric")
    w("\n" + "=" * 88)
    w("FORMATTING DRIFT vs THOSE EFFECTS")
    w("=" * 88)
    w(f"{'metric':>16} {'format drift':>13} {'cross-lang':>12} "
      f"{'drift/cross':>12} {'within-lang':>12} {'drift/within':>13}")
    w("-" * 88)
    for name, r in ref.items():
        if name not in dr.index:
            continue
        fd = float(dr.loc[name, "med"])
        w(f"{name:>16} {fd:>12.2%} {r['cross']:>11.2%} "
          f"{fd/max(r['cross'],1e-12):>11.2f} {r['within']:>11.2%} "
          f"{fd/max(r['within'],1e-12):>12.2f}")
    w("\n  drift/cross > ~0.3 means reformatting moves the metric an")
    w("  appreciable fraction of what changing LANGUAGE moves it.")
else:
    w("\n(drift_summary.csv not present yet -- run analyze_drift.py first)")

open("effect_report.txt", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
