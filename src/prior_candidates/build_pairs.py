# -*- coding: utf-8 -*-
"""Candidate C, Test 1: build semantically-null formatting pairs.

Each pair is (A, A') where A' is A under typographic changes a copy editor would
call free variation: curly apostrophes and quotes, an ellipsis character, a
newline instead of a space after a stop or a comma, double-spacing, a leading
space, an em dash, and non-breaking spaces. Meaning is identical; the byte
string, and therefore the BPE segmentation, is not.

WHY NOT BYTE FALLBACK. Forcing byte-level fragmentation (the Amharic regime)
moves every metric, but it is off-distribution -- one would be measuring
degeneracy, not tokenizer sensitivity. Every operator here produces text common
in the pretraining distribution.

TWO CORRECTIONS AFTER THE FIRST RUN, both of which mattered:
  1. wikitext-2-raw is Moses-tokenised (`Arsenal , also`), which is not prose and
     not what any of these metrics are computed on in practice. It is
     detokenised back to normal spacing first.
  2. The thousands-separator operator turned `1840` into `1,840`. That is a year,
     so the edit was not meaning-preserving and the pair was invalid. Separators
     are now applied only at five digits or more, where the string is a
     quantity rather than a date.

NBSP IS THE STRONGEST OPERATOR AND IS TRACKED SEPARATELY. Non-breaking space
renders identically and is ubiquitous in web-scraped text, but it is the
operator furthest from ordinary prose. Every pair records whether it used one,
so the headline can be recomputed on the conservative subset alone.
"""
import io
import json
import os
import random
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

WIKI = (os.path.expanduser("~") + "/.cache/huggingface/hub/datasets--wikitext/"
        "snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/"
        "wikitext-2-raw-v1/train-00000-of-00001.parquet")
MODEL = "meta-llama/Llama-3.2-1B"
LO, HI = 0.05, 0.30
TARGET = 200


def detok(s):
    """Undo Moses spacing so the input is ordinary prose."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r' ([,.;:!?%)\]])', r"\1", s)
    s = re.sub(r'([(\[]) ', r"\1", s)
    s = s.replace(" 's", "'s").replace(" n't", "n't").replace(" 've", "'ve")
    s = s.replace(" 're", "'re").replace(" 'll", "'ll").replace(" 'd", "'d")
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r' " ([^"]*) " ', r' "\1" ', s)
    return s


# --- semantically null typographic operators ------------------------------
def op_curly(s):
    s = re.sub(r"(?<=\w)'(?=\w)", "’", s)
    s = re.sub(r"(?<=\w)'(?=s\b)", "’", s)
    return s


def op_quotes(s):
    out, open_q = [], True
    for ch in s:
        if ch == '"':
            out.append("“" if open_q else "”")
            open_q = not open_q
        else:
            out.append(ch)
    return "".join(out)


def op_ellipsis(s):
    return s.replace("...", "…")


def op_newline_stop(s):
    return re.sub(r"(?<=[.!?]) (?=[A-Z])", "\n", s)


def op_newline_comma(s):
    return re.sub(r", ", ",\n", s)


def op_numsep(s):
    """Only 5+ digits: 4-digit strings are usually years, and `1,840` for a
    year is not a meaning-preserving edit."""
    return re.sub(r"\b\d{5,9}\b", lambda m: format(int(m.group(0)), ","), s)


def op_double_space(s):
    return re.sub(r"(?<=[.!?]) (?=[A-Z])", "  ", s)


def op_lead(s):
    return " " + s


def op_dash(s):
    return s.replace(" - ", " — ")


def nbsp(frac):
    """Replace a deterministic `frac` of spaces with U+00A0. Graded, so the
    delta can be tuned into the band instead of overshooting it."""
    def f(s):
        rng = random.Random(len(s))
        return "".join(" " if (c == " " and rng.random() < frac) else c
                       for c in s)
    f.__name__ = f"nbsp{frac}"
    return f


LIGHT = [op_curly, op_quotes, op_ellipsis, op_numsep, op_dash]
BUNDLES = [
    (LIGHT, False),
    (LIGHT + [op_newline_stop], False),
    (LIGHT + [op_newline_stop, op_lead, op_double_space], False),
    (LIGHT + [op_newline_comma], False),
    (LIGHT + [op_newline_comma, op_newline_stop, op_lead], False),
    (LIGHT + [op_newline_comma, op_newline_stop, op_lead, nbsp(0.15)], True),
    (LIGHT + [op_newline_comma, op_newline_stop, op_lead, nbsp(0.35)], True),
    (LIGHT + [op_newline_comma, op_newline_stop, op_lead, nbsp(0.60)], True),
    (LIGHT + [op_newline_comma, op_newline_stop, op_lead, nbsp(1.0)], True),
]


def apply(s, ops):
    for f in ops:
        s = f(s)
    return s


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = pd.read_parquet(WIKI)["text"].tolist()

    cands = []
    for raw in texts:
        t = str(raw).strip()
        if len(t) < 250 or t.startswith("="):
            continue
        t = detok(t)
        m = list(re.finditer(r"(?<=[.!?]) (?=[A-Z])", t))
        end = None
        for mm in m:
            if 250 <= mm.start() <= 650:
                end = mm.start() + 1
        if end is None:
            continue
        t = t[:end].strip()
        if t.count(",") < 2 or " ," in t or " ." in t:
            continue
        cands.append(t)
        if len(cands) >= 4000:
            break
    print(f"candidate paragraphs (detokenised): {len(cands)}")

    # STRATIFIED, because the first version of this loop chose whichever bundle
    # hit the delta target and therefore selected NBSP for all 200 pairs. That
    # would have rested the entire result on the single operator furthest from
    # ordinary prose. Half the corpus is now built from typographic operators
    # only (whatever delta they happen to reach), half from graded NBSP to cover
    # the top of the band, and the two are analysed separately.
    half = TARGET // 2
    strata = {"typographic": [], "nbsp": []}
    want = [LO + (HI - LO) * (i % 10) / 9.0 for i in range(len(cands))]
    for t, tgt in zip(cands, want):
        na = len(tok(t, add_special_tokens=False)["input_ids"])
        for key, allow_nbsp in (("typographic", False), ("nbsp", True)):
            if len(strata[key]) >= half:
                continue
            best, berr = None, 9e9
            for ops, uses_nbsp in BUNDLES:
                if uses_nbsp != allow_nbsp:
                    continue
                t2 = apply(t, ops)
                if t2 == t:
                    continue
                nb = len(tok(t2, add_special_tokens=False)["input_ids"])
                d = (nb - na) / na
                if not (LO <= abs(d) <= HI):
                    continue
                err = abs(abs(d) - tgt)
                if err < berr:
                    best, berr = (t2, na, nb, d, len(ops), uses_nbsp), err
            if best is None:
                continue
            t2, na, nb, d, nops, un = best
            strata[key].append(dict(a=t, b=t2, n_a=na, n_b=nb, delta=d,
                                    n_ops=nops, nbsp=un, stratum=key))
        if all(len(v) >= half for v in strata.values()):
            break
    pairs = strata["typographic"] + strata["nbsp"]

    print(f"pairs in band [{LO:.0%}, {HI:.0%}]: {len(pairs)}")
    ds_ = sorted(abs(p["delta"]) for p in pairs)
    q = lambda f: ds_[int(f * (len(ds_) - 1))]  # noqa: E731
    print(f"|delta| p10={q(.1):.1%} p25={q(.25):.1%} median={q(.5):.1%} "
          f"p75={q(.75):.1%} p90={q(.9):.1%} max={ds_[-1]:.1%}")
    ncon = sum(1 for p in pairs if not p["nbsp"])
    print(f"conservative (no NBSP): {ncon} / {len(pairs)}")
    nt = sorted(p["n_a"] for p in pairs)
    print(f"token count A: median={nt[len(nt)//2]} min={nt[0]} max={nt[-1]}")

    print("\nEXAMPLE, light bundle:")
    ex = next((p for p in pairs if not p["nbsp"]), pairs[0])
    print("  A :", repr(ex["a"][:200]))
    print("  A':", repr(ex["b"][:200]))
    print(f"  {ex['n_a']} -> {ex['n_b']} ({ex['delta']:+.1%})")

    json.dump(pairs, open("pairs.json", "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"\nwrote pairs.json ({len(pairs)})")


if __name__ == "__main__":
    main()
