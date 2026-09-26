# -*- coding: utf-8 -*-
"""Audit the manuscript's prose rules and print every hit with file and line.

Three rules, each of which a reviewer or collaborator caught by hand before
this script existed:

  semicolon      a semicolon in prose
  dash           "--" used as a parenthetical dash rather than inside a name
  open-compound  an attributive compound modifier written open ("per arm
                 positions", "held out set") where standard English hyphenates
                 it ("per-arm positions", "held-out set")

The compound rule works from an explicit list, OPEN below, of the compounds
this manuscript uses. A phrase from the list is flagged only in attributive
position, that is when the next word is not a preposition, a verb, a
conjunction or the end of a clause, so "the set is held out" and "measured per
arm." pass while "a per arm estimate" does not. A compound modifier after an
"-ly" adverb ("a carefully chosen prompt") is open by rule and is never listed.
Names keep their own spelling (gemma-2-2b, TopK, WikiText-103), and the list
holds no name, so the rule cannot touch one.

Control sequences and inline maths are stripped first, because a range inside
$0.78--0.91$ is not a prose defect.

Two properties make the result trustworthy, and both are checked here rather
than assumed. The script runs from a file, so no shell eats a backslash and
turns an exclusion pattern into one that matches every line. And SELFTEST holds
one line known to violate each rule, and one known to pass the compound rule
in predicative position, asserted at startup, so a pattern that has silently
stopped matching (or started matching everything) is caught instead of
reporting a clean paper.

    python src/voice_audit.py              # audits paper/main-v2.tex
    python src/voice_audit.py paper/x.tex  # audits the files named
"""
import io
import os
import re
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

DEFAULT = ["paper/main-v2.tex"]

# Attributive compounds the manuscript uses, in their OPEN spelling. Each is
# hyphenated when it precedes the noun it modifies. Longest first, so that
# "single top position" is reported rather than its shorter tail.
OPEN = [
    "second to ninety eighth", "mutual nearest neighbour", "latent by arm",
    "like for like", "top activating", "ablation based", "interchange based",
    "number generating", "frequency matched", "activation frequency decile",
    "single top position", "single neuron", "single latent", "single token",
    "cross dictionary", "decoder free", "soft frozen", "arm symmetric",
    "inter arm", "minimum gated rule", "min gated", "non selective",
    "non monotone", "near duplicate", "near universal", "per arm",
    "per latent", "per feature", "per layer", "per vocab", "within cell",
    "between latent", "within paper", "within layer", "between sample",
    "fixed latent", "fixed set", "live set", "module level", "latent level",
    "latent wise", "whole dictionary", "one arm", "two arm", "six arm",
    "six column", "special token", "random effects", "held out", "zero shot",
    "open weight", "long context", "pre fix", "re measuring", "re examine",
    "re examination", "released dictionary", "production scale",
    "position selection", "shared position",
]
# A following word that makes the phrase predicative or prepositional rather
# than attributive: the compound then stays open.
NOT_NOUN = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "and", "or", "but", "nor", "than", "then", "that", "which", "who",
    "of", "in", "at", "on", "to", "by", "for", "with", "from", "as", "into",
    "across", "over", "under", "within", "between", "against", "per", "via",
    "the", "a", "an", "its", "their", "this", "these", "those", "each",
    "every", "it", "they", "we", "so", "since", "because", "while", "where",
    "when", "if", "not", "no", "also", "only", "here", "there", "do", "does",
    "rather", "alone", "itself", "themselves", "above", "below",
}
# A count or size joined to its unit before a noun: "a 512-token sequence",
# "the 96-sequence corpus", "a 12M-token budget". Inline maths has already
# collapsed to the digit 0 by the time this runs, so a number reads as "0".
NUMERIC_OPEN = re.compile(r"\b(\d+[MKB]?)\s+(token|sequence|layer)"
                          r"\s+([A-Za-z]+)")
# A prefix written as a separate word ("non monotone", "re examine") is wrong in
# any position, attributive or not, so these are flagged unconditionally.
PREFIX_OPEN = re.compile(r"\b(non|re|pre|co|sub)\s+([a-z]{3,})\b")
PREFIX_OK = {"re": {"the", "a", "an"}}
_OPEN_RE = re.compile(
    r"\b(" + "|".join(re.escape(p).replace(r"\ ", r"\s+")
                      for p in sorted(OPEN, key=len, reverse=True))
    + r")\b(?=(\s+[A-Za-z]+)?)", re.I)

MATH = re.compile(r"\$[^$]*\$|\\\[[^]]*?\\\]|\\\([^)]*?\\\)", re.S)
VERB = re.compile(r"\\(?:texttt|url|href|cite[a-z]*|ref|label|includegraphics"
                  r"|eqref)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}")
CTRL = re.compile(r"\\[a-zA-Z@]+\*?\s*")
ENVS = re.compile(r"\\begin\{(equation|align|tabular|thebibliography)\*?\}"
                  r".*?\\end\{\1\*?\}", re.S)
COMMENT = re.compile(r"(?<!\\)%.*$")

DASH = re.compile(r"--")
SEMI = re.compile(r";")

SELFTEST = [
    ("semicolon", "The component collapses; the coefficient rises.", True),
    ("dash", "Every arm sees the same tokens -- the design requires it.", True),
    ("open-compound", "Measured at the top activating token of each latent.",
     True),
    ("open-compound", "Each estimate is taken on a held out set.", True),
    ("open-compound", "Each arm sees a $12$M token budget.", True),
    ("open-compound", "The coefficient is non monotone on both.", True),
    ("open-compound", "Across $15$ arm pairs the share is stable.", False),
    # predicative and hyphenated forms must pass
    ("open-compound", "The set is held out and the rest is used.", False),
    ("open-compound", "Measured at the top-activating token of each latent.",
     False),
    ("open-compound", "Positions are chosen per arm.", False),
]


def strip(line):
    """Remove everything that is not prose from one line."""
    line = COMMENT.sub("", line)
    line = VERB.sub(" ", line)
    # Maths collapses to a digit, not a space, so that a range written between
    # two maths groups ($0.50$--$0.60$) still reads as a range and is kept by
    # the adjacency test below rather than counted as a parenthetical dash.
    line = MATH.sub("0", line)
    line = CTRL.sub(" ", line)
    return line.replace("{", " ").replace("}", " ").replace("~", " ")


def hits(text):
    """Return (offset, rule, matched text) for every prose rule the text
    breaks. The text may span lines, so a compound broken across a source line
    ("top" then "activating" on the next line) is still seen."""
    found = []
    for m in SEMI.finditer(text):
        found.append((m.start(), "semicolon", ";"))
    for m in DASH.finditer(text):
        before = text[m.start() - 1:m.start()]
        after = text[m.end():m.end() + 1]
        # A range (0--0) and a name (Hodges--Lehmann) close up against their
        # neighbours. A parenthetical dash takes a space on at least one side,
        # and that is the only form the rule forbids.
        if before.isalnum() and after.isalnum():
            continue
        found.append((m.start(), "dash",
                      text[max(0, m.start() - 12):m.end() + 12].strip()))
    for m in _OPEN_RE.finditer(text):
        # a hyphen on either side means the compound is already joined to a
        # neighbour ("soft-frozen", "per-arm"), which the \b match would miss
        if text[m.start() - 1:m.start()] == "-" or text[m.end():m.end() + 1] == "-":
            continue
        nxt = m.group(2)
        if nxt is None:
            continue                      # end of line or clause: predicative
        if nxt.strip().lower() in NOT_NOUN:
            continue
        found.append((m.start(), "open-compound",
                      " ".join((m.group(1) + nxt).split())))
    for m in NUMERIC_OPEN.finditer(text):
        if text[m.end(2):m.end(2) + 1] == "-" or m.group(3).lower() in NOT_NOUN:
            continue
        found.append((m.start(), "open-compound",
                      " ".join(m.group(0).split())))
    for m in PREFIX_OPEN.finditer(text):
        if m.group(2) in PREFIX_OK.get(m.group(1), set()):
            continue
        found.append((m.start(), "open-compound",
                      " ".join(m.group(0).split())))
    return found


def selftest():
    for rule, line, should in SELFTEST:
        got = {r for _, r, _ in hits(strip(line))}
        if should and rule not in got:
            sys.exit(f"SELFTEST FAILED: rule '{rule}' no longer matches its "
                     f"known violation: {line!r} -> {sorted(got)}")
        if not should and rule in got:
            sys.exit(f"SELFTEST FAILED: rule '{rule}' matches a line it must "
                     f"pass: {line!r}")


def main():
    selftest()
    paths = sys.argv[1:] or DEFAULT
    total = 0
    for path in paths:
        if not os.path.exists(path):
            sys.exit(f"ERROR: no such file: {path}")
        body = io.open(path, encoding="utf-8").read()
        body = ENVS.sub(lambda m: "\n" * m.group(0).count("\n"), body)
        prose = "\n".join(strip(line) for line in body.splitlines())
        for off, rule, what in sorted(hits(prose)):
            n = prose.count("\n", 0, off) + 1
            print(f"{path}:{n}: {rule:13s} {what}")
            total += 1
    print(f"\n{total} hit(s) across {len(paths)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
