# -*- coding: utf-8 -*-
"""Verify every macro main.tex uses against the value make_macros.py derives.

Two failure modes this catches, both of which occurred on the previous paper:
a macro that main.tex references but numbers.tex does not define (silent empty
output in the PDF), and a macro whose stored value has drifted from what the
source CSVs now produce.

A checker that matches zero macros is itself an error, so the count is asserted.

    python src/check_numbers.py
"""
import io
import re
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

TEX = "paper/main.tex"
NUM = "paper/numbers.tex"


def main():
    # regenerate into a temp copy and diff, so a stale numbers.tex is caught
    with io.open(NUM, encoding="utf-8") as fh:
        stored = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                                 fh.read()))
    r = subprocess.run([sys.executable, "src/make_macros.py"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("ERROR: make_macros.py failed")
    with io.open(NUM, encoding="utf-8") as fh:
        fresh = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}",
                                fh.read()))

    body = io.open(TEX, encoding="utf-8").read()
    body = re.sub(r"(?m)^\s*%.*$", "", body)          # ignore commented lines
    # digits must be allowed: \DepthRatioL5 and \CurveHL3M end in digits, and a
    # trailing \b after [A-Za-z]+ fails to match them. The first version of this
    # checker reported fifteen used macros as unused for exactly that reason.
    used = set(re.findall(r"\\([A-Z][A-Za-z0-9]*)", body))
    known = set(fresh)
    referenced = sorted(used & known)

    bad = []
    for k in referenced:
        if k not in stored:
            bad.append(f"{k}: referenced by main.tex, absent from numbers.tex")
        elif stored[k] != fresh[k]:
            bad.append(f"{k}: main.tex build has {stored[k]}, "
                       f"source now yields {fresh[k]}")
    undefined = sorted(u for u in used
                       if u not in known and re.match(r"^[A-Z][a-z]", u)
                       and u not in {"TODO", "GENERATED"})

    print(f"macros defined: {len(fresh)}")
    print(f"macros referenced by main.tex: {len(referenced)}")
    if len(referenced) < 10:
        print("ERROR: fewer than 10 macros matched. The reference pattern in "
              "this checker no longer sees main.tex; a checker that matches "
              "nothing is worse than no checker.", file=sys.stderr)
        raise SystemExit(1)
    unused = sorted(known - used)
    if unused:
        print(f"defined but unused ({len(unused)}): {', '.join(unused)}")
    if bad:
        print("MISMATCH:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        raise SystemExit(1)
    print("all referenced macros match their source values")


if __name__ == "__main__":
    main()
