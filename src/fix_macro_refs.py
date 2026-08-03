# -*- coding: utf-8 -*-
"""Rewrite digit-bearing macro references in main.tex to the sanitised names.

LaTeX control sequences may contain letters only, so \\DepthRatioL12 is invalid
and reports 'Missing \\begin{document}' rather than anything informative.
make_macros.py maps digits to words per character, so 12 becomes OneTwo.

Plain string substitution, no regex: escaping a literal backslash through the
shell, Python and a regex engine at once produced three separate failures.
"""
import io

PAIRS = [
    ("SDRatioDrop1", "SDRatioDropOne"),
    ("DepthRatioL12", "DepthRatioLOneTwo"),
    ("DepthRatioL19", "DepthRatioLOneNine"),
    ("DepthRatioL24", "DepthRatioLTwoFour"),
    ("DepthRatioL5", "DepthRatioLFive"),
    ("DepthNL12", "DepthNLOneTwo"),
    ("DepthNL19", "DepthNLOneNine"),
    ("DepthNL24", "DepthNLTwoFour"),
    ("DepthNL5", "DepthNLFive"),
    ("CurveHL12M", "CurveHLOneTwoM"),
    ("CurveHL3M", "CurveHLThreeM"),
    ("CurveHL6M", "CurveHLSixM"),
    ("CurveHL9M", "CurveHLNineM"),
    ("Top5MassGS", "TopFiveMassGS"),
]

s = io.open("paper/main.tex", encoding="utf-8").read()
done = []
for old, new in PAIRS:            # longest-first ordering matters: L12 before L5
    tok = chr(92) + old
    if tok in s:
        s = s.replace(tok, chr(92) + new)
        done.append(old)
io.open("paper/main.tex", "w", encoding="utf-8").write(s)
print(f"renamed {len(done)}: {', '.join(done)}")
