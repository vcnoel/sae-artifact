# -*- coding: utf-8 -*-
"""Is Gemma-2's +0.110 / +0.110 across 384 and 1536 a plateau or a coincidence?

The fixed-latent curve (E25b) reports the same paired gain at two corpus sizes
differing by a factor of four, on 62 latents. A tie that exact invites the
reading that the effect has reached a ceiling. Before anything is built on
that, the components underneath the two rows have to be compared: a plateau
means the same variance structure at both corpora, whereas two different
structures whose DIFFERENCE happens to coincide is arithmetic, not a ceiling.

Prints v_a, v_ab, v_e and E rho^2 for both designs at each corpus on the
identical latent set, plus the difference of differences.
"""
import argparse
import io
import sys

if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

import numpy as np

from boundary_check import comp_cube
from moderator import build as build_cubes
from corpus_curve import CURVE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2-2b")
    ap.add_argument("--out", default="results/tie_check.txt")
    a = ap.parse_args()
    lines = []

    def p(s=""):
        print(s, flush=True)
        lines.append(s)

    built = {}
    for n, pap, shp in CURVE[a.model]:
        _, _, ca, cs, fids = build_cubes(pap, shp)
        built[n] = (ca, cs, list(fids))
    sizes = [n for n, _, _ in CURVE[a.model]]
    common = set(built[sizes[0]][2])
    for n in sizes[1:]:
        common &= set(built[n][2])
    common = sorted(common)

    p("=" * 96)
    p(f"{a.model}: fixed latent set, n={len(common)}")
    p("=" * 96)
    p(f"{'n_seq':>6}{'design':>10}{'v_a':>10}{'v_ab':>10}{'v_e':>10}"
      f"{'n/cell':>8}{'Erho2':>9}")
    E = {}
    for n in sizes:
        ca, cs, fl = built[n]
        idx = np.array([fl.index(f) for f in common])
        row = {}
        for lab, cube in (("per-arm", ca[idx]), ("shared", cs[idx])):
            c = comp_cube(cube)
            row[lab] = c
            p(f"{n:>6}{lab:>10}{c['v_a']:>10.5f}{c['v_ab']:>10.5f}"
              f"{c['v_e']:>10.5f}{c['n']:>8}{c['erho2']:>9.4f}")
        E[n] = row
        p("")

    p("=" * 96)
    p("THE TIE")
    p("=" * 96)
    for n in sizes[1:]:
        d = E[n]["shared"]["erho2"] - E[n]["per-arm"]["erho2"]
        p(f"  n_seq {n:>5}: shared {E[n]['shared']['erho2']:.4f} "
          f"- per-arm {E[n]['per-arm']['erho2']:.4f} = {d:+.4f}")
    a1, a2 = sizes[-2], sizes[-1]
    d1 = E[a1]["shared"]["erho2"] - E[a1]["per-arm"]["erho2"]
    d2 = E[a2]["shared"]["erho2"] - E[a2]["per-arm"]["erho2"]
    p("")
    p(f"  difference of differences ({a2} minus {a1}) = {d2 - d1:+.5f}")
    p("")
    same = []
    for lab in ("per-arm", "shared"):
        for k in ("v_a", "v_ab", "v_e", "erho2"):
            x, y = E[a1][lab][k], E[a2][lab][k]
            rel = abs(x - y) / max(abs(x), 1e-12)
            same.append(rel < 0.01)
            p(f"  {lab:<8} {k:<6} {x:>10.5f} -> {y:>10.5f}   "
              f"relative change {rel:>7.2%}")
    p("")
    if all(same):
        p("  VERDICT: every component is within 1% across a fourfold corpus")
        p("  increase. That is a genuine plateau and can be described as one.")
    else:
        p("  VERDICT: the components are NOT the same at the two corpora. Both")
        p("  designs' coefficients move, and the tie is a coincidence in their")
        p("  DIFFERENCE, not a ceiling in the effect. The correct statement is")
        p("  that the fixed-set gain rises from the smallest corpus and is")
        p("  unchanged thereafter within an interval far wider than the")
        p("  movement -- which is not evidence of a ceiling, and no ceiling")
        p("  should be claimed.")

    with io.open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
