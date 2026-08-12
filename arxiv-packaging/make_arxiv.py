"""Build the arXiv submission tarball from the paper sources.

Three facts drive everything here:

  * the uploaded LaTeX is PUBLICLY DOWNLOADABLE, comments included, and this project's comments
    record withdrawn claims, error causes and internal tooling -- the single worst category to ship;
  * arXiv runs `pdflatex main.tex` with NO flags, so the staged main.tex must default to the
    attributed target rather than the anonymous one the repository builds by default;
  * arXiv provides only TeX Live, so any style file not in it must travel in the tarball.

Comment stripping preserves LaTeX semantics rather than deleting every `%`:

  full-line comment   the line is REMOVED, not blanked -- a blank line is a paragraph break
  trailing comment    truncated to a bare `%`, which keeps the line-join it may be performing
  \\%                  an escaped percent is text, not a comment, and is left alone

    python scripts/make_arxiv.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "arxiv"
STAGE = OUT / "src"

TEX = ["main.tex", "generated_numbers.tex", "statements.tex", "appendix.tex",
       "wrappers_table.tex", "authors.tex"]
SUPPORT = ["iclr2027_conference.sty", "iclr2027_conference.bst", "fancyhdr.sty"]
FIGS = ["fig1_wrappers.pdf", "fig2_floor.pdf", "fig3_variance.pdf", "fig4_dstudy.pdf",
        "fig5_signdist.pdf"]

# Why each comment is removed, for STRIPPED.md. Order matters: first match wins.
CATEGORY = [
    (re.compile(r"withdraw|retract|no longer|was wrong|used to say", re.I),
     "withdrawn or superseded claim"),
    (re.compile(r"ICLR|desk-reject|double-blind|conference|reviewer", re.I),
     "venue, review process or reviewer reference"),
    (re.compile(r"\.py\b|script|pipeline|generator|ORDER list|fill_numbers|backfill|check_",
                re.I), "internal tooling or build-process reference"),
    (re.compile(r"[0-9a-f]{12}|HEAD|hash|mtime|\.json", re.I),
     "build provenance (hashes, inputs, repository state)"),
    (re.compile(r"caught|error|bug|failed|mistake|guard", re.I),
     "internal error-record note"),
]


def categorise(text):
    for rx, name in CATEGORY:
        if rx.search(text):
            return name
    return "editorial or explanatory note"


def strip_comments(text, counter):
    """Remove comment CONTENT while preserving LaTeX line semantics."""
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            body = stripped.lstrip("%").strip()
            if body:
                counter[categorise(body)] += 1
            continue                                  # drop the line entirely
        # find the first unescaped %
        i, j = None, 0
        while True:
            k = line.find("%", j)
            if k == -1:
                break
            if k > 0 and line[k - 1] == "\\":         # \% is text
                j = k + 1
                continue
            i = k
            break
        if i is None:
            out.append(line)
            continue
        body = line[i + 1:].strip()
        if body:
            counter[categorise(body)] += 1
        out.append(line[:i] + "%")                    # keep the join, drop the words
    return "\n".join(out)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    # Clear only the STAGING tree. An earlier version removed all of arxiv/, which deleted
    # CLEANROOM.md and STRIPPED.md -- hand-written deliverables that live beside the tarball -- on
    # the next rebuild. Rebuilding an artifact must not destroy the report describing it.
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    (STAGE / "figures").mkdir()

    counter = Counter()
    for name in TEX:
        src = (PAPER / name).read_text(encoding="utf-8")
        if name == "main.tex":
            # arXiv runs pdflatex with no flags, so the DEFAULT here must be the attributed target.
            # Leaving \anontrue would silently upload the anonymous, venue-headed, ruler-marked
            # version -- the exact artifact the directive forbids.
            src2 = src.replace("\\anontrue\n\\ifdefined\\arxivbuild\\anonfalse\\fi",
                               "\\anonfalse  % attributed preprint: this IS the arXiv target")
            if src2 == src:
                raise SystemExit("could not force the attributed target in main.tex; "
                                 "the \\ifanon block changed shape -- fix this script")
            src = src2
        (STAGE / name).write_text(strip_comments(src, counter), encoding="utf-8")

    for name in SUPPORT:
        p = PAPER / name
        if p.exists():
            shutil.copy2(p, STAGE / name)
    for f in FIGS:
        shutil.copy2(PAPER / "figures" / f, STAGE / "figures" / f)

    # The .bbl must be built here: arXiv may run bibtex, but shipping the .bbl removes the question.
    # It is produced from the STAGED tree so it matches what is uploaded, not the working directory.
    shutil.copy2(PAPER / "refs.bib", STAGE / "refs.bib")
    env_run = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    subprocess.run(env_run, cwd=STAGE, capture_output=True)
    subprocess.run(["bibtex", "main"], cwd=STAGE, capture_output=True)
    for _ in range(2):
        subprocess.run(env_run, cwd=STAGE, capture_output=True)
    if not (STAGE / "main.bbl").exists():
        raise SystemExit("bibtex produced no main.bbl; cannot ship a paper without references")
    (STAGE / "refs.bib").unlink()          # redundant once the .bbl is present

    keep = set(TEX) | set(SUPPORT) | {"main.bbl"}
    for p in sorted(STAGE.iterdir()):
        if p.is_file() and p.name not in keep:
            p.unlink()

    tar = OUT / "eval-awareness-spectral-arxiv.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(STAGE).as_posix())

    print(f"  staged {len(list(STAGE.rglob('*')))} paths -> {tar.name} "
          f"({tar.stat().st_size / 1024:.0f} KB)")
    for cat, n in counter.most_common():
        print(f"    {n:>3}  {cat}")
    (OUT / "_strip_counts.txt").write_text(
        "\n".join(f"{n}\t{c}" for c, n in counter.most_common()), encoding="utf-8")


if __name__ == "__main__":
    main()
