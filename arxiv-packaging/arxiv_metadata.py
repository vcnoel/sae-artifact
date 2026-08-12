"""Emit arxiv/METADATA.txt, with the abstract resolved from the same macros the PDF prints.

arXiv's abstract field is plain text. Retyping the abstract there is how a preprint ends up quoting
a number its own PDF contradicts, so this reads the abstract out of main.tex and substitutes macro
values from generated_numbers.tex rather than trusting anyone to copy them across.

    python scripts/arxiv_metadata.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
OUT = ROOT / "arxiv" / "METADATA.txt"
REPO = "https://github.com/vcnoel/eval-awareness-spectral"


BIB = (PAPER / "refs.bib").read_text(encoding="utf-8", errors="replace")


def macros():
    gen = (PAPER / "generated_numbers.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"newcommand\{\\(\w+)\}\{(.*)\}", gen))


def to_plain(tex, m):
    t = tex
    # macros, longest name first so \GtMT is not eaten by a prefix match
    for name in sorted(m, key=len, reverse=True):
        t = t.replace("\\" + name + "{}", m[name]).replace("\\" + name, m[name])
    t = re.sub(r"\\ensuremath\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(emph|texttt|textbf|textit)\{([^{}]*)\}", r"\2", t)
    # \citet renders as "Author (Year)" in the PDF, so dropping it would change the sentence and
    # leaving it would print a bibtex key. Resolve from refs.bib -- the same file the .bbl is
    # built from -- rather than hand-writing the names here.
    def cite(mo):
        key = mo.group(2).split(",")[0].strip()
        ent = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n@", BIB + "\n@", re.S)
        if not ent:
            return ""
        body = ent.group(1)
        au = re.search(r"author\s*=\s*[{\"](.*?)[}\"]\s*,", body, re.S)
        yr = re.search(r"year\s*=\s*[{\"]?(\d{4})", body)
        if not au or not yr:
            return ""
        first = au.group(1).split(" and ")
        name = first[0].split(",")[0].strip() if "," in first[0] else first[0].split()[-1]
        if len(first) > 2:
            name += " et al."
        elif len(first) == 2:
            second = first[1]
            name += " and " + (second.split(",")[0].strip() if "," in second
                               else second.split()[-1])
        return f"{name} ({yr.group(1)})" if mo.group(1) == "t" else f"({name} {yr.group(1)})"

    t = re.sub(r"\\cite([tp]?)\{([^}]*)\}", cite, t)
    t = t.replace("\\Erho", "Erho").replace("$\\Erho$", "Erho")
    t = t.replace(r"\log_{10}", "log10")
    t = re.sub(r"_\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\^\{?2\}?", "^2", t)
    t = re.sub(r"\$([^$]*)\$", r"\1", t)
    t = t.replace("\\%", "%").replace("\\&", "&").replace("~", " ")
    t = t.replace("---", "--").replace("``", '"').replace("''", '"')
    t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


ACCENTS = {r'\"e': "e\u0308", r"\'e": "e\u0301", r"\`e": "e\u0300", r"\^e": "e\u0302",
           r'\"i': "i\u0308", r"\'a": "a\u0301", r"\`a": "a\u0300", r"\c c": "c\u0327"}


def author_block():
    """(name, affiliation, contact, is_placeholder) read from paper/authors.tex.

    One field per line in the \\author block. Parsed rather than retyped into METADATA.txt, for the
    same reason the abstract is resolved from the macros: a value copied by hand is a value that can
    disagree with the PDF.
    """
    raw = (PAPER / "authors.tex").read_text(encoding="utf-8")
    body = re.search(r"\\author\{(.*?)\n\}", raw, re.S).group(1)
    lines = []
    for ln in body.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        ln = re.sub(r"\\texttt\{([^}]*)\}", r"\1", ln)
        # Accents BEFORE any backslash stripping, or No\"el becomes No"el. arXiv's author field
        # takes UTF-8, so the escape is resolved to the character the PDF actually shows.
        for tex, uni in ACCENTS.items():
            ln = ln.replace(tex + "{}", uni).replace("{" + tex + "}", uni).replace(tex, uni)
        # \ and \[4pt] are line breaks, not content; the optional spacing argument has to go
        # with them or it lands in the field ("...@devoteam.com [4pt]").
        ln = re.sub(r"\\\\(\[[^\]]*\])?", "", ln).replace("%", "").strip()
        ln = re.sub(r"\s+", " ", ln).strip()
        if ln:
            lines.append(ln)
    placeholder = any("[[" in x for x in lines)
    contact = next((x for x in lines if "@" in x), "")
    rest = [x for x in lines if "@" not in x]
    name = rest[0] if rest else ""
    affil = rest[1] if len(rest) > 1 else ""
    import unicodedata
    name, affil = (unicodedata.normalize("NFC", x) for x in (name, affil))
    return name, affil, contact, placeholder


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    m = macros()
    who, affil, contact, placeholder = author_block()
    if placeholder:
        author_flag = "                              *** BLOCKING: NOT YET SUPPLIED ***"
        author_lines = ("paper/authors.tex still holds its placeholder. Fill it, re-run\n"
                        "scripts/make_arxiv.py, then regenerate this file.\n")
    else:
        author_flag = ""
        author_lines = (f"  Name (as it appears in the PDF): {who}\n"
                        f"  Affiliation:                     {affil or '(not set)'}\n"
                        f"  Contact email:                   {contact}\n"
                        "  ORCID:                           (not set; add at submission time)\n"
                        "\nRead from paper/authors.tex, the same file the PDF renders, so the two\n"
                        "cannot disagree. Enter authors in this order.\n")
    src = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
    plain = to_plain(abstract, m)
    if "\\" in plain or "{" in plain:
        raise SystemExit(f"unresolved LaTeX left in the plain-text abstract:\n  {plain[:300]}")

    title = re.search(r"\\title\{([^}]*)\}", src).group(1).strip()
    # every source, not just main+appendix: fig5 is referenced from wrappers_table.tex and an
    # undercount here would understate the submission in a field a reader checks against the PDF.
    alltex = "".join((PAPER / n).read_text(encoding="utf-8")
                     for n in ("main.tex", "appendix.tex", "wrappers_table.tex", "statements.tex"))
    nfig = len(set(re.findall(r"figures/(\w+)\.pdf", alltex)))
    napp = len(re.findall(r"^\\section\{", (PAPER / "appendix.tex").read_text(encoding="utf-8"),
                          re.M)) + 1                     # +1 for the wrappers table appendix
    pages = 20
    try:
        import fitz
        pages = fitz.open(PAPER / "main.pdf").page_count
    except Exception:
        pass

    txt = f"""arXiv submission metadata -- copy/paste, one field at a time
Generated by scripts/arxiv_metadata.py; abstract numbers resolved from
paper/generated_numbers.tex, the same source the PDF prints from.

================================================================================
TITLE
================================================================================
{title}

================================================================================
AUTHORS{author_flag}
================================================================================
{author_lines}
================================================================================
ABSTRACT (plain text)
================================================================================
{plain}

================================================================================
CATEGORIES
================================================================================
Primary:     cs.LG   (Machine Learning)
Cross-list:  cs.CL   (Computation and Language)
Cross-list:  cs.AI   (Artificial Intelligence)

Confirm against the current taxonomy at arxiv.org/category_taxonomy. If you have
not posted in cs.LG before, endorsement may be required -- check BEFORE the
submission window, not during it.

================================================================================
COMMENTS FIELD
================================================================================
{pages} pages, {nfig} figures, {napp} appendices. Code, data manifests and the full
corrections record: {REPO}

================================================================================
LICENSE
================================================================================
Choose deliberately. The arXiv default (non-exclusive licence to distribute) is
the most restrictive. CC BY 4.0 is the usual choice for work intended to be
built on, and is consistent with this repository's AGPL-3.0 code licence.

  Recommended: CC BY 4.0

================================================================================
REPORT NUMBER / JOURNAL REF / DOI
================================================================================
Leave blank at preprint stage.

================================================================================
ONE-LINE CLAIM (for the author's own posts)
================================================================================
A published probe statistic for "the model knows it is being evaluated" is
mostly a property of the prompt the experimenter wrote: the same 36-wrapper
design reproduces both published signs of the scaling result.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)} ({len(plain)} char abstract, "
          f"{pages} pages, {nfig} figures, {napp} appendices)")


if __name__ == "__main__":
    main()
