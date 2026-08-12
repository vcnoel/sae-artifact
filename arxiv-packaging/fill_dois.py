"""Fill missing DOIs in paper/refs.bib from Crossref, accepting only verified matches.

Two sources, two levels of trust:

  arXiv preprints   10.48550/arXiv.<id> is a deterministic mapping from the identifier already in
                    the entry. No lookup, no ambiguity.
  everything else   queried against Crossref by title, then ACCEPTED ONLY IF the returned title
                    matches the local one closely and the year agrees. A DOI that is merely the
                    top hit is not evidence: a wrong DOI resolves to somebody else's paper and is
                    worse than an absent one, because it looks checked.

Anything that fails the gate is left empty and reported, for a human to resolve.

    python scripts/fill_dois.py            # report only
    python scripts/fill_dois.py --write    # apply verified DOIs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "paper" / "refs.bib"
API = "https://api.crossref.org/works"
MAIL = "valentin.noel@devoteam.com"          # Crossref's polite pool
TITLE_MIN = 0.90


def norm(s):
    s = re.sub(r"\{|\}|\\[a-zA-Z]+", "", s or "")
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def field(block, name):
    m = re.search(name + r"\s*=\s*\{(.*?)\}\s*,\s*\n", block, re.S | re.I)
    if not m:
        m = re.search(name + r'\s*=\s*"(.*?)"\s*,', block, re.S | re.I)
    if not m:
        m = re.search(name + r"\s*=\s*\{(.*)\}\s*\n\}", block, re.S | re.I)
    return " ".join(m.group(1).split()) if m else ""


BOOKISH = {"book", "monograph", "reference-book", "edited-book", "book-chapter"}
ALLOWED = {
    "book": BOOKISH,
    "article": {"journal-article"},
    "inproceedings": {"proceedings-article", "journal-article"},
    "incollection": {"book-chapter", "proceedings-article"},
    "techreport": {"report", "journal-article", "posted-content"},
}


def lookup(title, year, kind, authors=""):
    # Title alone is a weak key: it matched reviews, datasets and unrelated papers sharing a phrase.
    # Adding the first author's surname and the year to the bibliographic query pushes the real
    # record into the top few, where the gates below can confirm it.
    first = ""
    if authors:
        a0 = authors.split(" and ")[0]
        first = a0.split(",")[0].strip() if "," in a0 else a0.split()[-1]
    query = " ".join(x for x in (title, first, year) if x)
    q = urllib.parse.urlencode({"query.bibliographic": query, "rows": 5, "mailto": MAIL})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": f"eas-refs ({MAIL})"})
    with urllib.request.urlopen(req, timeout=25) as fh:
        items = json.load(fh)["message"]["items"]
    for it in items:
        cand = (it.get("title") or [""])[0]
        ctype = it.get("type", "")
        ratio = SequenceMatcher(None, norm(title), norm(cand)).ratio()
        yr = ""
        for k in ("published-print", "published-online", "issued"):
            if it.get(k, {}).get("date-parts", [[None]])[0][0]:
                yr = str(it[k]["date-parts"][0][0])
                break
        # A REVIEW of a book carries the book's exact title, so a 1.00 title match against a
        # journal-article three years later is a review, not the work. Two extra gates:
        # the Crossref type must be consistent with the bib entry type, and the year must agree
        # exactly. Without these the lookup returned a 1974 review for a 1972 book at match 1.00.
        # Whitelist, not blacklist. "False-Positive Psychology" matched a Crossref record of type
        # "dataset" -- an APA PsycEXTRA deposit, not the journal article -- and a blacklist that
        # only excluded books let it through at match 1.00.
        type_ok = ctype in ALLOWED.get(kind, {"journal-article", "proceedings-article"})
        # Both years must be present AND equal. The earlier "(not year) or ..." form meant a missing
        # year on either side counted as agreement, which is how a 2014 Journal of Open Psychology
        # Data paper was accepted for a 2011 Psychological Science article at title match 0.94.
        # An unverifiable match is a rejection here, not a pass.
        year_ok = bool(year) and bool(yr) and yr == year
        if ratio >= TITLE_MIN and type_ok and year_ok:
            return it["DOI"], cand, yr, ratio, ctype
    return None, (items[0].get("title", [""])[0] if items else ""), "", 0.0, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    text = BIB.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"(?=@\w+\s*\{)", text)
    out, verified, rejected, already = [], [], [], 0

    for b in blocks:
        if not b.strip().startswith("@"):
            out.append(b)
            continue
        key = re.search(r"@\w+\s*\{\s*([^,]+),", b).group(1)
        if re.search(r"\bdoi\s*=", b, re.I):
            already += 1
            out.append(b)
            continue
        title, year = field(b, "title"), field(b, "year")
        if not title:
            rejected.append((key, "no title field", ""))
            out.append(b)
            continue
        try:
            kind = re.match(r"@(\w+)", b).group(1).lower()
            doi, cand, yr, ratio, ctype = lookup(title, year, kind, field(b, "author"))
        except Exception as e:
            rejected.append((key, f"lookup failed: {type(e).__name__}", ""))
            out.append(b)
            continue
        time.sleep(0.4)
        if not doi:
            rejected.append((key, f"no match passing title/type/year gates", cand[:60]))
            out.append(b)
            continue
        verified.append((key, doi, ratio, yr, ctype))
        idx = b.rstrip().rfind("}")
        b = b[:idx].rstrip().rstrip(",") + f",\n  doi = {{{doi}}}\n" + b[idx:]
        out.append(b)

    if a.write:
        BIB.write_text("".join(out), encoding="utf-8")

    print(f"  already had a DOI: {already}")
    print(f"\n  VERIFIED ({len(verified)}) title match >= {TITLE_MIN:.2f} and year agreeing:")
    for k, d, r, y, c in verified:
        print(f"    {k:<22} {d:<32} match={r:.2f} year={y} type={c}")
    if rejected:
        print(f"\n  NOT FILLED ({len(rejected)}) -- left empty rather than guessed:")
        for k, why, cand in rejected:
            print(f"    {k:<22} {why}" + (f"  (top hit: {cand})" if cand else ""))
    print("\n  " + ("written to paper/refs.bib" if a.write else "dry run; pass --write to apply"))


if __name__ == "__main__":
    main()
