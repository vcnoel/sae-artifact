# -*- coding: utf-8 -*-
"""Verify every macro main.tex uses against the value make_macros.py derives.

Two failure modes this catches, both of which occurred on the previous paper:
a macro that main.tex references but numbers.tex does not define (silent empty
output in the PDF), and a macro whose stored value has drifted from what the
source CSVs now produce.

A checker that matches zero macros is itself an error, so the count is asserted.

PROSE FILES. main.tex gets this for free because it cites macros rather than
literals. PREREG.md and OUTLINE.md quote numbers directly, and did drift: the
D3 table recorded a bootstrap CI of [1.159, 2.459] for an interval the source
now yields as [1.171, 2.498]. Markdown cannot expand macros, so a literal there
is bound to its macro with an inline HTML comment, which renders as nothing:

    | KL / unit perturbation norm | 1.687 <!--{NormFlipHL}--> | ...

Every annotated literal is checked against its macro. Unannotated numbers in
those files are NOT checked -- this catches drift in claims someone chose to
track, and the fix for an untracked claim is to annotate it.

    python src/check_numbers.py
"""
import io
import os
import re
import subprocess
import sys

# idempotent: importing a module that also wraps stdout would otherwise close
# the already-wrapped stream (ValueError: I/O operation on closed file)
if getattr(sys.stdout, "encoding", "").lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

# The manuscript to check. A path may be given on the command line so that a
# second target (paper/main-v2.tex) is checked by the same pipeline rather than
# by a copy of it that can drift.
TEX = sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"
NUM = "paper/numbers.tex"
PROSE = ("PREREG.md", "paper/OUTLINE.md", "README.md")
# a literal followed by its macro binding: 1.687 <!--{NormFlipHL}-->
# the unit/markup between the number and the comment ("16.9%", "23.6x",
# "**16.9**") must be skipped, or the binding silently fails to match and the
# checker reports success while checking nothing
BOUND = re.compile(r"([-+]?\d[\d,]*\.?\d*)\s*[%x*)\]]*\s*<!--\{(\w+)\}-->")


def check_prose(fresh):
    """Literals in markdown that are annotated with the macro they mirror."""
    bad, n = [], 0
    for path in PROSE:
        try:
            text = io.open(path, encoding="utf-8").read()
        except OSError:
            continue
        for literal, macro in BOUND.findall(text):
            n += 1
            if macro not in fresh:
                bad.append(f"{path}: {literal} bound to unknown macro "
                           f"{{{macro}}}")
            elif literal.replace(",", "") != fresh[macro].replace(",", ""):
                bad.append(f"{path}: prose says {literal}, "
                           f"{{{macro}}} now yields {fresh[macro]}")
    return bad, n


ABSTRACT = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
# Numerals in the abstract that are NOT results, each with the reason. Anything
# not listed here must be a macro. Keep this list short and justified: the point
# of the rule is that a result typed as a literal is invisible to the macro
# check, which is exactly how the E16-E24 numbers went unverified.
ABSTRACT_OK = {
    "95": "confidence level, not a measured value",
    "2": "the '2' of Gemma-2-2B",
    "3": "the '3' of Gemma-3-1B",
    "1": "the '1' of Gemma-3-1B",
    "3.5": "the '3.5' of Qwen3.5, a model name",
    "6": "six arms -- a design constant, stated in the same sentence",
}


POPS = "paper/populations.tsv"
# Sentences that legitimately name two populations because the contrast IS the
# point. Each entry is a distinctive fragment; keeping the list short is the
# discipline, since anything here has opted out of the check.
POP_OK = (
    "even restricting to the latents a pair encodes almost identically",
    "Agreement within that population varies with similarity",
    "Weighting each band's agreement by its share",
)


def check_populations(tex):
    """Flag sentences quoting macros that describe different populations.

    check_numbers verifies PROVENANCE: each macro equals what its source
    produces. It cannot see that two correct macros describe different things,
    which is how a matrix-wide maximum came to sit beside a single-band figure.
    This is a review list, not a failure: most mixes are fine, and the ones
    that are not have all looked exactly like the ones that are.
    """
    if not os.path.exists(POPS):
        return []
    pop = {}
    for line in io.open(POPS, encoding="utf-8"):
        if line.startswith("#") or "\t" not in line:
            continue
        k, v = line.rstrip("\n").split("\t", 1)
        pop[k] = v
    body = tex.split("\\appendix")[0]
    body = re.sub(r"\s+", " ", body)
    out = []
    for sent in re.split(r"(?<=[.:;])\s+(?=[A-Z\\])", body):
        tags = {pop[m] for m in re.findall(r"\\([A-Z][A-Za-z0-9]*)", sent)
                if m in pop and pop[m] != "unclassified"}
        # Compare on DOMAIN and SLICE only, not model or corpus. Putting
        # Gemma-2 beside Gemma-3, or 96 beside 1536, is a designed contrast and
        # flagging it buries the real cases in noise -- the first version
        # produced 14 flags of which 10 were deliberate. What must not be mixed
        # silently is the slice: a matrix-wide average against a single band, a
        # full sample against a balanced cube, a band against a weighting.
        keys = {(t.split("/")[0], t.split("/")[-1]) for t in tags}
        if len(keys) > 1 and not any(f in sent for f in POP_OK):
            out.append((sent.strip(), {"%s .. %s" % k for k in keys}))
    return out


def check_abstract(tex):
    """Every numeral in the abstract must come from a macro.

    Mismatch was never the failure mode here -- invisibility was. A hand-typed
    literal matches nothing, so the macro loop above skips it silently and the
    checker reports success over numbers it never saw.
    """
    m = ABSTRACT.search(tex)
    if not m:
        print("ERROR: no abstract found; the abstract rule is checking "
              "nothing.", file=sys.stderr)
        raise SystemExit(1)
    body = m.group(1)
    # drop macro invocations first, so digits inside a macro NAME
    # (\GTwoSThreeEightyFourGain) are not mistaken for typed numerals
    body = re.sub(r"\\[A-Za-z]+", " ", body)
    bad = []
    for tok in re.findall(r"\d[\d,]*\.?\d*", body):
        if tok not in ABSTRACT_OK:
            bad.append(f"abstract: bare numeral {tok!r} is not a macro")
    return bad


def main():
    # regenerate into a temp copy and diff, so a stale numbers.tex is caught
    if not os.path.exists(NUM):
        r0 = subprocess.run([sys.executable, "src/make_macros.py"],
                            capture_output=True, text=True)
        if r0.returncode != 0:
            print(r0.stdout[-2000:], r0.stderr[-2000:], file=sys.stderr)
            raise SystemExit("ERROR: make_macros.py failed")
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

    if not os.path.exists(TEX):
        drift = sorted(k for k in fresh if stored.get(k) != fresh[k])
        print(f"macros defined: {len(fresh)}")
        print(f"{TEX} not found: manuscript cross-check skipped")
        prose_bad, n_prose = check_prose(fresh)
        print(f"macro-bound literals in prose files: {n_prose}")
        if drift or prose_bad:
            print("MISMATCH:", file=sys.stderr)
            for k in drift:
                print(f"  {k}: stored {stored.get(k)}, regenerated {fresh[k]}",
                      file=sys.stderr)
            for x in prose_bad:
                print("  " + x, file=sys.stderr)
            raise SystemExit(1)
        print("macros regenerate identically; prose bindings match")
        return
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

    mixed = check_populations(body)
    if mixed:
        print(f"\nSENTENCES MIXING POPULATIONS ({len(mixed)}) -- review, "
              f"not an error:")
        for s, tags in mixed:
            print(f"  {s[:104]}")
            for t in sorted(tags):
                print(f"      {t}")
        print("  Each macro above is individually correct. Quoting two "
              "populations\n  in one sentence is how a matrix-wide maximum "
              "came to sit beside a\n  single-band figure, and a 4% tail came "
              "to be read as the centre.\n  Confirm each sentence names the "
              "population it means.")

    abs_bad = check_abstract(body)
    if abs_bad:
        print("BARE NUMERALS IN ABSTRACT:", file=sys.stderr)
        for b in abs_bad:
            print("  " + b, file=sys.stderr)
        print("  A literal here is invisible to the macro check above, which is"
              "\n  how the E16-E24 numbers sat unverified for six turns while"
              "\n  this checker reported success. Bind it to a macro, or add it"
              "\n  to ABSTRACT_OK with the reason it is not a result.",
              file=sys.stderr)
        raise SystemExit(1)

    prose_bad, n_prose = check_prose(fresh)
    print(f"macro-bound literals in prose files: {n_prose}")
    # Same reasoning as the macro guard above: the first version of BOUND did
    # not skip the unit between the number and the comment ("16.9%"), so it
    # silently matched 12 of 22 annotations and still reported success.
    n_annot = sum(io.open(p, encoding="utf-8").read().count("<!--{")
                  for p in PROSE if os.path.exists(p))
    if n_prose < n_annot:
        print(f"ERROR: {n_annot} macro bindings are written in the prose files "
              f"but only {n_prose} parsed. BOUND is not matching them all.",
              file=sys.stderr)
        raise SystemExit(1)
    bad += prose_bad

    if bad:
        print("MISMATCH:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        raise SystemExit(1)
    print("all referenced macros match their source values")


if __name__ == "__main__":
    main()
