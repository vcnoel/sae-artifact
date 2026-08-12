# arXiv packaging: the format used for *A Probe Direction Is a Property of Its Prompt*

Dropped here because `paper/main.tex` notes that the ICLR 2027 style was not available locally and
`article` is standing in. It is available: `style/` has it, plus the `.bst` and the `fancyhdr.sty`
the style requires.

Everything here was used to produce a real submission tarball, and each gate exists because it
caught something.

## The format

One source, two targets, selected by `\ifanon`. **Not two copies of `main.tex`** — the failure this
guards against is a number corrected in one version and not the other.

| target | authors | header |
|---|---|---|
| default | anonymous | "Under review as a conference paper at ICLR 2027" |
| `\def\arxivbuild{}` | real, from `authors.tex` | none |

```latex
\newif\ifanon
\anontrue
\ifdefined\arxivbuild\anonfalse\fi

\documentclass{article}
\usepackage{iclr2027_conference,times}
\ifanon\else\iclrfinalcopy\fi

\ifanon
  \author{Anonymous authors\\Paper under double-blind review}
\else
  \input{authors}
\fi

\begin{document}
\maketitle
\ifanon\else\lhead{}\chead{}\rhead{}\fi
```

```bash
pdflatex main                              # anonymous ICLR target
pdflatex "\def\arxivbuild{}\input{main}"   # attributed preprint
```

Three traps in that block, all of which cost a build:

1. **The flag must be a `\def`, not `\anonfalse`.** The command line executes *before* `main.tex` is
   read, so a `\newif` switch does not exist yet and setting it is an undefined control sequence —
   which fails quietly and gives you the anonymous PDF anyway.
2. **`\iclrfinalcopy` sets the head to "Published as a conference paper at ICLR 2027."** That claims
   an acceptance you do not have. Use it for its author handling and clear the head immediately
   after `\maketitle`, which is where the style sets it. Then *check the rendered pages*, not the
   source.
3. **The style hard-codes the anonymous author block** in its default mode and ignores `\author`
   entirely, and adds review line-number rulers. Both disappear under `\iclrfinalcopy`.

Also: this style never prints `\@date`, so `\date{...}` renders nothing. Put a date inside the
author block if you want one visible — or omit it, since arXiv stamps its own.

## The scripts

| script | what it does |
|---|---|
| `make_arxiv.py` | stages sources, strips comments, builds the `.bbl`, writes the tarball |
| `arxiv_metadata.py` | `METADATA.txt` with the abstract resolved from your macro file |
| `fill_dois.py` | adds DOIs to `refs.bib`, Crossref-verified or derived |

Paths at the top of each assume `paper/` and `scripts/`; adjust the `TEX`, `SUPPORT` and `FIGS`
lists for this repo's layout. `arxiv_metadata.py` reads `generated_numbers.tex` — point it at
`paper/numbers.tex` here.

### Comment stripping is not optional

The uploaded LaTeX is publicly downloadable, comments included. On the probe paper the scan removed
71, including two that recorded a *withdrawn claim*. Rules that preserve LaTeX semantics:

- a full-line comment is **deleted**, not blanked — a blank line is a paragraph break;
- a trailing comment is truncated to a bare `%`, which preserves the line-join it may be doing;
- `\%` is text and is left alone.

### DOI verification

`fill_dois.py` gates every Crossref match three ways, and each gate was added after a wrong DOI got
through at a title match of 1.00:

- **title similarity alone accepted book reviews.** A review carries the reviewed work's exact
  title, so a 1974 JSTOR item matched Cronbach 1972 perfectly.
- **record type must match the entry type, as a whitelist.** Blacklisting books still let
  "False-Positive Psychology" match a Crossref record of type `dataset` — an APA deposit, not the
  article.
- **years must both exist and be equal.** Treating a missing year as agreement then accepted a 2014
  paper for a 2011 article.

arXiv DOIs are *derived*, not looked up: `10.48550/arXiv.<id>` is a deterministic mapping.
Everything that fails the gates is left empty and reported. A wrong DOI resolves to somebody else's
paper and is worse than an absent one, because it looks checked.

## Verification: build in a clean room

The working directory contains files the tarball does not, and their absence does not always error.

```bash
mkdir /tmp/cr && cd /tmp/cr && tar xzf .../submission.tar.gz
pdflatex main.tex && pdflatex main.tex     # twice, no bibtex, no latexmk
```

Check: zero errors, **references actually rendered** (proving the `.bbl` shipped), no `[?]`, all
figures, expected page count, and zero venue mentions across every page.

Then diff against the working build. Compare as a *token multiset*, not line-by-line: removing the
ruler margin makes `pdftotext` serialise table columns differently, which produces a large diff with
no content difference.

On the probe paper an attributed build made with `-jobname` could not find a matching `.bbl`,
rendered **no bibliography at all**, reported zero errors, and came out 18 pages instead of 20. The
page count was the only thing that caught it.

## What arXiv actually requires (checked 2026-08-11)

Two widely-repeated rules are out of date:

- **BibTeX does run.** arXiv detects and runs bibtex, or biblatex with an auto-selected backend, and
  accepts either a `.bbl` or your `.bib`. Shipping the `.bbl` still removes the question.
- **Do not set `\pdfoutput=1`.** Current guidance is explicitly against it.

TeX Live 2025 is the default and nothing beyond it is provided, which is why `style/` has to travel
inside the tarball.
