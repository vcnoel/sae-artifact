# A Reporting Standard for Sparse Autoencoder Ablation Scores: code and results

This is the anonymous repository for the ICLR 2027 submission *A Reporting Standard for Sparse
Autoencoder Ablation Scores*. It holds the training, evaluation and analysis code, and every result
file that the manuscript's numbers and figures are generated from.

## Layout

```
src/                  training (train_arms.py, train_saes.py), evaluation (eval_arms.py, eval_shared.py,
                      eval_saes.py, released_agreement.py, released_qwen35.py, scope_matrix.py), analysis
                      (variance_decomp.py, union_check.py, position_structure.py, ...), and run scripts
src/model_configs.py  base model, layer, width, k and output directory for every set of arms
src/make_macros.py    writes every number printed in the paper as a LaTeX macro, from results/
src/check_numbers.py  regenerates the macros and checks them against the stored copy, PREREG.md and,
                      if its source is given, the manuscript
src/figures_v2.py     the four figures of the paper, from results/
src/voice_audit.py    prose lint the manuscript was held to (takes a .tex path)
results/              CSV and text reports, one family of files per experiment
PREREG.md             predictions and decision rules, each written before its run
```

The six arms of a base model are six TopK dictionaries (width 16,384, k = 82) that start from one shared
initialisation and differ in a single fitting choice each: `free`, `tau080`, `tau090` (decoder soft-frozen
to its initialisation at cosine 0.80 or 0.90), `lr1e4`, `k41` and `order1`. Result files are named by
base model: no suffix or `g2` is Gemma-2-2B, `g3` Gemma-3-1B, `q35` Qwen3.5-2B, `q354b` Qwen3.5-4B,
`q359b` Qwen3.5-9B, `o2` OLMo-2-1B, `s3` SmolLM3-3B, `q35full` Qwen3.5-2B at 48M training tokens.
A corpus suffix `_s384` or `_s1536` gives the evaluation corpus size in sequences (no suffix is 96),
and `_shared` or `_union` gives the position rule (no suffix is each arm's own positions).

## Installation

Python 3.11. The results were produced with the versions below; nearby versions should work.

```bash
pip install -r requirements.txt
```

The figures use matplotlib's pgf backend and need a LaTeX installation with LuaLaTeX on the path.
Only the retraining and re-evaluation steps need a GPU.

## Reproducing every number and figure (CPU, minutes)

Run from the repository root.

```bash
python src/make_macros.py      # writes paper/numbers.tex (every printed number) and paper/populations.tsv
python src/check_numbers.py    # regenerates the macros and verifies them, and the numbers quoted in PREREG.md
python src/figures_v2.py       # writes paper/fig1_selection.pdf ... paper/fig4_ladder.pdf
python src/union_check.py      # re-derives results/union_check.{csv,txt} (the union-rule test, PREREG E28)
```

`paper/` is created on first run and is not tracked. Each macro in `paper/numbers.tex` carries the name
of the quantity it holds, so any number in the PDF can be traced to the result file and the lines of
`make_macros.py` that produce it. Given the manuscript source, `python src/check_numbers.py main.tex`
also checks every macro the manuscript cites and fails on a result typed into the abstract as a literal.

## Regenerating the results (GPU)

Every step skips work whose output already exists, so each script can be re-run after an interruption.
Set `PYTHON` to your interpreter and `HF_HOME` to a disk with room for the base models.

**Train the six arms of a base model** (WikiText-103, seed 0, 12M tokens per arm, one GPU):

```bash
mkdir -p logs
bash src/chain6.sh                                          # Gemma-2-2B  -> data/arm_*.pt
bash src/chain6_g3.sh                                       # Gemma-3-1B  -> data/g3/
bash src/chain6_q35.sh       > logs/q35_chain.log 2>&1      # Qwen3.5-2B  -> data/q35/
bash src/chain6_q354b.sh     > logs/q354b_chain.log 2>&1    # Qwen3.5-4B  -> data/q35_4b/
bash src/chain6_q359b.sh     > logs/q359b_chain.log 2>&1    # Qwen3.5-9B  -> data/q35_9b/
bash src/chain6_o2.sh        > logs/o2_chain.log 2>&1       # OLMo-2-1B   -> data/o2/
bash src/chain6_s3.sh        > logs/s3_chain.log 2>&1       # SmolLM3-3B  -> data/s3/
bash src/chain6_q35_full.sh  > logs/q35full_chain.log 2>&1  # Qwen3.5-2B, 48M tokens per arm -> data/q35_full/
# one arm by hand:
python src/train_arms.py --base qwen35-2b --tag tau080 --tau 0.80 --tokens 12000000
```

The `eval_*.sh` scripts below wait for the closing line of the matching chain log, then refuse to start
unless all six arm files exist, so they can be launched alongside training.

**Evaluate them** (WikiText-2 raw; 240 latents, 6 positions each; corpus ladder 96 / 384 / 1536
sequences; own-position, shared-position and union rules), then decompose the variance:

```bash
bash src/eval_decomp_g3.sh   # Gemma-3-1B; eval_s1536.sh and eval_union.sh add Gemma-2-2B's larger corpora
bash src/eval_q35.sh; bash src/eval_q354b.sh; bash src/eval_q359b.sh
bash src/eval_o2.sh;  bash src/eval_s3.sh;    bash src/eval_q35_full.sh
# one cell by hand:
python src/eval_arms.py --base qwen35-2b --n_seq 384 --n_feat 240 --n_pos 6 --pos_mode shared \
    --out results/eval_arms_q35_s384_shared.csv
python src/variance_decomp.py --crossed results/eval_arms_q35_s384_shared.csv
```

`bash src/run_all.sh` retrains and re-evaluates the Gemma-2-2B and Gemma-3-1B arms and runs the
reproducibility check (`src/repro_check.py`) in one go.

**Released dictionaries** (Gemma Scope and the released Qwen3.5 dictionaries, downloaded from their hubs):

```bash
python src/released_agreement.py --suite 2b --n_seq 384   # Gemma Scope 2B pairs
bash src/pod_run.sh                                       # Gemma Scope 9B (and 27B) pairs, one 80 GB GPU
bash src/queue9b_v2.sh                                    # 9B specification check, 2B depth sweep, 9B matrix
python src/prefetch_released_qwen35.py
python src/released_qwen35.py --suite 2b
python src/released_qwen35.py --suite 9b --truncate --bs 4 --device_map cuda
```

The remaining scripts in `src/` each regenerate one result family named in their docstring (for example
`sae_rare.py`, `multi_dict.py`, `corpus_curve.py`, `displacement_stats.py`, `firing_density.py`,
`cosine_dist.py`, `position_structure.py`). `results/provenance.json` records the corpus file hash, the
base-model and Gemma Scope revisions, and the package versions of the large-GPU runs.

Seeds are fixed at 0 and all six arms of a model share one initialisation. Retraining in a different
software environment reproduces the variance components to within 1.8 percentage points, not bitwise.

## Not included, and why

| Item | Reason |
|---|---|
| Trained dictionaries (`data/`, about 12 GB) | Size. Regenerated by the `chain6*.sh` scripts from the fixed seeds. |
| Run logs | They record machine-specific paths and hold no result that is not in `results/`. |
| Base models, Gemma Scope and Qwen3.5 dictionaries, WikiText | Public, under their own licences; the scripts download them from their original hubs. |
| Outputs quarantined after a pipeline fix | Superseded by the corrected results in `results/`. |
| Manuscript source | Submitted separately. |
| The PDF of Cho et al. (2025) | Third-party paper. `src/extract_target_tables.py` reads its per-layer tables from a local copy saved as `st.pdf` (needs PyMuPDF); its output is kept in `results/target_tables.csv`. |
| Scripts from candidate analyses the paper does not report | Not used by any number or figure. |

## Licences

The code is released under the MIT licence (see `LICENSE`). The files in `results/` are measurements
computed on WikiText-103 and WikiText-2 (CC BY-SA) with public base models and dictionaries, each under
its own licence (Gemma 2 and Gemma 3 under the Gemma Terms of Use, Gemma Scope under CC BY 4.0; see the
model cards of Qwen3.5, OLMo-2, SmolLM3 and the released Qwen3.5 dictionaries). Apart from single decoded
tokens in `results/trace_*.csv`, `results/st_detect_*.csv` and
`results/sweep_orthography.csv`, they hold numbers and indices, not corpus
text. Nothing in this repository relicenses those models, dictionaries or corpora.
