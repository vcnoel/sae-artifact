#!/usr/bin/env bash
# E25 + E26 + E27, sequentially, one log. Nothing here runs concurrently:
# the gemma2 shared evaluation at n_seq 1536 already swap-thrashed once when it
# had the machine to itself, and two of these steps hold a base model plus a
# multi-gigabyte residual at the same time.
#
# RESUMABLE throughout -- every step is skipped when its output exists, so a
# kill costs the current step and nothing before it.
#
# The 9B leg is allowed to fail without taking the chain down: gemma-2-9b in
# bfloat16 is ~18GB against a 16GB card, so it may not fit. A failure there is
# reported, not hidden, and the 2B result answers the premise objection on its
# own.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }
echo "interpreter: $P"
"$P" -c "import torch;assert torch.cuda.is_available(),'no CUDA in this interpreter'" \
    || { echo "FATAL: interpreter has no CUDA; set PYTHON to a CUDA-enabled interpreter"; exit 1; }

step () {   # label, output-to-check, command...
    local label="$1"; local out="$2"; shift 2
    if [ -s "$out" ]; then
        echo "### SKIP $label ($out exists)  $(date +%H:%M:%S)"
        return 0
    fi
    echo "### START $label  $(date +%H:%M:%S)"
    if "$@"; then
        echo "### DONE  $label  $(date +%H:%M:%S)"
    else
        echo "### FAILED $label (exit $?)  $(date +%H:%M:%S)"
        return 1
    fi
}

# ---- E25: the third point on the corpus curve --------------------------------
step "e25 g3 per_arm" results/eval_arms_g3_s1536.csv \
    "$P" -u src/eval_arms.py --base gemma3-1b --n_seq 1536 --n_feat 240 \
        --n_pos 6 --pos_mode per_arm --out results/eval_arms_g3_s1536.csv
step "e25 g3 shared" results/eval_arms_g3_s1536_shared.csv \
    "$P" -u src/eval_arms.py --base gemma3-1b --n_seq 1536 --n_feat 240 \
        --n_pos 6 --pos_mode shared --out results/eval_arms_g3_s1536_shared.csv
step "e25 g2 per_arm" results/eval_arms_g2_s1536.csv \
    "$P" -u src/eval_arms.py --base gemma2-2b --n_seq 1536 --n_feat 240 \
        --n_pos 6 --pos_mode per_arm --out results/eval_arms_g2_s1536.csv
step "e25 g2 shared" results/eval_arms_g2_s1536_shared.csv \
    "$P" -u src/eval_arms.py --base gemma2-2b --n_seq 1536 --n_feat 240 \
        --n_pos 6 --pos_mode shared --out results/eval_arms_g2_s1536_shared.csv

# Analysis is CPU-only and cheap; always recomputed so it can never be stale
# with respect to the CSVs above.
echo "### START e25 analysis  $(date +%H:%M:%S)"
"$P" -u src/corpus_curve.py --out results/corpus_curve.txt \
    && echo "### DONE  e25 analysis  $(date +%H:%M:%S)" \
    || echo "### FAILED e25 analysis  $(date +%H:%M:%S)"

# ---- E26: released production dictionaries -----------------------------------
step "e26 gemma scope 2b" results/released_agreement_2b.csv \
    "$P" -u src/released_agreement.py --suite 2b --n_seq 384

step "e26 gemma scope 9b" results/released_agreement_9b.csv \
    "$P" -u src/released_agreement.py --suite 9b --n_seq 384 \
    || echo "### 9B LEG UNAVAILABLE -- reported as blocked, not dropped"

# ---- E27: peakedness against a released dictionary ---------------------------
step "e27 peakedness" results/peakedness.csv \
    "$P" -u src/peakedness.py --base gemma2-2b --n_seq 384

echo "### CHAIN COMPLETE  $(date +%H:%M:%S)"
