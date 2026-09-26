#!/usr/bin/env bash
# E25: the third point on the corpus-size curve. Forward passes only -- no
# retraining, the same six arms per model that produced the 96 and 384 runs.
#
# Ordering is deliberate: gemma3-1b first. Its residual cube is half the width
# of gemma2-2b's (d_model 1152 vs 2304), so if host RAM or GPU memory is short
# the smaller model still lands and the run degrades to one model rather than
# none. Predictions and the decision rule are in PREREG.md E25, written before
# this script was run.
#
# RESUMABLE. Each step is skipped when its CSV already exists, because the
# gemma3 chain was killed by a CUDA OOM at arm 5 of 6 once and restarting from
# zero cost three hours.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }

N=1536

run () {   # base, pos_mode, out
    if [ -s "$3" ]; then
        echo "### SKIP $3 (exists)  $(date +%H:%M:%S)"
        return
    fi
    echo "### $1 n_seq=$N $2  $(date +%H:%M:%S)"
    "$P" -u src/eval_arms.py --base "$1" --n_seq "$N" --n_feat 240 --n_pos 6 \
        --pos_mode "$2" --out "$3"
}

run gemma3-1b per_arm results/eval_arms_g3_s1536.csv
run gemma3-1b shared  results/eval_arms_g3_s1536_shared.csv
run gemma2-2b per_arm results/eval_arms_g2_s1536.csv
run gemma2-2b shared  results/eval_arms_g2_s1536_shared.csv

echo "### E25 EVAL DONE  $(date +%H:%M:%S)"
