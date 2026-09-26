#!/usr/bin/env bash
# Evaluate + decompose the gemma3-1b arms. Assumes chain6_g3.sh already
# finished; run_all.sh sequences the two so no log-polling is needed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
# Set PYTHON to your interpreter, or rely on the active environment.
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }

echo "### g3 eval, per_arm positions  $(date +%H:%M:%S)"
"$P" -u src/eval_arms.py --base gemma3-1b --n_seq 96 --n_feat 240 --n_pos 6 \
    --pos_mode per_arm --out results/eval_arms_g3.csv

echo "### g3 eval, shared positions  $(date +%H:%M:%S)"
"$P" -u src/eval_arms.py --base gemma3-1b --n_seq 96 --n_feat 240 --n_pos 6 \
    --pos_mode shared --out results/eval_arms_g3_shared.csv

echo "### g3 decomposition (per_arm)  $(date +%H:%M:%S)"
"$P" -u src/variance_decomp.py --crossed results/eval_arms_g3.csv
echo "### g3 decomposition (shared)  $(date +%H:%M:%S)"
"$P" -u src/variance_decomp.py --crossed results/eval_arms_g3_shared.csv
echo "### g3 position structure  $(date +%H:%M:%S)"
"$P" -u src/position_structure.py --crossed results/eval_arms_g3.csv \
    --out results/position_structure_g3.txt
echo "### EVAL+DECOMP DONE (gemma3-1b) $(date +%H:%M:%S)"
