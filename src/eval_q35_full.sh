#!/usr/bin/env bash
# Evaluate and decompose the six Qwen3.5-2B-Base arms: the second architecture family.
#
# Same settings as the Gemma runs, so the three base models differ only in the base: n_feat 240, n_pos 6,
# the corpus ladder 96 / 384 / 1536 in per_arm and shared modes (eval_decomp_g3.sh, eval_s1536.sh), and
# the union rule at 384 (eval_union.sh). --min_arms stays at its default, all arms, as in those runs.
#
# Waits for chain6_q35.sh to print its closing line, then refuses to start unless all six arm files exist,
# because eval_arms.py evaluates whatever arm_*.pt it finds and would score a partial set as if it were whole.
# RESUMABLE: each step is skipped when its CSV already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }

until grep -q "SIX ARMS DONE (qwen35-2b-full)" logs/q35full_chain.log 2>/dev/null; do sleep 60; done
n=$(ls data/q35_full/arm_*.pt 2>/dev/null | wc -l)
[ "$n" -eq 6 ] || { echo "### ABORT: expected 6 arms in data/q35_full, found $n  $(date +%H:%M:%S)"; exit 1; }
echo "### six arms present, evaluation starts  $(date +%H:%M:%S)"

run () {   # n_seq, pos_mode, out
    if [ -s "$3" ]; then echo "### SKIP $3 (exists)  $(date +%H:%M:%S)"; return; fi
    echo "### q35full n_seq=$1 $2  $(date +%H:%M:%S)"
    "$P" -u src/eval_arms.py --base qwen35-2b-full --n_seq "$1" --n_feat 240 --n_pos 6 \
        --pos_mode "$2" --out "$3"
}

# Cheapest first, so the smaller corpus sizes land even if the largest one runs out of time.
run 96   per_arm results/eval_arms_q35full.csv
run 96   shared  results/eval_arms_q35full_shared.csv
run 384  per_arm results/eval_arms_q35full_s384.csv
run 384  shared  results/eval_arms_q35full_s384_shared.csv
run 384  union   results/eval_arms_q35full_s384_union.csv
run 1536 per_arm results/eval_arms_q35full_s1536.csv
run 1536 shared  results/eval_arms_q35full_s1536_shared.csv

for f in results/eval_arms_q35full.csv results/eval_arms_q35full_shared.csv \
         results/eval_arms_q35full_s384.csv results/eval_arms_q35full_s384_shared.csv \
         results/eval_arms_q35full_s1536.csv results/eval_arms_q35full_s1536_shared.csv; do
    echo "### decomposition $f  $(date +%H:%M:%S)"
    "$P" -u src/variance_decomp.py --crossed "$f" || echo "### decomposition skipped for $f (make_macros decomposes itself)"
done
echo "### q35full position structure  $(date +%H:%M:%S)"
"$P" -u src/position_structure.py --crossed results/eval_arms_q35full.csv \
    --out results/position_structure_q35full.txt
echo "### EVAL+DECOMP DONE (qwen35-2b-full) $(date +%H:%M:%S)"
