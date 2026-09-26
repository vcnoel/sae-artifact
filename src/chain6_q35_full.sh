#!/usr/bin/env bash
# Six Qwen3.5-2B-Base arms at 48M tokens each (four times the 12M budget), all from seed 0, trained one after another
# on one 80 GB card (six in parallel recompute the same activations and run slower in total). Same fitting choices as chain6_q35.sh; only the token budget differs.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/q35_full
P="${PYTHON:-python}"
T=48000000   # four times the 12M budget: the most six arms reach on one card before the freeze
run(){ local tag=$1; shift
  if [ -f "data/q35_full/arm_${tag}.pt" ]; then echo "### $tag already present"; return 0; fi
  echo "### $tag start $(date +%H:%M:%S)"
  "$P" -u src/train_arms.py --base qwen35-2b-full --tag "$tag" "$@" --tokens $T > "logs/q35full_${tag}.log" 2>&1 \
    && echo "### $tag done $(date +%H:%M:%S)" || echo "### FAILED $tag $(date +%H:%M:%S)"
}
run free
run tau080 --tau 0.80
run tau090 --tau 0.90
run lr1e4  --lr 1e-4
run k41    --k 41
run order1 --order 1
n=$(ls data/q35_full/arm_*.pt 2>/dev/null | wc -l)
echo "### SIX ARMS DONE (qwen35-2b-full) $n arms $(date +%H:%M:%S)"
