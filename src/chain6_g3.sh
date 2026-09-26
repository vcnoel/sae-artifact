#!/usr/bin/env bash
# Six gemma3-1b arms, all from seed 0 -- the second-base-model replication.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
# Set PYTHON to your interpreter, or rely on the active environment.
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }
# Resumable: skip arms already on disk. A CUDA OOM on arm 5 of 6 previously
# killed the chain under `set -e`, and without this the retry retrains all six.
# Also wait for VRAM to be released between arms -- the OOM hit at arm start,
# immediately after the previous arm exited, so the allocator had not yet freed.
run(){
  local tag=$1
  if [ -f "data/g3/arm_${tag}.pt" ]; then
    echo "### $tag  already present, skipping"; return 0
  fi
  echo "### $tag  $(date +%H:%M:%S)"
  "$P" -u src/train_arms.py --base gemma3-1b --tag "$@"
  sleep 20
}
run free                       --tokens 12000000
run tau080  --tau 0.80         --tokens 12000000
run tau090  --tau 0.90         --tokens 12000000
run lr1e4   --lr 1e-4          --tokens 12000000
run k41     --k 41             --tokens 12000000
run order1  --order 1          --tokens 12000000
echo "### SIX ARMS DONE (gemma3-1b) $(date +%H:%M:%S)"
