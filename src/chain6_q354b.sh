#!/usr/bin/env bash
# Six Qwen3.5-4B-Base arms, all from seed 0 -- Qwen3.5, 4B, the Qwen3.5 scale ladder.
#
# The Gemma-2 and Gemma-3 runs establish the standard inside one family. This run is what
# lets the paper say the standard is not a Gemma result. Same width, same k, same token
# budget, same six fitting choices: only the base model differs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/q35_4b
# Set PYTHON to your interpreter, or rely on the active environment.
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }
# Resumable: skip arms already on disk. A CUDA OOM on arm 5 of 6 previously
# killed the chain under `set -e`, and without this the retry retrains all six.
# Also wait for VRAM to be released between arms -- the OOM hit at arm start,
# immediately after the previous arm exited, so the allocator had not yet freed.
run(){
  local tag=$1
  if [ -f "data/q35_4b/arm_${tag}.pt" ]; then
    echo "### $tag  already present, skipping"; return 0
  fi
  echo "### $tag  $(date +%H:%M:%S)"
  "$P" -u src/train_arms.py --base qwen35-4b --tag "$@"
  sleep 20
}
run free                       --tokens 12000000
run tau080  --tau 0.80         --tokens 12000000
run tau090  --tau 0.90         --tokens 12000000
run lr1e4   --lr 1e-4          --tokens 12000000
run k41     --k 41             --tokens 12000000
run order1  --order 1          --tokens 12000000
echo "### SIX ARMS DONE (qwen35-4b) $(date +%H:%M:%S)"
