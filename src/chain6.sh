#!/usr/bin/env bash
# Six gemma2-2b arms, all from seed 0. About 4 h on one 16 GB GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
# Set PYTHON to your interpreter, or rely on the active environment.
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }
run(){
  local tag=$1
  if [ -f "data/arm_${tag}.pt" ]; then
    echo "### $tag  already present, skipping"; return 0
  fi
  echo "### $tag  $(date +%H:%M:%S)"
  "$P" -u src/train_arms.py --tag "$@"
  sleep 20
}
run free                       --tokens 12000000
run tau080  --tau 0.80         --tokens 12000000
run tau090  --tau 0.90         --tokens 12000000
run lr1e4   --lr 1e-4          --tokens 12000000
run k41     --k 41             --tokens 12000000
run order1  --order 1          --tokens 12000000
echo "### SIX ARMS DONE $(date +%H:%M:%S)"
