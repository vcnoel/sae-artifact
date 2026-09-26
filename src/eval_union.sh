#!/usr/bin/env bash
# E28: does the interaction collapse survive a shared-position rule that does
# NOT select away from contested tokens? Predictions and decision rule are in
# PREREG.md E28, written before this ran. Resumable.
#
# The first version of this script swallowed a failure: run() ignored the exit
# status, so when the gemma2 leg died mid-measurement the chain printed nothing
# and moved on to gemma3, leaving one CSV on disk and a log that read like a
# completed run. A step that fails must say so and stop the model it belongs
# to. Same lesson as "job status comes from the job", one level down: a step's
# status comes from the step, not from the loop that ran it.
set -uo pipefail
cd "$(dirname "$0")/.."
P="${PYTHON:-python}"
"$P" -c "import torch;assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }

FAILED=0
run () {   # base, out
  if [ -s "$2" ]; then echo "### SKIP $2 (exists)"; return 0; fi
  echo "### START $1 union n_seq=384  $(date +%H:%M:%S)"
  "$P" -u src/eval_arms.py --base "$1" --n_seq 384 --n_feat 240 --n_pos 6 \
      --pos_mode union --out "$2"
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$2" ]; then
    echo "### FAILED $1 (exit $rc, output present: $([ -s "$2" ] && echo yes || echo no))  $(date +%H:%M:%S)"
    FAILED=1
    return 1
  fi
  echo "### DONE  $1  $(date +%H:%M:%S)"
}

run gemma2-2b results/eval_arms_g2_s384_union.csv
run gemma3-1b results/eval_arms_g3_s384_union.csv

if [ $FAILED -ne 0 ]; then
  echo "### E28 INCOMPLETE -- at least one leg failed  $(date +%H:%M:%S)"
  exit 1
fi
echo "### E28 DONE  $(date +%H:%M:%S)"
