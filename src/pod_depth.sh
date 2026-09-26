#!/usr/bin/env bash
# E30: does the premise hold across depth? Predictions in PREREG.md E30,
# written before this ran.
#
# Same-width 16k pairs at every depth, because only layer 12 (2B) and layer 20
# (9B) publish the 16k/32k/65k grid. Comparing a 16k-vs-65k pair at one depth
# against a 16k-vs-131k pair at another would confound depth with pair type.
# The mid layer is recomputed on the same restricted pair so all three depths
# are like-for-like; the paper's headline numbers stay at the existing pairs.
#
# Costs no new gated download: both base models are already in the cache, and
# Gemma Scope dictionaries are ungated. That is why this runs and 27B does not.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
P="${PYTHON:-python3}"
mkdir -p results logs

run () {   # suite, layer, hi_l0, lo_l0
  local out="results/depth_${1}_L${2}.csv"
  if [ -s "$out" ]; then echo "### SKIP $out"; return 0; fi
  echo "### START ${1} layer ${2}  (16k L0 ${3} vs ${4})  $(date +%H:%M:%S)"
  "$P" -u src/released_agreement.py --suite "$1" --layer "$2" --n_seq 384 \
      --pairs "width_16k/average_l0_${3}|width_16k/average_l0_${4}|same width, L0 ${3} vs ${4}" \
      --out "results/depth_${1}_L${2}.txt"
  local rc=$?
  if [ -s "results/released_agreement_${1}.csv" ]; then
    mv "results/released_agreement_${1}.csv" "$out"
  fi
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    echo "### FAILED ${1} L${2} (exit $rc)  $(date +%H:%M:%S)"; return 1
  fi
  echo "### DONE  ${1} L${2}  $(date +%H:%M:%S)"
}

# 2B: 26 layers -> 19% / 46% / 77% depth
run 2b 5  68 18
run 2b 12 82 22
run 2b 20 71 22
# 9B: 42 layers -> 21% / 48% / 74% depth
run 9b 9  51 16
run 9b 20 68 20
run 9b 31 63 20

echo "### DEPTH SWEEP COMPLETE  $(date +%H:%M:%S)"
