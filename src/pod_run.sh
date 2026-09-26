#!/usr/bin/env bash
# E29: establish the premise on larger released dictionaries, on one 80 GB GPU.
#
# Set HF_HOME to a disk with room for the model downloads (about 24 GB for the 9B
# leg and about 85 GB for the 27B leg); every output is written under the repo.
#
# RESUMABLE AND IDEMPOTENT. Every step is skipped when its output exists, so a
# dead machine costs the current step and nothing before it. scope_matrix.py is
# additionally resumable WITHIN a step: it appends each pair to its CSV as that
# pair finishes and skips pairs already present.
#
# ISOLATION. The 27B leg runs last and its failure cannot take down the 9B
# results, which are the ones worth having. 27B is a weaker test regardless:
# Google published only width-131k dictionaries for it, at three layers, so a
# 27B pair can differ in sparsity but never in width.
set -uo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME" results logs
P="${PYTHON:-python3}"
"$P" -c "import torch;assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }
echo "### HF_HOME=$HF_HOME  free: $(df -h "$HF_HOME" | tail -1 | awk '{print $4}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

FAILED=""

step () {   # label, output-file, command...
  local label="$1" out="$2"; shift 2
  if [ -s "$out" ]; then echo "### SKIP $label ($out exists)  $(date +%H:%M:%S)"; return 0; fi
  echo "### START $label  $(date +%H:%M:%S)"
  "$@"
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -s "$out" ]; then
    echo "### FAILED $label (exit $rc)  $(date +%H:%M:%S)"
    FAILED="$FAILED $label"
    return 1
  fi
  echo "### DONE  $label  $(date +%H:%M:%S)"
}

# ---------------- 9B: complete, before 27B is attempted -------------------
step "9b bands"  results/released_agreement_9b.csv \
    "$P" -u src/released_agreement.py --suite 9b --n_seq 384
step "9b matrix" results/scope_matrix_9b.csv \
    "$P" -u src/scope_matrix.py --suite 9b --n_seq 384

if [ -n "$FAILED" ]; then
  echo "### 9B INCOMPLETE ($FAILED) -- not starting 27B  $(date +%H:%M:%S)"
  exit 1
fi
echo "### 9B COMPLETE  $(date +%H:%M:%S)"

# ---------------- 27B: conditional, isolated ------------------------------
# device_map=auto because 27B in bf16 is ~54GB and leaves little room beside
# the activation cache on an 80GB card.
step "27b bands"  results/released_agreement_27b.csv \
    "$P" -u src/released_agreement.py --suite 27b --n_seq 384 --device_map auto \
    || echo "### 27B BANDS UNAVAILABLE -- 9B results stand"
step "27b matrix" results/scope_matrix_27b.csv \
    "$P" -u src/scope_matrix.py --suite 27b --n_seq 384 --device_map auto \
    || echo "### 27B MATRIX UNAVAILABLE -- 9B results stand"

echo "### CHAIN COMPLETE  $(date +%H:%M:%S)  failed:${FAILED:- none}"
