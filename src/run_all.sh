#!/usr/bin/env bash
# Everything that needs the GPU, in dependency order. One GPU, so these run
# sequentially rather than as two concurrent jobs: concurrent training would
# contend for VRAM and neither result is needed before the other finishes.
#
#   1. retrain the six gemma2-2b arms from seed 0 (the originals were deleted)
#   2. archive the checkpoints OUTSIDE the repo
#   3. re-evaluate them, per_arm AND shared positions
#   4. reproducibility check of the rerun against the archived eval_arms.csv
#   5. the same for gemma3-1b
#
# Total ~9h. Resumable: each stage skips itself if its outputs already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
# Set PYTHON to your interpreter, or rely on the active environment.
P="${PYTHON:-python}"
[ -x "$P" ] || { echo "no interpreter at $P"; exit 1; }
ARCHIVE="${ARCHIVE:-../sae-arm-archive}"
mkdir -p "$ARCHIVE"

stamp(){ echo "=== $* $(date +%F_%H:%M:%S)"; }

# ---------- 1. gemma2-2b arms ----------
if [ "$(ls -1 data/arm_*.pt 2>/dev/null | wc -l)" -ge 6 ]; then
    stamp "gemma2 arms already present, skipping training"
else
    stamp "gemma2 six-arm retrain"
    bash src/chain6.sh
fi

# ---------- 2. archive ----------
stamp "archiving gemma2 checkpoints to $ARCHIVE/gemma2-2b"
mkdir -p "$ARCHIVE/gemma2-2b"
cp -n data/arm_*.pt "$ARCHIVE/gemma2-2b/" || true
"$P" -c "import transformers, torch, json, sys; print(json.dumps({'transformers': transformers.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0)}, indent=2))" \
    > "$ARCHIVE/gemma2-2b/ENVIRONMENT.json"

# ---------- 3. re-evaluate ----------
# Writes the *_fixsamp names. The bare eval_arms_rerun.csv / eval_arms_shared.csv
# are the PRE-sampler-fix artifacts and are quarantined (R6): this chain must
# not regenerate fixed-sampler data under a name the paper's history associates
# with the broken sample.
stamp "gemma2 eval, per_arm positions"
"$P" -u src/eval_arms.py --n_seq 96 --n_feat 240 --n_pos 6 \
    --pos_mode per_arm --out results/eval_arms_rerun_fixsamp.csv
stamp "gemma2 eval, shared positions"
"$P" -u src/eval_arms.py --n_seq 96 --n_feat 240 --n_pos 6 \
    --pos_mode shared --out results/eval_arms_shared_fixsamp.csv

# ---------- 4. decompositions + reproducibility ----------
stamp "decomposition, per_arm rerun"
"$P" -u src/variance_decomp.py --crossed results/eval_arms_rerun_fixsamp.csv
stamp "decomposition, SHARED positions (the position-crossed design)"
"$P" -u src/variance_decomp.py --crossed results/eval_arms_shared_fixsamp.csv
stamp "position structure, shared"
"$P" -u src/position_structure.py \
    --crossed results/eval_arms_shared_fixsamp.csv \
    --out results/position_structure_shared.txt
stamp "sampler consistency across every compared pair of files"
"$P" -u src/sampler_check.py
stamp "reproducibility check vs the original run"
"$P" -u src/repro_check.py
# THE decision point: how much of the transfer failure was position selection.
# Branches are pre-registered in paper/S5_BRANCHES.md.
stamp "transfer: per-arm vs shared positions"
"$P" -u src/transfer_compare.py

# ---------- 5. gemma3-1b ----------
if [ "$(ls -1 data/g3/arm_*.pt 2>/dev/null | wc -l)" -ge 6 ]; then
    stamp "gemma3 arms already present, skipping training"
else
    stamp "gemma3-1b six-arm training"
    bash src/chain6_g3.sh
fi
stamp "archiving gemma3 checkpoints"
mkdir -p "$ARCHIVE/gemma3-1b"
cp -n data/g3/arm_*.pt "$ARCHIVE/gemma3-1b/" || true
bash src/eval_decomp_g3.sh

stamp "ALL DONE"
