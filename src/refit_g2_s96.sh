#!/usr/bin/env bash
# results/eval_arms_rerun.csv (gemma2, n_seq 96) predates the sampler fix: it
# drew from the live-set intersection, so its 240 latents share only 7 with the
# 384 and 1536 runs, which share 232 with each other. Every Gemma-2 statement
# comparing 96 against a larger corpus is therefore a between-sample comparison
# wearing a corpus label. Re-measured here at the same corpus with the current
# sampler, to NEW files -- the old ones are left on disk so the difference
# stays inspectable.
set -uo pipefail
cd "$(dirname "$0")/.."
P="${PYTHON:-python}"
"$P" -c "import torch;assert torch.cuda.is_available()" || { echo "no CUDA"; exit 1; }
for MODE in per_arm shared; do
  OUT="results/eval_arms_g2_s96_fixsamp.csv"
  [ "$MODE" = shared ] && OUT="results/eval_arms_g2_s96_fixsamp_shared.csv"
  if [ -s "$OUT" ]; then echo "### SKIP $OUT"; continue; fi
  echo "### g2 n_seq=96 $MODE  $(date +%H:%M:%S)"
  "$P" -u src/eval_arms.py --base gemma2-2b --n_seq 96 --n_feat 240 --n_pos 6 \
      --pos_mode "$MODE" --out "$OUT"
done
echo "### REFIT DONE  $(date +%H:%M:%S)"
