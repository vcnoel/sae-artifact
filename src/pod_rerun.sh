#!/usr/bin/env bash
# Re-run every released-dictionary measurement after the encode fix.
#
# THE FIX. JumpReLU.encode subtracted the decoder bias before the encoder.
# Gemma Scope's published encode does not. Measured on layer 12 width 16k
# L0 82: ours gave mean L0 301.8 and explained variance 0.158, the released
# convention gives 83.8 and 0.766, and the released spec for that dictionary is
# L0 = 82. Everything computed with the old reader is in results_WRONG/.
#
# NOT AFFECTED, and deliberately not re-run: cosine_dist_*.csv (decoder
# geometry only, never calls encode) and every result from our own six arms
# (eval_arms.py uses its own TopK gate, a separate code path).
#
# Ordered by what the paper needs first, staggered so several 9B model loads do
# not spike VRAM together. 252 cores, so these run concurrently by design.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
P="${PYTHON:-python3}"
mkdir -p results logs

launch () {   # label, logfile, command...
  local label="$1" log="$2"; shift 2
  echo "### LAUNCH $label  $(date +%H:%M:%S)"
  setsid nohup "$@" > "logs/$log" 2>&1 < /dev/null &
}

# --- 2B first: the paper's headline released evidence, and the cheapest -----
launch "2b bands"   rr_2b_bands.log \
    "$P" -u src/released_agreement.py --suite 2b --n_seq 384
sleep 20
launch "2b matrix"  rr_2b_matrix.log \
    "$P" -u src/scope_matrix.py --suite 2b --n_seq 384
sleep 20
launch "2b firing"  rr_2b_firing.log \
    "$P" -u src/firing_density.py --suite 2b --n_seq 384

# --- 9B next, staggered so the 18.5GB loads do not collide -----------------
sleep 60
launch "9b bands"   rr_9b_bands.log \
    "$P" -u src/released_agreement.py --suite 9b --n_seq 384
sleep 90
launch "9b matrix"  rr_9b_matrix.log \
    "$P" -u src/scope_matrix.py --suite 9b --n_seq 384

echo "### ALL LAUNCHED  $(date +%H:%M:%S)"
echo "### depth sweep NOT relaunched: it is robustness, and the headline"
echo "### released numbers must be correct first. Start it once these land."
