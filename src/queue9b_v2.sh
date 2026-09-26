#!/usr/bin/env bash
# Everything queued behind the 2B legs, in the order the paper needs it.
#
# Written as a NEW FILE rather than an edit to queue9b.sh, because bash reads a
# script incrementally from a byte offset: editing a running script makes the
# interpreter resume mid-token and execute garbage. Kill the old, launch this.
#
# WHY SERIAL. Four jobs time-slicing one 80 GB GPU ran ~4x slower than the same work
# alone: the pre-fix 9B band sweep took ~1h by itself, and after the encode fix
# four concurrent jobs were on track for 7h (2B bands) and 14h (9B matrix),
# which the remaining machine time does not cover. Serialising restored ~4x.
#
# EVERY LEG IS NON-FATAL. `set -e` is deliberately absent and each leg logs its
# own failure and continues. A leg that dies, or one refused by the
# overwrite assertion on a retry, must not strand the legs behind it -- the
# whole point of the queue is that the tail of the night runs unattended.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
P="${PYTHON:-python3}"

leg () {   # label, then the command
  local label="$1"; shift
  echo "### START $label  $(date +%H:%M:%S)"
  if "$@"; then
    echo "### DONE  $label  $(date +%H:%M:%S)"
  else
    echo "### FAILED $label (exit $?) -- continuing to the next leg  $(date +%H:%M:%S)"
  fi
}

while pgrep -f "src/released_agreement.py --suite 2b" > /dev/null \
   || pgrep -f "src/scope_matrix.py --suite 2b" > /dev/null; do sleep 60; done
echo "### 2b legs done $(date +%H:%M:%S)"

# --- 9B L0 spec check, before anything consumes 9B numbers ------------------
# The 2B deviations run systematically high and scale with width: +1.4% at 16k,
# +2.3% at 32k, +3.9% at 65k, r = 0.76 against log width. A corpus difference or
# a BOS-handling difference should be roughly uniform across widths, so the
# shape is wrong for both benign readings. Six dictionaries at three widths is
# too few to call it. The 9B suite doubles the sample at the same three widths
# for ~15 min and validates those dictionaries before their numbers are used.
leg "9b L0 spec check" \
    "$P" -u src/verify_dicts.py --suite 9b --n_seq 32

leg "9b bands" \
    "$P" -u src/released_agreement.py --suite 9b --n_seq 384

# --- 2B depth ---------------------------------------------------------------
# ORDER IS 20, THEN 12, THEN 5, and both parts of that matter.
#
# Layer 20 first (77% depth). The generality objection is that mid-network is
# special, and the deep end is where features are least token-aligned, so layer
# 20 is the point that makes the axis mean anything. An earlier version of this
# script kept 5 and dropped 20, which would have left a two-point contrast at
# 19% and 46% -- both in the shallow half, and unable to answer the objection
# they exist to answer. If only one extra depth fits, it must be this one.
#
# Layer 12 IS recomputed, not copied. Copying pair 1 of the headline run is
# defensible as arithmetic, but src/depth_check.py exists to compare the
# mid-layer recomputation against the existing number and veto the trend if
# they disagree; fed a copy, it compares a number to itself, passes trivially,
# and reports nothing. A reader bug invalidated a full night of results tonight,
# so a genuine reproduction of the mid layer under the corrected reader is worth
# 65 minutes more than a third depth point is.
#
# Layer 5 last, if anything remains. Ordering costs nothing and protects the
# more valuable points.
#
# NO mv after each leg. released_agreement.py now derives the CSV path from
# --out, so this writes results/depth_2b_L$1.csv directly. The old fallback here
# moved results/released_agreement_2b.csv into place, which would have MOVED THE
# HEADLINE BAND CSV AWAY had it still been sitting at that path.
for L in "20 71 22" "12 82 22" "5 68 18"; do
  set -- $L
  leg "2b depth layer $1 (16k L0 $2 vs $3)" \
      "$P" -u src/released_agreement.py --suite 2b --layer "$1" --n_seq 384 \
          --pairs "width_16k/average_l0_$2|width_16k/average_l0_$3|same width, L0 $2 vs $3" \
          --out "results/depth_2b_L$1.txt"
done

# --- 9B matrix, last --------------------------------------------------------
# Shuffled pair order. combinations() walks the grid in width order, so an
# interrupted run would hold every 16k-vs-X pair and no wide-vs-wide one, and
# its agreement range would read as the full range while being a systematic
# subsample. With a seed, any prefix is unbiased. Reporting rule is fixed in
# PREREG E33 BEFORE the numbers exist: report only at >= 10 of 15 pairs, as a
# subsample with the pairs named and the count stated; below that, do not
# report it and rely on the 9B bands.
leg "9b matrix" \
    "$P" -u src/scope_matrix.py --suite 9b --n_seq 384 --shuffle_pairs 20260811

echo "### QUEUE COMPLETE $(date +%H:%M:%S)"
