#!/usr/bin/env bash
# Layer-20 depth point at a MATCHED L0 ratio.
#
# WHY. The existing depth_2b_L20 uses 16k L0 71 vs 22, ratio 3.23, against 3.78
# at layer 5 and 3.73 at layer 12. The axis varied within each pair differs by
# 15% at exactly the layer producing the surprising numbers (low band 6.0, top
# band 67.7), so the divergence-with-depth reading could not be written. Layer 20
# publishes L0 22/38/71/139/294 at width 16k, and 139 vs 38 gives ratio 3.66 --
# a closer match than layer 12's own 3.73.
#
# THE TRADE, stated because it is a swap of confounds and not their removal.
# Matching the ratio unmatches the LEVEL: 139 vs 38 is far denser than layer
# 12's 82 vs 22. Firing density is the denominator of position agreement, so a
# denser pair has more candidate positions and mechanically lower agreement.
# That is why this script also measures firing density for the exact two
# dictionaries: if the new layer-20 pair fires at roughly double layer 12's
# rate, that belongs in the same sentence as the agreement number.
#
# The honest framing for the write-up: depth is compared at matched ratio and
# unmatched level; the level difference runs in the direction that would DEPRESS
# agreement at layer 20; the low-band trend survives that because it falls
# monotonically anyway. If it does not survive, report the trend as confounded.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
P="${PYTHON:-python3}"
PAIR="width_16k/average_l0_139|width_16k/average_l0_38|same width, L0 139 vs 38"

echo "### START L0 spec check  $(date +%H:%M:%S)"
"$P" -u src/verify_dicts.py --suite 2b --layer 20 --n_seq 32 --paths "width_16k/average_l0_139,width_16k/average_l0_38" \
    || echo "### L0 SPEC MISMATCH -- continuing, but read this before the numbers"

echo "### START firing density (139 and 38)  $(date +%H:%M:%S)"
"$P" -u src/firing_density.py --suite 2b --layer 20 --n_seq 384 \
    --paths "width_16k/average_l0_139,width_16k/average_l0_38" \
    --out results/firing_density_2b_L20_r366.csv \
    || echo "### firing density FAILED -- continuing"

echo "### START bands  $(date +%H:%M:%S)"
"$P" -u src/released_agreement.py --suite 2b --layer 20 --n_seq 384 \
    --pairs "$PAIR" --out results/depth_2b_L20_r366.txt \
    || echo "### bands FAILED"

echo "### COMPLETE $(date +%H:%M:%S)"
