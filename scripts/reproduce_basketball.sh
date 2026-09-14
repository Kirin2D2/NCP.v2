#!/usr/bin/env bash
# Basketball demonstration (paper Section 4.2.1): prune VGG16 to 80% with CNP (ablating the
# ball subspace) and with vanilla LRP pruning, then draw the LRP comparison figure.
#
# Usage:  scripts/reproduce_basketball.sh
# Environment:
#   OUT_DIR     results root (default: results/basketball)
#   EXTRA_ARGS  extra flags passed to both runs
#
# The paper ablates "subspace 4" (1-indexed), i.e. --irrelevant_subspaces 3.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=${OUT_DIR:-results/basketball}
EXTRA_ARGS=${EXTRA_ARGS:-}

COMMON=(
  --pr_step 0.05
  --total_pr 0.80
  --iter_finetune_epochs 2
  --final_finetune_epochs 5
  --rank_n_images 500
)

# shellcheck disable=SC2086
python experiments/run_PFT_basketball.py --pruner vanilla "${COMMON[@]}" \
  --out_dir "$OUT_DIR/vanilla" $EXTRA_ARGS

# shellcheck disable=SC2086
python experiments/run_PFT_basketball.py --pruner ncp --irrelevant_subspaces 3 "${COMMON[@]}" \
  --out_dir "$OUT_DIR/ncp_ss3" $EXTRA_ARGS

python analysis/basketball_lrp_comparison.py \
  --cnp_checkpoint     "$OUT_DIR/ncp_ss3/ncp-2.pth" \
  --vanilla_checkpoint "$OUT_DIR/vanilla/van.pth" \
  --out                "$OUT_DIR/lrp_comparison.png"
