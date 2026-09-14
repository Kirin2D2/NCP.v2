#!/usr/bin/env bash
# Reproduce the watermark experiments (paper Figures 3-4): CNP (--pruner ncp) vs. vanilla
# LRP pruning on carton/dugong and crate/packet, 5 seeds each, then plot the figures.
#
# Usage:
#   scripts/reproduce_watermark.sh                    # everything: {carton,crate} x {ncp,vanilla} x seeds 0-4
#   scripts/reproduce_watermark.sh carton             # one experiment
#   scripts/reproduce_watermark.sh carton ncp 0 1     # one pruner, selected seeds
#
# Environment:
#   OUT_DIR     results root (default: results/watermark_experiment)
#   EXTRA_ARGS  extra flags passed to every run, e.g. EXTRA_ARGS="--save_model"
#
# Subspaces are 0-indexed: carton ablates 3 (paper "subspace 4"), crate ablates 1 (paper "subspace 2").
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1   # logs update live when output is redirected to a file

EXPERIMENTS=${1:-all}
PRUNERS=${2:-all}
shift $(( $# < 2 ? $# : 2 ))
if [ $# -gt 0 ]; then SEEDS=("$@"); else SEEDS=(0 1 2 3 4); fi

[ "$EXPERIMENTS" = all ] && EXPERIMENTS="carton crate"
[ "$PRUNERS" = all ] && PRUNERS="ncp vanilla"

OUT_DIR=${OUT_DIR:-results/watermark_experiment}
EXTRA_ARGS=${EXTRA_ARGS:-}

spurious_subspace() {
  case "$1" in
    carton) echo 3 ;;
    crate)  echo 1 ;;
    *) echo "unknown experiment: $1" >&2; exit 1 ;;
  esac
}

for exp in $EXPERIMENTS; do
  for pruner in $PRUNERS; do
    for seed in "${SEEDS[@]}"; do
      if [ -f "$OUT_DIR/$exp/$pruner/seed$seed/stats.pt" ]; then
        echo "=== skip experiment=$exp pruner=$pruner seed=$seed (stats.pt exists) ==="
        continue
      fi
      echo "=== experiment=$exp pruner=$pruner seed=$seed ==="
      # shellcheck disable=SC2086
      python experiments/run_watermark_pruning_experiment.py \
        --experiment            "$exp" \
        --pruner                "$pruner" \
        --seed                  "$seed" \
        --spurious_subspace     "$(spurious_subspace "$exp")" \
        --pr_step               0.05 \
        --total_pr              0.80 \
        --lr                    1e-4 \
        --momentum              0.9 \
        --train_batch_size      32 \
        --test_batch_size       32 \
        --warmup_epochs         15 \
        --iter_finetune_epochs  2 \
        --final_finetune_epochs 5 \
        --rank_loader_type      positive_only \
        --eval_on_test \
        --out_dir               "$OUT_DIR" \
        $EXTRA_ARGS
    done
  done
done

# Figures (skips experiments/pruners with no results yet)
python analysis/plot_watermark_results.py --results_dir "$OUT_DIR" --no_titles
python analysis/plot_watermark_results.py --results_dir "$OUT_DIR" --no_titles --c0w1_only
