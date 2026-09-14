#!/bin/bash
# Example SLURM array for the watermark experiments (paper Figures 3-4).
# One job per (experiment, pruner, seed): 2 × 2 × 5 = 20 tasks.
#
#   0– 4 : carton / ncp (CNP) / seeds 0–4
#   5– 9 : carton / vanilla   / seeds 0–4
#  10–14 : crate  / ncp (CNP) / seeds 0–4
#  15–19 : crate  / vanilla   / seeds 0–4
#
# Submit from the repository root:  mkdir -p logs && sbatch scripts/slurm/watermark_pruning.slurm
# Adjust the #SBATCH resources and the environment activation for your cluster.
# Plot afterwards with: python analysis/plot_watermark_results.py --no_titles [--c0w1_only]
#
#SBATCH --job-name=cnp_watermark
#SBATCH --output=logs/cnp_watermark_%A_%a.out
#SBATCH --error=logs/cnp_watermark_%A_%a.err
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

# Activate your environment, e.g.:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate cnp
${ACTIVATE_ENV:-true}

TASK=${SLURM_ARRAY_TASK_ID}
EXPERIMENTS=(carton crate)
PRUNERS=(ncp vanilla)

EXPERIMENT=${EXPERIMENTS[$(( TASK / 10 ))]}
PRUNER=${PRUNERS[$(( (TASK % 10) / 5 ))]}
SEED=$(( TASK % 5 ))

echo "Task ${TASK}: experiment=${EXPERIMENT} pruner=${PRUNER} seed=${SEED}"

# Run the per-seed training only; plotting is done once after all tasks finish.
EXTRA_ARGS=${EXTRA_ARGS:-} OUT_DIR=${OUT_DIR:-results/watermark_experiment} \
python experiments/run_watermark_pruning_experiment.py \
  --experiment "$EXPERIMENT" --pruner "$PRUNER" --seed "$SEED" \
  --spurious_subspace "$([ "$EXPERIMENT" = carton ] && echo 3 || echo 1)" \
  --pr_step 0.05 --total_pr 0.80 --lr 1e-4 --momentum 0.9 \
  --train_batch_size 32 --test_batch_size 32 \
  --warmup_epochs 15 --iter_finetune_epochs 2 --final_finetune_epochs 5 \
  --rank_loader_type positive_only --eval_on_test \
  --out_dir "${OUT_DIR:-results/watermark_experiment}" ${EXTRA_ARGS:-}
