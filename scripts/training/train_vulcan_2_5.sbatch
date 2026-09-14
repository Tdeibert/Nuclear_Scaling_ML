#!/bin/bash
# Vulcan 2.5 phase-1 training, submitted as a batch job.
#
#   sbatch scripts/training/train_vulcan_2_5.sbatch
#
# Partition choice (measured 2026-09-14, ~5.1 min/epoch, 200 epochs = ~17 h):
#
#   amperenodes         12 h cap, queue est. ~2 h  -> 11 h = ~129 epochs
#   amperenodes-medium  48 h cap, queue est. ~8.5 h
#
# amperenodes is the default because it starts ~7 h sooner and ModelCheckpoint
# writes the best weights every time val_nucleus_quality improves -- so even if
# the 11 h walltime kills the job mid-run, the best model is already on disk.
# What a kill costs is only the `_final` save and EarlyStopping's
# restore_best_weights; there is no resume (model.fit has no initial_epoch).
# If training turns out to need the full 200 epochs, switch to:
#   #SBATCH --partition=amperenodes-medium
#   #SBATCH --time=24:00:00
#
# --cpus-per-task MUST stay 8: cfg.max_parallel_workers is read from
# SLURM_CPUS_PER_TASK and is a hashed config field, so a different value sends
# the run to a different run_dir.

#SBATCH --job-name=vulcan25_p1
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=11:00:00
#SBATCH --output=/home/tdeibert/Data/Machine_Learning_Dev/Outputs/slurm/vulcan25_p1_%j.out
#SBATCH --error=/home/tdeibert/Data/Machine_Learning_Dev/Outputs/slurm/vulcan25_p1_%j.err

set -euo pipefail

REPO=/home/tdeibert/Projects/Nuclear_Scaling
PY=/home/tdeibert/.conda/envs/ml_tf_2.15/bin/python   # absolute path: no conda
                                                      # activate needed in a
                                                      # non-interactive shell

echo "job    : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "start  : $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "$REPO"
"$PY" -u scripts/training/run_notebook_training.py

echo "end    : $(date)"
