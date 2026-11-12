#!/bin/bash
#SBATCH --job-name=vulcan311
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --gres=gpu:1
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Vulcan 3.1.1 training on Cheaha (standalone auto-mined artifacts; halo ablated).
# Reads ITS OWN frozen notebook, vulcan_training_3_1_1.ipynb, so editing
# vulcan_training_3_0.ipynb (e.g. to resume 3.1) can never change what this chain trains.
# Submit from Python/Model_Training:
#
#   sbatch --export=ALL,VULCAN_CHECK_ONLY=1 vulcan_3_1_1.sbatch   # verify chain + resume state only
#   sbatch vulcan_3_1_1.sbatch                                    # train / resume
#
# Each job resumes from the last finished epoch, trains to ~45 min before walltime, and
# queues its continuation (afterany) until early stop / epoch 200 / VULCAN_MAX_RESUBMITS.
# Never overlap an interactive training session on the same run.
set -uo pipefail

MAX_RESUBMITS="${VULCAN_MAX_RESUBMITS:-8}"
COUNT="${VULCAN_RESUBMIT_COUNT:-0}"
SBATCH_FILE="${SLURM_SUBMIT_DIR}/vulcan_3_1_1.sbatch"     # $0 is a spool copy inside a job
export VULCAN_NOTEBOOK="${SLURM_SUBMIT_DIR}/vulcan_training_3_1_1.ipynb"

cd "${SLURM_SUBMIT_DIR}"
[[ -f "${VULCAN_NOTEBOOK}" ]] || { echo "missing ${VULCAN_NOTEBOOK}"; exit 1; }
module reset
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Anaconda3
conda activate ml_tf_2.15

echo "job ${SLURM_JOB_ID} on $(hostname) | resubmit ${COUNT}/${MAX_RESUBMITS} | $(date)"
echo "notebook: ${VULCAN_NOTEBOOK}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA=()
[[ "${VULCAN_CHECK_ONLY:-0}" == "1" ]] && EXTRA+=(--check-only)
# ${EXTRA[@]+...}: empty-array-safe under set -u on bash < 4.4 (Cheaha)
python -u run_vulcan_3_0.py --job-id "${SLURM_JOB_ID}" ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?

if [[ $rc -eq 75 && "${VULCAN_CHECK_ONLY:-0}" != "1" ]]; then
  if (( COUNT + 1 > MAX_RESUBMITS )); then
    echo "training incomplete and resubmit limit ${MAX_RESUBMITS} reached; resubmit manually"
    exit 1
  fi
  sbatch --export=ALL,VULCAN_RESUBMIT_COUNT=$((COUNT + 1)) \
         --dependency=afterany:${SLURM_JOB_ID} "${SBATCH_FILE}"
  echo "training incomplete: queued continuation $((COUNT + 1))"
  exit 0
fi
echo "finished with exit code ${rc} | $(date)"
exit $rc
