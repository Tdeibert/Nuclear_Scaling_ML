#!/bin/bash
#SBATCH --job-name=vulcan30
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
# Vulcan 3.0 training on Cheaha, unattended. Submit from Python/Model_Training:
#
#   sbatch vulcan_3_0.sbatch                     # train / resume
#   sbatch --export=ALL,VULCAN_CHECK_ONLY=1 vulcan_3_0.sbatch   # verify chain + resume state only
#
# Each job resumes the run from its last finished epoch (model + optimizer + early
# stopping), trains until ~45 min before walltime, stops at an epoch boundary, and
# queues the next job (afterany) until training completes (early stop or epoch 200)
# or VULCAN_MAX_RESUBMITS is reached. ~12 h = ~23 epochs at ~29 min/epoch.
#
# Never let this overlap an interactive training session on the same run: two writers
# would resume the same checkpoint. Chain it behind the interactive job instead:
#   sbatch --dependency=afterany:<interactive_job_id> vulcan_3_0.sbatch
set -uo pipefail

MAX_RESUBMITS="${VULCAN_MAX_RESUBMITS:-8}"
COUNT="${VULCAN_RESUBMIT_COUNT:-0}"
SBATCH_FILE="${SLURM_SUBMIT_DIR}/vulcan_3_0.sbatch"     # $0 is a spool copy inside a job

cd "${SLURM_SUBMIT_DIR}"
module reset
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Anaconda3
conda activate ml_tf_2.15

echo "job ${SLURM_JOB_ID} on $(hostname) | resubmit ${COUNT}/${MAX_RESUBMITS} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EXTRA=()
[[ "${VULCAN_CHECK_ONLY:-0}" == "1" ]] && EXTRA+=(--check-only)
python -u run_vulcan_3_0.py --job-id "${SLURM_JOB_ID}" ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?

if [[ $rc -eq 75 ]]; then
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
