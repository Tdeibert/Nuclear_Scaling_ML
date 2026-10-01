#!/bin/bash
#SBATCH --job-name=vulcan253
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=amperenodes-medium
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Vulcan 2.5.3 batch job for Cheaha. Submit from Python/Model_Training (next to the notebook).
#
#   sbatch vulcan_2_5_3.sbatch check                     # config, pools, GPU; writes nothing
#   sbatch vulcan_2_5_3.sbatch classical,train,segtest   # build classical pool, train, test
#
# Partition / time (docs.rc.uab.edu/cheaha/slurm/gpu): amperenodes allows up to 12 h,
# amperenodes-medium up to 48 h. 24 h covers the classical build plus a full 200-epoch
# run with headroom; for a shorter job use the faster-queuing partition, e.g.
#   sbatch --partition=amperenodes --time=12:00:00 vulcan_2_5_3.sbatch train,segtest
# Training is resumable: if the wall time runs short it stops MARGIN_MIN (30) minutes
# early, saves model + optimizer + early-stopping state, and this script queues the
# next job. The classical build is NOT resumable: if a job dies inside it, move the
# incomplete training_patches_* folder aside before resubmitting.
#
# Data: train/segtest copy the patch pools to node-local NVMe (/local/$USER/$SLURM_JOB_ID),
# as RC recommends for A100s, and the copy is deleted when the job ends. Set
# VULCAN_SCRATCH=0 to read straight from GPFS instead.
#
# Resources: 16 CPUs feed the tf.data augmentation workers (RC suggests >= 2 per GPU).
# 64 GB is ample for training and the classical build; check MaxRSS with
#   sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,AllocTRES%60
# after the first run and adjust.
set -uo pipefail

STAGES="${1:-check}"
MARGIN_MIN="${MARGIN_MIN:-30}"
MAX_RESUBMITS="${VULCAN_MAX_RESUBMITS:-6}"
COUNT="${VULCAN_RESUBMIT_COUNT:-0}"
SBATCH_FILE="${SLURM_SUBMIT_DIR}/vulcan_2_5_3.sbatch"   # $0 is a spool copy inside a job
SCRATCH_DIR="/local/${USER}/${SLURM_JOB_ID}"

ARGS=(--stages "${STAGES}" --margin-min "${MARGIN_MIN}")
[[ "${STAGES}" == *classical* ]] && ARGS+=(--reviewed-checks)   # you reviewed E1/E2 interactively
if [[ "${VULCAN_SCRATCH:-1}" != "0" && ( "${STAGES}" == *train* || "${STAGES}" == *segtest* ) ]]; then
  ARGS+=(--scratch --scratch-dir "${SCRATCH_DIR}")
  trap 'rm -rf "${SCRATCH_DIR}"' EXIT                         # RC: clean local scratch before the job ends
fi

cd "${SLURM_SUBMIT_DIR}"
# RC best practice: reset modules first, then load. CUDA 12.2 / cuDNN 8.9 match TF 2.15.
module reset
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Anaconda3
conda activate ml_tf_2.15

echo "job ${SLURM_JOB_ID} on $(hostname): stages=${STAGES} resubmit=${COUNT}/${MAX_RESUBMITS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python vulcan_2_5_3_batch.py "${ARGS[@]}"
rc=$?

if [[ $rc -eq 75 ]]; then
  if (( COUNT + 1 > MAX_RESUBMITS )); then
    echo "training incomplete and resubmit limit ${MAX_RESUBMITS} reached; resubmit manually"
    exit 1
  fi
  NEXT="train"; [[ "${STAGES}" == *segtest* ]] && NEXT="train,segtest"
  sbatch --export=ALL,VULCAN_RESUBMIT_COUNT=$((COUNT + 1)) \
         --dependency=afterany:${SLURM_JOB_ID} "${SBATCH_FILE}" "${NEXT}"
  echo "training incomplete: queued the next job (${NEXT})"
  exit 0
fi
exit $rc
