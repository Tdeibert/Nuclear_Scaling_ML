#!/bin/bash
#SBATCH --job-name=vulcan253
#SBATCH --partition=amperenodes
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=vulcan253_%j.out
#SBATCH --error=vulcan253_%j.out
#
# Vulcan 2.5.3 batch job. Submit from Python/Model_Training (next to the notebook):
#
#   sbatch vulcan_2_5_3.sbatch check                     # config, pools, GPU; writes nothing
#   sbatch vulcan_2_5_3.sbatch classical                 # classical patches (own job: not resumable)
#   sbatch vulcan_2_5_3.sbatch train                     # trains; resubmits itself until done
#   sbatch vulcan_2_5_3.sbatch segtest                   # segmentation test on the best checkpoint
#
# Chain classical -> train so training starts only if the build succeeded:
#   jid=$(sbatch --parsable vulcan_2_5_3.sbatch classical)
#   sbatch --dependency=afterok:$jid vulcan_2_5_3.sbatch train
#
# Training stops cleanly ~30 min before the wall time (MARGIN_MIN), saves model +
# optimizer + early-stopping state, exits 75, and this script queues the next job
# (up to VULCAN_MAX_RESUBMITS). Add segtest after training with:
#   sbatch vulcan_2_5_3.sbatch train,segtest
set -uo pipefail

STAGES="${1:-check}"
MARGIN_MIN="${MARGIN_MIN:-30}"
MAX_RESUBMITS="${VULCAN_MAX_RESUBMITS:-12}"
COUNT="${VULCAN_RESUBMIT_COUNT:-0}"
SBATCH_FILE="${SLURM_SUBMIT_DIR}/vulcan_2_5_3.sbatch"     # $0 is a spool copy inside a job
EXTRA=""
[[ "$STAGES" == *classical* ]] && EXTRA="--reviewed-checks"   # you reviewed E1/E2 interactively

cd "${SLURM_SUBMIT_DIR}"
# Activate the same environment the VS Code tunnel session uses.
module load Anaconda3 2>/dev/null || module load Miniforge3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml_tf_2.15

echo "job ${SLURM_JOB_ID} on $(hostname): stages=${STAGES} resubmit=${COUNT}/${MAX_RESUBMITS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python vulcan_2_5_3_batch.py --stages "${STAGES}" --margin-min "${MARGIN_MIN}" ${EXTRA}
rc=$?

if [[ $rc -eq 75 ]]; then
  if (( COUNT + 1 > MAX_RESUBMITS )); then
    echo "training incomplete and resubmit limit ${MAX_RESUBMITS} reached; resubmit manually"
    exit 1
  fi
  # Remaining stages after an incomplete train: keep train (+ segtest if requested).
  NEXT="train"; [[ "$STAGES" == *segtest* ]] && NEXT="train,segtest"
  sbatch --export=ALL,VULCAN_RESUBMIT_COUNT=$((COUNT + 1)) \
         --dependency=afterany:${SLURM_JOB_ID} "${SBATCH_FILE}" "${NEXT}"
  echo "training incomplete: queued the next job (${NEXT})"
  exit 0
fi
exit $rc
