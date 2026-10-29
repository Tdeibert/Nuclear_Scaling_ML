#!/bin/bash
#SBATCH --job-name=vulcanfull
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Full-frame segmentation with Vulcan 3.1.1 final (e55), all timepoints, one t at a time.
# Resumable: finished timepoints are skipped, so if 12 h is not enough just resubmit.
# Submit from Python/Model_Training:
#
#   sbatch --export=ALL,VULCAN_CHECK_ONLY=1 segment_full_3_1_1.sbatch   # paths + map size only
#   sbatch segment_full_3_1_1.sbatch                                    # run / resume
#
# Results: /data/user/tdeibert/Nuclear_Scaling/Evaluations/full_frame_v3_1_1_e55/full_detections.csv
set -uo pipefail
export VULCAN_NOTEBOOK="${SLURM_SUBMIT_DIR}/vulcan_training_3_1_1.ipynb"
cd "${SLURM_SUBMIT_DIR}"
[[ -f "${VULCAN_NOTEBOOK}" ]] || { echo "missing ${VULCAN_NOTEBOOK}"; exit 1; }
module reset
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0
module load Anaconda3
conda activate ml_tf_2.15

echo "job ${SLURM_JOB_ID} on $(hostname) | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
EXTRA=()
[[ "${VULCAN_CHECK_ONLY:-0}" == "1" ]] && EXTRA+=(--check-only)
python -u segment_full_3_1_1.py ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?
echo "finished with exit code ${rc} | $(date)"
exit $rc
