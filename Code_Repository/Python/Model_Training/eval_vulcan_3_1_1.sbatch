#!/bin/bash
#SBATCH --job-name=vulcaneval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=amperenodes
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Evaluate Vulcan 3.1.1 (best + final) against 3.0 e13: H4 for each, then the pixel/object
# overlap comparison vs hand-traced gold. Resumable: finished stages are skipped, so if
# 12 h is not enough just resubmit. Submit from Python/Model_Training:
#
#   sbatch --export=ALL,VULCAN_CHECK_ONLY=1 eval_vulcan_3_1_1.sbatch   # verify paths only
#   sbatch eval_vulcan_3_1_1.sbatch                                    # run / resume
#
# Results: /data/user/tdeibert/Nuclear_Scaling/Evaluations/vulcan_3_1_1_vs_3_0/model_compare.csv
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
python -u eval_vulcan_3_1_1.py ${EXTRA[@]+"${EXTRA[@]}"}
rc=$?
echo "finished with exit code ${rc} | $(date)"
exit $rc
