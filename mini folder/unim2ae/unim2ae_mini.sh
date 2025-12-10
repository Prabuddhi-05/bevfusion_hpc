#!/bin/bash
#SBATCH --job-name=unim2ae_dataprep
#SBATCH --output=logs/unim2ae_dataprep_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/work/users/pwariyapperuma/UniM2AE

# This script ONLY prepares nuScenes v1.0-mini for UniM²AE/BEVFusion.
# It does NOT launch any training.

set -euo pipefail

echo "[INFO] Starting nuScenes (mini) data prep for UniM²AE at $(date)"

# ---- Paths ----
REPO_ROOT="/work/users/pwariyapperuma/UniM2AE"
# User-provided dataset path (points to the 'v1.0-mini' folder)
NUSCENES_VERSION_DIR="/work/users/pwariyapperuma/UniM2AE/Pretrain/mini_data/v1.0-mini"

# If the given path ends with v1.0-mini, use its parent as root; else use it directly.
if [[ "${NUSCENES_VERSION_DIR}" =~ /v1\.0-mini/?$ ]]; then
  NUSCENES_ROOT="${NUSCENES_VERSION_DIR%/v1.0-mini}"
else
  NUSCENES_ROOT="${NUSCENES_VERSION_DIR}"
fi

echo "[INFO] Using nuScenes root: ${NUSCENES_ROOT}"
echo "[INFO] Expecting version folder: ${NUSCENES_ROOT}/v1.0-mini"

# Sanity checks
if [ ! -d "${NUSCENES_ROOT}/v1.0-mini" ]; then
  echo "[ERROR] Cannot find folder: ${NUSCENES_ROOT}/v1.0-mini"
  echo "        Please ensure your mini dataset is at: ${NUSCENES_ROOT}/v1.0-mini"
  exit 1
fi

if [ ! -d "${NUSCENES_ROOT}/maps" ]; then
  echo "[WARN] Cannot find 'maps' folder under ${NUSCENES_ROOT}."
  echo "      Map JSONs are recommended; some tools may fail without them."
fi

# ---- Prepare output dirs ----
cd "${REPO_ROOT}/Finetune/bevfusion"
mkdir -p data/nuscenes
echo "[INFO] Working in $(pwd)"
echo "[INFO] Output will be in: $(pwd)/data/nuscenes"

# ---- Run MMDet3D/BEVFusion data converter for nuScenes mini ----
echo "[INFO] Running data converter ..."
python tools/create_data.py nuscenes   --version v1.0-mini   --root-path "${NUSCENES_ROOT}"   --out-dir ./data/nuscenes   --extra-tag nuscenes

echo "[INFO] Data converter finished."

# ---- Symlink for Pretrain stage ----
echo "[INFO] Linking data into Pretrain/data ..."
mkdir -p "${REPO_ROOT}/Pretrain/data"
ln -sfn "${REPO_ROOT}/Finetune/bevfusion/data/nuscenes" "${REPO_ROOT}/Pretrain/data/nuscenes"

echo "[INFO] Done. Generated files should be under:"
echo "       ${REPO_ROOT}/Finetune/bevfusion/data/nuscenes"
echo "       (and linked at ${REPO_ROOT}/Pretrain/data/nuscenes)"
echo "[INFO] Completed at $(date)"
