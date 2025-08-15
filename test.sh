#!/bin/bash
#SBATCH --job-name=bevfusion_train
#SBATCH --output=logs/bevfusion_train_%j.out
#SBATCH --time=6-00:00:00
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_rtx_a6000:4
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/home/users/pwariyapperuma/bevfusion_final

# ---------------- Container Creation Only ----------------
set -euo pipefail

export IMG_SQFS="/home/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"
export CONT="bevfusion_enroot"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Creating container: $CONT"
echo "[INFO] From image: $IMG_SQFS"

# Force-remove old container
if enroot list | grep -qx "$CONT"; then
  echo "[INFO] Removing old container..."
  enroot remove -f "$CONT"
fi

# Create container with debug log level
echo "[INFO] Starting container creation with debug output..."
ENROOT_LOG_LEVEL=debug ENROOT_UNSQUASHFS_OPTIONS="-p 1" enroot create -n "$CONT" "$IMG_SQFS"

echo "[INFO] Container '$CONT' created successfully."

