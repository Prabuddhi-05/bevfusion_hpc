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

# Raise open-files soft limit (up to hard limit if allowed)
echo "[INFO] ulimit before: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
ulimit -n 131072 || true
echo "[INFO] ulimit after:  soft=$(ulimit -Sn) hard=$(ulimit -Hn)"

# Use fast local scratch for reading the .sqfs
SCRATCH="${SLURM_TMPDIR:-/tmp}"
LOCAL_IMG="$SCRATCH/$(basename "$IMG_SQFS")"
echo "[INFO] Copying image to local scratch: $LOCAL_IMG"
cp -f "$IMG_SQFS" "$LOCAL_IMG"

# Remove old container if exists
if enroot list | grep -qx "$CONT"; then
  echo "[INFO] Removing old container..."
  enroot remove -f "$CONT" || true
fi

# Remove old extracted container folder if exists
OLD_CONTAINER_PATH="$HOME/.local/share/enroot/$CONT"
if [ -d "$OLD_CONTAINER_PATH" ]; then
  echo "[INFO] Deleting old extracted container directory: $OLD_CONTAINER_PATH"
  rm -rf "$OLD_CONTAINER_PATH"
fi

# Try parallel extraction with fallback (p=16 -> 8 -> 4 -> 2 -> 1)
try_create() {
  local P="$1"
  echo "[INFO] Starting container creation with debug output (p=$P)..."
  ENROOT_LOG_LEVEL=debug ENROOT_UNSQUASHFS_OPTIONS="-p $P" enroot create -n "$CONT" "$LOCAL_IMG"
}

set +e
try_create 16 || try_create 8 || try_create 4 || try_create 2 || try_create 1
RC=$?
set -e

if [ $RC -ne 0 ]; then
  echo "[ERROR] Container creation failed after all attempts."
  exit $RC
fi

echo "[INFO] Container '$CONT' created successfully."
echo "[INFO] You can monitor size with:"
echo "watch -n 10 'du -sh ~/.local/share/enroot/$CONT && find ~/.local/share/enroot/$CONT | wc -l'"
