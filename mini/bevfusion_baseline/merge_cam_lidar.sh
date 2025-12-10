#!/bin/bash
#SBATCH --job-name=merge_cam_lidar
#SBATCH --output=logs/merge_cam_lidar_%j.out
#SBATCH --time=6-00:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_a100:4
#SBATCH --constraint=nvidia_a100
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/work/users/pwariyapperuma/bevfusion_final

set -euo pipefail

# ---------------- ENV FIXES ----------------
unset LD_LIBRARY_PATH
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_DISABLE_REQUIRE=1
export NVIDIA_DISABLE_LDCONFIG=1

IMG_SQFS="/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Using image: $IMG_SQFS"
echo "[INFO] PWD: $PWD"

# ---------------- RUN MERGE SCRIPT ----------------
srun --ntasks=1 --gpus=1 --gpu-bind=closest --mpi=none \
  --container-image="$IMG_SQFS" \
  --container-mounts="$PWD:/workspace,$PWD/wheelhouse:/workspace/wheelhouse,/dev/shm:/dev/shm" \
  bash -lc '
    set -euo pipefail
    cd /workspace

    echo "[IN-CTR] Python version: $(python -V)"
    echo "[IN-CTR] Running camera+lidar fusion checkpoint merge..."
    python tools/merge_cam_lidar_pretrain.py

    echo "[IN-CTR] Done. Contents of pretrained/mini/:"
    ls -lh pretrained/mini
  '

echo "[INFO] Fusion checkpoint merge job finished."

