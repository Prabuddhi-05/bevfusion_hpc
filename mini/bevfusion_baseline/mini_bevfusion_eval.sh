#!/bin/bash
#SBATCH --job-name=bevfusion_eval
#SBATCH --output=logs/mini_bevfusion_eval_%j.out
#SBATCH --time=0-22:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpu
#SBATCH --nodes=1   
#SBATCH --gpus=nvidia_a100:2
#SBATCH --constraint=nvidia_a100
#SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/work/users/pwariyapperuma/bevfusion_final

set -euo pipefail

# ---------------- prep ----------------
mkdir -p logs

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] PWD: $(pwd)"
echo "[INFO] SLURM_JOB_ID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPUs requested by SLURM: ${SLURM_GPUS:-N/A}"

# Enroot is already installed system-wide (no module load)
echo "[INFO] Enroot path: $(command -v enroot || echo 'NOT FOUND')"
enroot version || echo "[WARN] Could not print Enroot version (older Enroot often behaves like this)."

# Enroot/NVIDIA hook hardening (same spirit as train.sh)
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_DISABLE_REQUIRE=1
export NVIDIA_DISABLE_LDCONFIG=1

# NCCL & CPU threading (host-side defaults)
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

IMG_SQFS="/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"
CONT_NAME="bevfusion_enroot"

echo "[INFO] Using image: $IMG_SQFS"
echo "[INFO] Container name: $CONT_NAME"

if [[ ! -f "$IMG_SQFS" ]]; then
  echo "[ERROR] Squashfs image not found at: $IMG_SQFS"
  exit 2
fi

if ! enroot list | grep -qx "$CONT_NAME"; then
  echo "[INFO] Creating Enroot container..."
  enroot create -n "$CONT_NAME" "$IMG_SQFS"
else
  echo "[INFO] Enroot container already exists."
fi

# ---------------- run inside container ----------------
enroot start \
  --rw \
  --mount "$PWD":/workspace \
  --env HOME=/workspace \
  "$CONT_NAME" bash -lc '
    set -euo pipefail
    cd /workspace

    # Container-side sanitization & NCCL hardening (mirror train.sh)
    export NVIDIA_DISABLE_REQUIRE=1
    export NVIDIA_DISABLE_LDCONFIG=1

    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500
    export NCCL_DEBUG=INFO
    export NCCL_ASYNC_ERROR_HANDLING=1
    export TORCH_NCCL_BLOCKING_WAIT=1
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_P2P_LEVEL=NVL
    export NCCL_SHM_DISABLE=0
    export NCCL_NVLS_ENABLE=1
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    echo "[IN-CTR] uname -a: $(uname -a)"
    echo "[IN-CTR] whoami: $(whoami)"
    echo "[IN-CTR] PWD: $(pwd)"
    echo "[IN-CTR] Python: $(python -V || true)"
    echo "[IN-CTR] Conda : $(conda -V  || true)"
    nvidia-smi || true

    # Optional: uncomment if needed
    # source /opt/conda/etc/profile.d/conda.sh
    # conda activate bevfusion

    echo "[IN-CTR] Installing project in dev mode..."
    python setup.py develop

    echo "[IN-CTR] Dataset layout:"
    ls -lah ./data || true

    echo "[IN-CTR] Forcing numpy==1.23.5 via pip..."
    pip uninstall -y numpy || true
    pip install numpy==1.23.5

    #echo "[IN-CTR] Downloading pretrained weights..."
    #chmod +x ./tools/download_pretrained.sh
    #./tools/download_pretrained.sh

    echo "[IN-CTR] Starting 2-GPU evaluation..."
    echo "[IN-CTR] OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-unset} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-unset} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset} NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-unset}"
    torchpack dist-run -np 2 python tools/test.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_mini.yaml \
      runs/run-b1cbb6e1/epoch_6.pth \
      --eval bbox > runs/run-b1cbb6e1/eval_epoch_6_newrun.txt 2>&1
  '
echo "[INFO] Evaluation job finished."
