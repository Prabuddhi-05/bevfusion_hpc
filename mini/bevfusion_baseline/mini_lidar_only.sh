#!/bin/bash
#SBATCH --job-name=bevfusion_train
#SBATCH --output=logs/mini_lidar_%j.out
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

# NCCL + threading
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=8

IMG_SQFS="/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"

echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Using image: $IMG_SQFS"

# ---------------- RUN TRAINING ----------------
srun --ntasks=1 --gpus=4 --gpu-bind=closest --mpi=none \
  --container-image="$IMG_SQFS" \
  --container-mounts="$PWD:/workspace,$PWD/wheelhouse:/workspace/wheelhouse,/dev/shm:/dev/shm" \
  bash -lc '
    set -euo pipefail
    cd /workspace

    echo "[IN-CTR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
    echo "[IN-CTR] Python: $(python -V)"
    echo "[IN-CTR] Conda:  $(conda -V || true)"

    echo "[IN-CTR] Running nvidia-smi to verify GPUs..."
    nvidia-smi || true

    # ---------------- pip deps fix ----------------
    echo "[IN-CTR] Setting up fully isolated pip target..."
    export PIP_TARGET=/workspace/.pip
    mkdir -p $PIP_TARGET
    export PYTHONPATH="${PIP_TARGET}:${PYTHONPATH:-}"
    export PATH=$PIP_TARGET/bin:$PATH
    export PIP_CONFIG_FILE=/dev/null
    export PIP_NO_CACHE_DIR=1
    export PIP_USER=no
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_ISOLATED=1

    echo "[IN-CTR] Installing Python packages into $PIP_TARGET ..."
    python -m pip install --no-deps --ignore-installed --isolated --target=$PIP_TARGET \
        numpy==1.23.5 \
        yapf==0.30.0 \
        more-itertools

    python -c '\''import numpy; print("[IN-CTR] Numpy version used:", numpy.__version__)'\''

    # ---------------- install mmcv ----------------
    pip install --target=$PIP_TARGET /workspace/wheelhouse/mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl

    # ---------------- dev install ----------------
    export PYTHONPATH="/workspace/.pip/lib/python3.8/site-packages:${PYTHONPATH}"
    python setup.py develop --prefix /workspace/.pip

    # ---------------- misc QoL ----------------
    export MPLCONFIGDIR=/workspace/.mplconfig
    mkdir -p "$MPLCONFIGDIR"

    # ---------------- NCCL setup ----------------
    export MASTER_ADDR=127.0.0.1
    export MASTER_PORT=29500
    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=lo
    export NCCL_IB_DISABLE=1
    export NCCL_NET_GDR_LEVEL=0
    export NCCL_P2P_LEVEL=NVL
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

    # ---------------- Single-GPU slot ----------------
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export OMPI_MCA_rmaps_base_oversubscribe=1
    HOST=$(hostname)
    echo "[IN-CTR] Using host $HOST with 4 slot"

    # ---------------- Launch Training (no TensorBoard) ----------------
    echo "[IN-CTR] Launching training with 4 GPUs and 1 task..."
    torchpack dist-run -np 4 -H ${HOST}:1 python tools/train_no_tensorboard.py \
      configs/nuscenes/det/transfusion/secfpn/lidar/voxelnet_0p075_mini.yaml
  '

echo "[INFO] Training job finished."

