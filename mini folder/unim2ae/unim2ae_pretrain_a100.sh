#!/bin/bash
#SBATCH --job-name=unim2ae_pretrain
#SBATCH --output=logs/unim2ae_pretrain_%j.out
#SBATCH --time=7-00:00:00
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
#SBATCH --chdir=/work/users/pwariyapperuma/UniM2AE

echo "[INFO] Starting UniM2AE pretraining job (container)..."

# --- Paths ---
CONTAINER_IMAGE=/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs
WORKDIR=/work/users/pwariyapperuma/UniM2AE
NUSCENES_DATA=/work/users/pwariyapperuma/bevfusion_final/data/nuscenes

mkdir -p logs

# (Optional) Pyxis/Enroot modules - safe to omit, but harmless if they warn
module load Pyxis Enroot 2>/dev/null || echo "[WARN] Could not load Pyxis/Enroot (ignore if already loaded)"

# --- Run inside container ---
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/work/users:/work/users \
  --container-workdir="${WORKDIR}" \
  bash -lc '
    set -euo pipefail

    echo "[IN-CTR] Host: $(hostname)"
    echo "[IN-CTR] PWD:  $(pwd)"

    # ---------------- Python venv (NO conda) ----------------
    VENV_DIR="/work/users/pwariyapperuma/unim2ae_venv"

    echo "[IN-CTR] Python version (base):"
    python --version || true

    if [ ! -d "$VENV_DIR" ]; then
      echo "[IN-CTR] Creating venv at $VENV_DIR ..."
      python -m venv "$VENV_DIR"
    fi

    echo "[IN-CTR] Activating venv..."
    # shellcheck disable=SC1090
    source "$VENV_DIR/bin/activate"

    echo "[IN-CTR] Python in venv:"
    python --version

    # ---------------- Install dependencies ----------------
    echo "[IN-CTR] Upgrading pip..."
    python -m pip install --upgrade pip

    # Redirect HOME (and thus any caches) to /work (writable)
    export HOME=/work/users/pwariyapperuma
    mkdir -p "$HOME/.cache"

    echo "[IN-CTR] Installing core numpy (>=1.21 for plyfile)..."
    python -m pip install "numpy==1.21.6"

    echo "[IN-CTR] Installing PyTorch 1.9.1 + cu111 and torchvision 0.10.1..."
    python -m pip install "torch==1.9.1+cu111" "torchvision==0.10.1+cu111" \
      -f https://download.pytorch.org/whl/torch_stable.html

    echo "[IN-CTR] Installing mmcv-full 1.4.0..."
    python -m pip install "mmcv-full==1.4.0" \
      -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

    echo "[IN-CTR] Installing mmdet 2.14.0..."
    python -m pip install "mmdet==2.14.0" --no-build-isolation

    echo "[IN-CTR] Installing remaining pkgs (skimage, timm, numba, yapf, HF hub, etc.)..."
    python -m pip install \
      matplotlib==3.5.2 \
      pyquaternion==0.9.9 \
      scikit-learn==1.1.3 \
      setuptools==59.5.0 \
      scikit-image==0.19.3 \
      timm==0.4.12 \
      "ipython<8.0" \
      ipdb==0.13.13 \
      "numba==0.48.0" \
      "llvmlite==0.31.0" \
      "yapf==0.30.0" \
      "huggingface_hub==0.19.4" \
      "httpx==0.24.1" \
      "trimesh==3.9.29" \
      "nuscenes-devkit==1.1.10" \
      future \
      tensorboard

    # ---------------- Link nuScenes data ----------------
    cd "'"${WORKDIR}"'"
    echo "[IN-CTR] Linking nuScenes from: '"${NUSCENES_DATA}"'"

    mkdir -p Pretrain/data
    mkdir -p Finetune/bevfusion/data
    mkdir -p Finetune/sst/data

    if [ ! -e Pretrain/data/nuscenes ]; then
      ln -s "'"${NUSCENES_DATA}"'" Pretrain/data/nuscenes
    fi
    if [ ! -e Finetune/bevfusion/data/nuscenes ]; then
      ln -s "'"${NUSCENES_DATA}"'" Finetune/bevfusion/data/nuscenes
    fi
    if [ ! -e Finetune/sst/data/nuscenes ]; then
      ln -s "'"${NUSCENES_DATA}"'" Finetune/sst/data/nuscenes
    fi

    echo "[IN-CTR] Pretrain/data:"
    ls -R Pretrain/data || true

    # ---------------- Install UniM2AE Pretrain code ----------------
    echo "[IN-CTR] Running python setup.py develop --no-deps in Pretrain/..."
    cd Pretrain
    python setup.py develop --no-deps

    # ---------------- Launch UniM2AE pretraining (4 GPUs) ----------------
#     export CUDA_VISIBLE_DEVICES=0,1,2,3  # (disabled: let SLURM map GPUs)
    export OMPI_MCA_rmaps_base_oversubscribe=1

    HOST=$(hostname)
    echo "[IN-CTR] Using host $HOST with GPUs: $CUDA_VISIBLE_DEVICES"

    echo "[IN-CTR] Launching UniM2AE pretraining with 4 GPUs..."
# ----- Multi-GPU (4x) NCCL/launcher settings (minimal changes) -----
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}   # set to 0 only if IB is configured & healthy
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29517}
# -------------------------------------------------------------------
    bash tools/dist_train.sh configs/unim2ae_mmim.py 4

    echo "[IN-CTR] UniM2AE pretraining finished."
  '

echo "[INFO] Job finished."

