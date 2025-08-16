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
#SBATCH --exclude=hpc-novel-gpu01

set -euo pipefail

# ---------------- prep ----------------
mkdir -p logs
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] PWD:      $(pwd)"
echo "[INFO] SLURM_JOB: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPUs req: ${SLURM_GPUS:-N/A}"

echo "[INFO] Enroot:  $(command -v enroot || echo NOT FOUND)"
enroot version || echo "[WARN] no enroot version printed"

# Sanitize env / NVIDIA hook quirks
unset LD_LIBRARY_PATH
export NVIDIA_DISABLE_REQUIRE=1
export NVIDIA_DISABLE_LDCONFIG=1
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

# NCCL + threading
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=4

IMG_SQFS="/home/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"
CONT="bevfusion_enroot"

echo "[INFO] Image: $IMG_SQFS"
echo "[INFO] Container: $CONT"
[[ -f "$IMG_SQFS" ]] || { echo "[ERROR] missing $IMG_SQFS"; exit 2; }

echo "[INFO] Skipping container creation"
if ! enroot list | grep -qx "$CONT"; then
  echo "[ERROR] Container '$CONT' not found. Please create it manually before running this job."
  exit 1
else
  echo "[INFO] Container '$CONT' is available. Proceeding to training."
fi

# ---------------- run inside container via srun (bind GPUs) ----------------
srun --ntasks=1 --gpus=4 --gpu-bind=closest --mpi=none \
  enroot start \
    --rw \
    --mount "$PWD":/workspace \
    --mount /dev/shm:/dev/shm \
    --env HOME=/workspace \
    "$CONT" bash -lc '
      set -euo pipefail
      cd /workspace

      # --- print without tripping set -u if undefined ---
      echo "[IN-CTR] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-UNSET}"
      echo "[IN-CTR] Python: $(python -V)"
      echo "[IN-CTR] Conda:  $(conda -V  || true)"
      echo "[IN-CTR] ldconfig: $(command -v ldconfig || echo missing)"
      command -v ldconfig >/dev/null 2>&1 && ldconfig -vN 2>/dev/null | head -n 10 || true
      df -h /dev/shm || true
      nvidia-smi || true

      # If SLURM/NVIDIA didn’t populate it, fall back explicitly
      if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        echo "[IN-CTR] CUDA_VISIBLE_DEVICES was unset; defaulting to $CUDA_VISIBLE_DEVICES"
      fi

      echo "[IN-CTR] Install dev mode"
      python setup.py develop

      echo "[IN-CTR] Data layout"
      ls -lah data || true

      echo "[IN-CTR] Fix deps"
      pip uninstall -y numpy || true
      pip install numpy==1.23.5
      pip install yapf==0.30.0

      echo "[IN-CTR] Patch TensorBoard hook"
      TB_PATH="/opt/conda/envs/bevfusion/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py"
      cat <<'"'"'PATCH'"'"' > "$TB_PATH"
import tensorboard
try:
    from setuptools._distutils.version import LooseVersion
except ImportError:
    from distutils.version import LooseVersion
if not hasattr(tensorboard, "__version__") or LooseVersion(tensorboard.__version__) < LooseVersion("1.15"):
    raise ImportError("TensorBoard logging requires version >=1.15")
del LooseVersion, tensorboard
from .writer import FileWriter, SummaryWriter  # noqa:F401
from tensorboard.summary.writer.record_writer import RecordWriter  # noqa:F401
PATCH
      echo "[IN-CTR] Patched $TB_PATH"

      echo "[IN-CTR] Fix MMCV"
      pip uninstall -y mmcv-full || true
      pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

      # --- Distributed/NCCL hardening ---
      export MASTER_ADDR=127.0.0.1
      export MASTER_PORT=29500
      export NCCL_DEBUG=INFO
      export NCCL_ASYNC_ERROR_HANDLING=1
      export TORCH_NCCL_BLOCKING_WAIT=1
      export NCCL_SOCKET_IFNAME=lo
      export NCCL_IB_DISABLE=1
      export NCCL_NET_GDR_LEVEL=0
      export NCCL_P2P_LEVEL=NVL
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      echo "[IN-CTR] Dist: MASTER_ADDR=$MASTER_ADDR PORT=$MASTER_PORT"
      echo "[IN-CTR] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"

      echo "[IN-CTR] Launch training"
      torchpack dist-run -np 4 python tools/train.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
        --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
        --load_from pretrained/lidar-only-det.pth
    '

echo "[INFO] Training job finished."
