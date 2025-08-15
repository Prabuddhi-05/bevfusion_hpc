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

set -euo pipefail

# ---------------- prep ----------------
mkdir -p logs
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] PWD:      $(pwd)"
echo "[INFO] SLURM_JOB: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPUs req: ${SLURM_GPUS:-N/A}"

echo "[INFO] Enroot:  $(command -v enroot || echo NOT FOUND)"
enroot version || echo "[WARN] no enroot version printed"

export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=4

IMG_SQFS="/home/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"
CONT="bevfusion_enroot"

echo "[INFO] Image: $IMG_SQFS"
echo "[INFO] Container: $CONT"
[[ -f "$IMG_SQFS" ]] || { echo "[ERROR] missing $IMG_SQFS"; exit 2; }
if ! enroot list | grep -qx "$CONT"; then
  echo "[INFO] Creating container..."
  enroot create -n "$CONT" "$IMG_SQFS"
else
  echo "[INFO] Container already exists"
fi

# ---------------- run inside container ----------------
enroot start \
  --rw \
  --mount "$PWD":/workspace \
  --env HOME=/workspace \
  "$CONT" bash -lc '
    set -euo pipefail
    cd /workspace

    echo "[IN-CTR] Python: $(python -V)"
    echo "[IN-CTR] Conda:  $(conda -V  || true)"
    nvidia-smi

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
    cat <<'PATCH' > "$TB_PATH"
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

    echo "[IN-CTR] Download weights"
    #chmod +x tools/download_pretrained.sh
    #./tools/download_pretrained.sh

    echo "[IN-CTR] Dist env"
    export MASTER_HOST=localhost
    export MASTER_PORT=29500

    echo "[IN-CTR] Launch training"
    torchpack dist-run -np 4 python tools/train.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
      --load_from pretrained/lidar-only-det.pth
  '

echo "[INFO] Training job finished."

