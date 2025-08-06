#!/bin/bash
#SBATCH --job-name=bevfusion_train            # Job name (Batch script)
#SBATCH --output=logs/bevfusion_train_%j.out  # Output log file (print, crash messages)
#SBATCH --time=6-00:00:00                     # Max runtime: 6 days (Allowable - 7 days)
#SBATCH --mem=256G                            # Allocate 256 GB memory
#SBATCH --ntasks=1  			      # Number of tasks
#SBATCH --cpus-per-task=32                    # CPUs per task
#SBATCH --partition=gpu                       # Use GPU partition
#SBATCH --nodes=1                             # Run on 1 node
#SBATCH --gpus=nvidia_rtx_a6000:4             # Request 4 A6000 GPUs
#SBATCH --constraint=nvidia_rtx_a6000         # Restrict to A6000 GPUs
#SBATCH --qos=long                            # Use long queue
#SBATCH --mail-type=BEGIN,END,FAIL            # Send email on start, end, or fail
#SBATCH --mail-user=26619055@students.lincoln.ac.uk                # Email address
#SBATCH --chdir=/home/users/pwariyapperuma/bevfusion_final         # Set working dir

set -euo pipefail                             # Safe script: stop on error or unset vars

# ---------------- prep ----------------
mkdir -p logs                                 # Make log folder
echo "[INFO] Hostname: $(hostname)"           # Print host
echo "[INFO] PWD:      $(pwd)"                # Print working dir
echo "[INFO] SLURM_JOB: ${SLURM_JOB_ID:-N/A}" # Print job ID
echo "[INFO] GPUs req: ${SLURM_GPUS:-N/A}"    # Print requested GPUs

echo "[INFO] Enroot:  $(command -v enroot || echo NOT FOUND)" # Check enroot
enroot version || echo "[WARN] no enroot version printed"     # Print version

export NVIDIA_VISIBLE_DEVICES=all                             # Use all visible GPUs
export NVIDIA_DRIVER_CAPABILITIES=compute,utility             # GPU features
export NCCL_IB_DISABLE=1                                      # Disable InfiniBand
export NCCL_P2P_LEVEL=NVL 				      # Set P2P level
export OMP_NUM_THREADS=4 				      # Set OpenMP threads

IMG_SQFS="/home/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"    # Path to container image
CONT="bevfusion_enroot"                                                              # Container name

echo "[INFO] Image: $IMG_SQFS"
echo "[INFO] Container: $CONT"
[[ -f "$IMG_SQFS" ]] || { echo "[ERROR] missing $IMG_SQFS"; exit 2; }
if ! enroot list | grep -qx "$CONT"; then
  echo "[INFO] Creating container..."
  enroot create -n "$CONT" "$IMG_SQFS"  					    # Create container
else
  echo "[INFO] Container already exists"
fi

# ---------------- run inside container ----------------
enroot start \
  --rw \                      # Enable read/write mode
  --mount "$PWD":/workspace \ # Mount current dir as /workspace
  --env HOME=/workspace \     # Set HOME env var
  "$CONT" bash -lc '          # Start container with login shell
    set -euo pipefail         # Safe bash settings
    cd /workspace             # Change to workspace dir

    echo "[IN-CTR] Python: $(python -V)"           # Print Python version
    echo "[IN-CTR] Conda:  $(conda -V  || true)"   # Print Conda version
    nvidia-smi                                     # Show GPU info

    echo "[IN-CTR] Install dev mode"               
    python setup.py develop                        # Install package in dev mode

    echo "[IN-CTR] Data layout"
    ls -lah data || true                           # List data folder contents

    echo "[IN-CTR] Fix deps"
    pip uninstall -y numpy || true                 # Uninstall old numpy
    pip install numpy==1.23.5                      # Install required numpy version
    pip install yapf==0.30.0                       # Install yapf

    echo "[IN-CTR] Patch TensorBoard hook"
    TB_PATH="/opt/conda/envs/bevfusion/lib/python3.8/site-packages/torch/utils/tensorboard/__init__.py"   # Overwrite TensorBoard init script
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
    pip uninstall -y mmcv-full || true                                                                      # Remove existing MMCV
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html   # Install compatible MMCV

    echo "[IN-CTR] Download weights"
    chmod +x tools/download_pretrained.sh
    ./tools/download_pretrained.sh                                                                          # Download pre-trained weights

    echo "[IN-CTR] Dist env"
    export MASTER_HOST=localhost                                                                            # Set master for distributed
    export MASTER_PORT=29500										    # Set master port

    echo "[IN-CTR] Launch training"
    torchpack dist-run -np 4 python tools/train.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
      --load_from pretrained/lidar-only-det.pth
  '

echo "[INFO] Training job finished." # Job end message 

