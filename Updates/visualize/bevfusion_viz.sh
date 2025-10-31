#!/bin/bash
#SBATCH --job-name=bevfusion_viz
#SBATCH --output=logs/bevfusion_viz_%j.out
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --constraint=nvidia_rtx_a6000
#SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=26619055@students.lincoln.ac.uk
#SBATCH --chdir=/home/users/pwariyapperuma/bevfusion_final
#SBATCH --exclude=hpc-novel-gpu01

set -euo pipefail

# ---------- user-configurable bits ----------
RUN_DIR="runs/run-db93ec3e-0301eba1"
VIZ_DIR="${RUN_DIR}/viz_results"
CFG="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
CKPT="${RUN_DIR}/epoch_6.pth"
TEXTLOG="${RUN_DIR}/visualize_val.txt"

# Container image (host) and enroot container name
IMG_SQFS="/home/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs"
CONT_NAME="bevfusion_enroot"

# Swin-T local weight (host path). Inside container it becomes /workspace/pretrained/...
SWIN_PTH_HOST="/home/users/pwariyapperuma/bevfusion_final/pretrained/swin_tiny_patch4_window7_224.pth"
SWIN_PTH_INCTR="/workspace/pretrained/swin_tiny_patch4_window7_224.pth"
# -------------------------------------------

mkdir -p logs "$RUN_DIR"

echo "[INFO] Host: $(hostname)"
echo "[INFO] Workdir: $(pwd)"

# --------- GPU / NVIDIA env ----------
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_DISABLE_REQUIRE=1
export NVIDIA_DISABLE_LDCONFIG=1
export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
# Silence Shapely spam from nuScenes map utils
export PYTHONWARNINGS="ignore:::shapely.errors.ShapelyDeprecationWarning"

# --------- Preflight checks (host) ----------
[[ -f "$IMG_SQFS" ]]       || { echo "[ERR] Missing image: $IMG_SQFS"; exit 2; }
[[ -f "$CFG"    ]]         || { echo "[ERR] Missing config: $CFG";  exit 3; }
[[ -f "$CKPT"   ]]         || { echo "[ERR] Missing checkpoint: $CKPT"; exit 4; }
[[ -f "$SWIN_PTH_HOST" ]]  || { echo "[ERR] Missing local Swin weight: $SWIN_PTH_HOST"; exit 5; }

# --------- Create / reuse Enroot container ----------
if ! enroot list | grep -qx "$CONT_NAME"; then
  echo "[INFO] Creating Enroot container: $CONT_NAME"
  enroot create -n "$CONT_NAME" "$IMG_SQFS"
else
  echo "[INFO] Reusing Enroot container: $CONT_NAME"
fi

# --------- Run inside container ----------
enroot start \
  --rw \
  --mount "$PWD":/workspace \
  --env HOME=/workspace \
  "$CONT_NAME" bash -lc "
    set -euo pipefail
    cd /workspace

    echo \"[IN-CTR] Python: \$(python -V || true)\"
    nvidia-smi || true

    # Verify the Swin-T local weight inside the container
    [[ -f \"$SWIN_PTH_INCTR\" ]] || { echo \"[ERR] Missing Swin weight in container at $SWIN_PTH_INCTR\"; exit 6; }

    # Optional: dev install for project (harmless if repeated)
    python setup.py develop

    # ---------- isolated pip env ----------
    export PIP_TARGET=/workspace/.pip
    mkdir -p \"\$PIP_TARGET\"
    export PYTHONPATH=\"\${PIP_TARGET}:\${PYTHONPATH:-}\"
    export PATH=\"\$PIP_TARGET/bin:\$PATH\"
    export PIP_CONFIG_FILE=/dev/null
    export PIP_NO_CACHE_DIR=1
    export PIP_USER=no
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export PIP_ISOLATED=1

    echo \"[IN-CTR] Ensuring tqdm is available...\"
    python -m pip install --no-deps --ignore-installed --isolated --target=\"\$PIP_TARGET\" --upgrade tqdm >/dev/null || true

    # Fresh output dir each run
    rm -rf \"$VIZ_DIR\" && mkdir -p \"$VIZ_DIR\"

    echo \"[IN-CTR] Starting single-GPU visualization...\"
    echo \"[IN-CTR] CFG=$CFG\"
    echo \"[IN-CTR] CKPT=$CKPT\"
    echo \"[IN-CTR] OUT=$VIZ_DIR\"
    echo \"[IN-CTR] SWIN local file OK: $SWIN_PTH_INCTR\"

    # (Optional) limit frames for a quick smoke test: uncomment next line
    # export VIZ_LIMIT=20

    # Detections only (camera + LiDAR), no segmentation/maps
    PYTHONUNBUFFERED=1 python -u tools/visualize.py \
      \"$CFG\" \
      --mode pred \
      --checkpoint \"$CKPT\" \
      --split val \
      --out-dir \"$VIZ_DIR\" \
      --no-map \
      --bbox-score 0.3 \
      2>&1 | tee \"$TEXTLOG\"

    echo \"[IN-CTR] Sample outputs:\"
    ls -lh \"$VIZ_DIR\" | head -n 10 || true
  "

echo "[INFO] Visualization job finished."
echo "[INFO] Images in: $VIZ_DIR"
echo "[INFO] Log: $TEXTLOG"

