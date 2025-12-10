#!/bin/bash
#SBATCH --job-name=unim2ae_mmim_mini
#SBATCH --output=logs/unim2ae_mmim_mini_%j.out
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

echo "[INFO] Starting UniM2AE MMIM camera+LiDAR training on nuScenes mini (container)..."

# --- Paths ---
CONTAINER_IMAGE=/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs
WORKDIR=/work/users/pwariyapperuma/UniM2AE
# mini dataset root (v1.0-mini)
NUSCENES_DATA=/work/users/pwariyapperuma/UniM2AE/Pretrain/mini_data/v1.0-mini

mkdir -p logs
module load Pyxis Enroot 2>/dev/null || echo "[WARN] Could not load Pyxis/Enroot (ignore if already loaded)"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/work/users:/work/users \
  --container-workdir="${WORKDIR}" \
  bash -lc '
    set -euo pipefail
    echo "[IN-CTR] Host: $(hostname)"
    echo "[IN-CTR] PWD:  $(pwd)"

    VENV_DIR="/work/users/pwariyapperuma/unim2ae_venv"
    python --version || true
    [ -d "$VENV_DIR" ] || python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    python --version
    python -m pip install --upgrade pip
    export HOME=/work/users/pwariyapperuma; mkdir -p "$HOME/.cache"

    # ---------------- Core deps ----------------
    python -m pip install numpy==1.21.6
    python -m pip install "torch==1.9.1+cu111" "torchvision==0.10.1+cu111" -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install "mmcv-full==1.4.0" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

    # IMPORTANT: use BEVFusion mmdet version so SwinTransformer is registered
    python -m pip install "mmdet==2.20.0" --no-build-isolation

    python -m pip install matplotlib==3.5.2 pyquaternion==0.9.9 scikit-learn==1.1.3 setuptools==59.5.0 \
                         scikit-image==0.19.3 timm==0.4.12 "ipython<8.0" ipdb==0.13.13 numba==0.48.0 \
                         llvmlite==0.31.0 yapf==0.30.0 huggingface_hub==0.19.4 httpx==0.24.1 \
                         trimesh==3.9.29 nuscenes-devkit==1.1.10 future tensorboard tqdm Pillow==8.4.0

    # Torchpack for BEVFusion training
    python -m pip install "torchpack==0.3.1" || python -m pip install torchpack

    # mpi4py so torchpack.distributed can import it
    python -m pip install "mpi4py==3.0.3"

    # ---------------- Link nuScenes data (same as before) ----------------
    cd "'"${WORKDIR}"'"
    echo "[IN-CTR] NuScenes root provided: '"${NUSCENES_DATA}"'"

    NU_ROOT="'"${NUSCENES_DATA}"'"
    case "$NU_ROOT" in */v1.0-mini) NU_ROOT="$(dirname "$NU_ROOT")";; esac
    echo "[IN-CTR] Using nuScenes ROOT: $NU_ROOT"

    if [ -d "$NU_ROOT/v1.0-mini" ]; then
      echo "[IN-CTR] Detected v1.0-mini under $NU_ROOT, creating root-level links for mini layout..."
      for sub in samples sweeps maps; do
        if [ -d "$NU_ROOT/v1.0-mini/$sub" ]; then
          if [ -e "$NU_ROOT/$sub" ] || [ -L "$NU_ROOT/$sub" ]; then
            rm -rf "$NU_ROOT/$sub"
          fi
          ln -sfn "$NU_ROOT/v1.0-mini/$sub" "$NU_ROOT/$sub"
          echo "[IN-CTR] Linked $NU_ROOT/$sub -> $NU_ROOT/v1.0-mini/$sub"
        fi
      done
    else
      mkdir -p "$NU_ROOT/maps"
    fi

    for d in data Pretrain/data Finetune/bevfusion/data Finetune/sst/data; do
      mkdir -p "$d"
      [ -e "$d/nuscenes" ] || [ -L "$d/nuscenes" ] && rm -rf "$d/nuscenes"
      ln -sfn "$NU_ROOT" "$d/nuscenes"
    done

    mkdir -p data
    if [ -e data/nuscenes_mini ] || [ -L data/nuscenes_mini ]; then
      rm -rf data/nuscenes_mini
    fi
    ln -sfn data/nuscenes/v1.0-mini data/nuscenes_mini

    echo "[IN-CTR] Symlink targets:"
    for d in data Pretrain/data Finetune/bevfusion/data Finetune/sst/data; do
      echo " - $d/nuscenes -> $(readlink -f "$d/nuscenes")"
    done
    echo " - data/nuscenes_mini -> $(readlink -f data/nuscenes_mini)"

    PKL_ROOT="data/nuscenes"
    MINI_DIR="$PKL_ROOT/v1.0-mini"
    if [ ! -d "$MINI_DIR" ]; then
      echo "[ERROR] Expected $MINI_DIR but not found. Current PKL_ROOT listing:"
      ls -la "$PKL_ROOT" || true
      exit 1
    fi

    pushd "$PKL_ROOT" >/dev/null
      SRC_TRAIN=""
      SRC_VAL=""
      if   [ -f "v1.0-mini/nuscenes_infos_train.pkl" ]; then SRC_TRAIN="v1.0-mini/nuscenes_infos_train.pkl"
      elif [ -f "v1.0-mini/nuscenes_mini_infos_train.pkl" ]; then SRC_TRAIN="v1.0-mini/nuscenes_mini_infos_train.pkl"; fi
      if   [ -f "v1.0-mini/nuscenes_infos_val.pkl" ];   then SRC_VAL="v1.0-mini/nuscenes_infos_val.pkl"
      elif [ -f "v1.0-mini/nuscenes_mini_infos_val.pkl" ];   then SRC_VAL="v1.0-mini/nuscenes_mini_infos_val.pkl";   fi
      [ -n "$SRC_TRAIN" ] && { rm -f nuscenes_infos_train.pkl; ln -sfn "$SRC_TRAIN" nuscenes_infos_train.pkl; }
      [ -n "$SRC_VAL"   ] && { rm -f nuscenes_infos_val.pkl;   ln -sfn "$SRC_VAL"   nuscenes_infos_val.pkl;   }
      echo "[IN-CTR] PKL aliases:"; ls -l nuscenes*_infos_*.pkl || true
    popd >/dev/null

    echo "[IN-CTR] Final data/nuscenes tree (first 120 lines):"
    ls -la "$PKL_ROOT" | sed -n "1,120p"

    # ---------------- Install Finetune/bevfusion code ----------------
    cd Finetune/bevfusion
    python setup.py develop --no-deps

    export OMPI_MCA_rmaps_base_oversubscribe=1
    export NCCL_ASYNC_ERROR_HANDLING=1
    export NCCL_P2P_LEVEL=NVL
    export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29517}
    
    # ---------------- Without MMIM camera+LiDAR training ----------------
    echo "[IN-CTR] Starting Without MMIM camera+LiDAR UniM2AE BEVFusion training on nuScenes mini ..."
    torchpack dist-run -np 4 python tools/train.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/bevfusion_sst.yaml \
      --load_from pretrained/unim2ae-stage1.pth

    echo "[IN-CTR] Without MMIM camera+LiDAR training finished."
  '

echo "[INFO] Job finished."

