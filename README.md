
# Novel HPC Cluster - for BEVFusion Training 

This summarizes the steps followed for running BEVFusion training on the Novel HPC cluster, including VPN access, data transfer, Enroot setup, etc.

---

## Local Machine Setup

### 1. Install Cisco Secure VPN Client
```bash
cd ~/Downloads
chmod +x cisco-secure-client-linux64-5.1.10.233-core-vpn-webdeploy-k9.sh
sudo ./cisco-secure-client-linux64-5.1.10.233-core-vpn-webdeploy-k9.sh
```


### 2. Launch VPN
```bash
/opt/cisco/secureclient/bin/vpnui
```

---

### 3. Connect to HPC Cluster

```bash
ssh -CX pwariyapperuma@login.novel.hpc.network.uni
```

---

## Transferring Data to HPC

### 1. Install SSHFS and mount HPC directory locally
```bash
sudo apt update
sudo apt install sshfs
mkdir ~/cluster
sshfs pwariyapperuma@login.novel.hpc.network.uni:/home/users/pwariyapperuma ~/cluster
```

### 2. Copy nuScenes data
Use the following scripts with ``scp`` commands:
- `copy_nuscenes_partial.sh`
- `copy_nuscenes_samples.sh`

### 3. Remove a folder on HPC
```bash
ssh pwariyapperuma@login.novel.hpc.network.uni "rm -rvf ~/bevfusion_final/data/nuscenes/sweeps/CAM_FRONT_RIGHT"
```

### 4. Watch live file counts in a folder
```bash
watch -n 10 'ssh pwariyapperuma@login.novel.hpc.network.uni "find /home/users/pwariyapperuma/bevfusion_final/data/nuscenes/sweeps/LIDAR_TOP -type f | wc -l"'

```

### rsync command (example)
```bash
rsync -avz -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
--progress "/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes/test" \
pwariyapperuma@login.novel.hpc.network.uni:/home/users/pwariyapperuma/bevfusion_final/data/nuscenes/

```

https://www.digitalocean.com/community/tutorials/how-to-copy-files-with-rsync-over-ssh

Good for large datasets, resumes on failure, copy only the differences (not all files always)

---

## Enroot & Docker Setup (Local Machine)

Enroot is a lightweight, rootless container runtime by NVIDIA designed for running GPU-accelerated workloads in HPC environments.

### 1. Install Enroot dependencies
```bash
sudo apt update
sudo apt install curl gawk jq squashfs-tools parallel bsdmainutils -y
```

### 2. Download Enroot
Download from: https://github.com/NVIDIA/enroot/releases  
Install:
```bash
sudo dpkg -i enroot_3.5.0-1_amd64.deb enroot+caps_3.5.0-1_amd64.deb
enroot version
```

### 3. Setup Docker
```bash
docker images
sudo usermod -aG docker $USER
newgrp docker
```

### 4. Convert Docker image to SQFS
```bash
enroot import -o bevfusion_final.sqfs dockerd://bevfusion_final:latest
scp ~/bevfusion_final.sqfs pwariyapperuma@login.novel.hpc.network.uni:/home/users/pwariyapperuma/bevfusion_final/docker/
```

---

## Count Files (Inside Cluster)

```bash
echo "Counting files in 'samples'..."
for folder in /home/users/pwariyapperuma/bevfusion_final/data/nuscenes/samples/*; do
    count=$(find "$folder" -type f | wc -l)
    name=$(basename "$folder")
    echo "$name: $count files"
done

echo "Counting files in 'sweeps'..."
for folder in /home/users/pwariyapperuma/bevfusion_final/data/nuscenes/sweeps/*; do
    count=$(find "$folder" -type f | wc -l)
    name=$(basename "$folder")
    echo "$name: $count files"
done
```

---

## GPU Cluster Usage

### Check GPU availability
```bash
sinfo -p gpu -o "%N %G %T %D %C"
```

```bash
squeue -p gpu -o "%.18i %.9P %.8u %.2t %.10M %.6D %R""
```

### Monitor Training Logs Live
```bash
tail -f logs/bevfusion_train_<jobID>.out
```

---

## Enroot Container Issues and Fixes

### Problem - 1
After training BEVFusion model successfully for weeks, an unexpected error occurred when launching the container.

```
enroot-switchroot: failed to execute: /bin/sh: No such file or directory
```

This means the container was corrupted or partially extracted.

### Solution
To resolve this, the container was cleanly re-extracted (`test.sh`) from the original .sqfs file.
- Forced cleanup of old containers
- Re-extract the container with using multiple threads

### Monitor Progress
```bash
watch -n 10 'du -sh ~/.local/share/enroot/bevfusion_enroot && find ~/.local/share/enroot/bevfusion_enroot | wc -l'
```
This shows real-time updates on size and file count during container creation. (Time taken = around 24 hours)

### Problem - 2 (on Node 1: `hpc-novel-gpu01`)

During training on Node 1, the following error repeatedly occurred.

```bash
nvidia-container-cli: ldcache error: process /usr/sbin/ldconfig terminated with signal 9
[ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 1
srun: error: hpc-novel-gpu01: task 0: Exited with exit code 1
```

#### Cause
GPU driver configuration problem

#### Solution

* Permanently exclude Node 1 from training jobs using the SLURM directive.

```bash
#SBATCH --exclude=hpc-novel-gpu01
```

---

### Problem - 3 (On other Nodes: `hpc-novel-gpu02–04`)

When training was run on other nodes after container creation, the following message would appear and the job would hang indefinitely.

```bash
NCCL version 2.10.3+cuda11.3
```

#### Cause

* NCCL multi-GPU communication failed due to missing or misconfigured environment variables.

#### Solution

Add the NVIDIA and NCCL environment variables in the SLURM script before training starts.

```bash
# NVIDIA Runtime Fixes
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export NVIDIA_DISABLE_REQUIRE=1
export NVIDIA_DISABLE_LDCONFIG=1

# NCCL + CUDA Communication Flags
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
```

Also, add a fallback inside the container in case `CUDA_VISIBLE_DEVICES` is not set:

```bash
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "[IN-CTR] CUDA_VISIBLE_DEVICES was unset; defaulting to $CUDA_VISIBLE_DEVICES"
fi
```
---

###  Problem - 4 (Container works only on one node (Node-dependent Enroot setup)

After successfully running training on a specific node (e.g., `hpc-novel-gpu04`), submitting the same job on other nodes (e.g., `gpu01` or `gpu03`), the container was stuck in the creation stage for many days (2+ days).

###  Cause
This happened because the container was created as a named Enroot container, which is stored locally on a single compute node.  
When the job was scheduled on a different node, the system tried to re-extract the entire container from the `.sqfs` file.  
This process can take many hours/days.

###  Solution
The workflow was updated to use Pyxis instead of a named Enroot container.

- **Old:** Pre-extracted named Enroot container tied to one node.  
- **New:** Directly mount the `.sqfs` container file at runtime using Pyxis (`--container-image`).

This makes the container:
- Node-independent (works on any compute node)  
- Start instantly (no pre-extraction required)  
- Easier to maintain and reproduce

###  Example SLURM Snippet (Pyxis) - refer `bevfusion_train_pyxis.sh`

```bash
srun --ntasks=1 --gpus=4 --gpu-bind=closest --mpi=none \
  --container-image="/work/users/pwariyapperuma/bevfusion_final/docker/bevfusion_final.sqfs" \
  --container-mounts="$PWD:/workspace,$PWD/wheelhouse:/workspace/wheelhouse,/dev/shm:/dev/shm" \
  bash -lc '
    set -euo pipefail
    cd /workspace
    torchpack dist-run -np 4 -H ${HOST}:4 python tools/train.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
      --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth \
      --load_from pretrained/lidar-only-det.pth
  '
```

**Result:** Container launches immediately on any node — no extraction delays, no node-specific failures.

---

# BEVFusion Visualization (Camera + LiDAR)

This visualizes **3D object detections** from trained BEVFusion models on both **LiDAR BEV** and **camera images**, using a single-GPU setup.

---

### 1. `visualize.py`
A **single-GPU visualization script** that generates 3D boxes from BEVFusion outputs on LiDAR BEV and camera views.

#### Features
- Adds `--no-map` (disables segmentation overlay).
- Supports two modes via `--mode`:
  - `"pred"` → draw model predictions.  
  - `"gt"` → draw ground-truth boxes for comparison.
- Uses `--bbox-score` to filter low-confidence detections.
- Uses `--bbox-classes` to select class IDs.
- Dataset control with `--split`:
  - `train` or `val` → chooses which dataset split to visualize.
- Uses `VIZ_LIMIT` (optional) to visualize only the first N validation samples.
- Automatically saves outputs in per-camera folders:
  ```
  viz_results/
    ├── lidar/
    ├── camera-0/
    ├── camera-1/
    ...
  ```
- Each object class is drawn with a fixed color. 
---

### 2. `default.yaml`
A **config file** for visualization.

#### Change
- Local Swin-T checkpoint path:
  ```
  /workspace/pretrained/swin_tiny_patch4_window7_224.pth
  ```
  *(No internet download required.)*
---

### 3. `bevfusion_viz.sh`
A **SLURM batch script** to run visualization inside your Enroot container.

#### What It Does
- Mounts pretrained weights to `/workspace`.
- Ensures `tqdm` is installed.
- Executes:
  ```bash
  python tools/visualize_images.py       configs/.../default_viz.yaml       --mode pred       --checkpoint runs/run-xxxx/epoch_6.pth       --out-dir runs/run-xxxx/viz_results       --no-map --bbox-score 0.3
  ```
- Logs progress and saves visuals under `viz_results/`.

---


### Why `test.py` Failed for Visualization
Initially, visualization was attempted via:
```bash
torchpack dist-run -np 4 python tools/test.py --show --show-dir ...
```
However:
- `multi_gpu_test()` in `test.py` **does not support `show` or `show-dir` arguments**.
- This led to:  
  ```
  TypeError: multi_gpu_test() got an unexpected keyword argument 'show'
  ```
Hence, the **dedicated `visualize.py`** was run directly.

---


