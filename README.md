
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
Use the following scripts:
- `copy_nuscenes_partial.sh`
- `copy_nuscenes_samples.sh`

### 3. Remove a folder on HPC
```bash
ssh pwariyapperuma@login.novel.hpc.network.uni "rm -rvf ~/bevfusion_final/data/nuscenes/sweeps/CAM_FRONT_RIGHT"
```

### 4. Watch file counts in LIDAR folder
```bash
watch -n 10 'ssh pwariyapperuma@login.novel.hpc.network.uni "find /home/users/pwariyapperuma/bevfusion_final/data/nuscenes/sweeps/LIDAR_TOP -type f | wc -l"'
```

---

## Enroot & Docker Setup (Local Machine)

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
sinfo -p gpu -o "%N %G"
```

**Example Output:**
```
NODELIST GRES
hpc-novel-gpu[01,04] gpu:nvidia_rtx_a6000:8(S:0-1)
hpc-novel-gpu[02-03] gpu:nvidia_rtx_a6000:4(S:0)
```

### Monitor Training Logs Live
```bash
tail -f logs/bevfusion_train_<jobID>.out
```

---
