#!/bin/bash

# Local path in the SSD
SRC="/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes/sweeps/LIDAR_TOP"

# Destination on HPC cluster
DEST_USER="pwariyapperuma"
DEST_HOST="login.novel.hpc.network.uni"
DEST_DIR="/home/users/pwariyapperuma/bevfusion_final/data/nuscenes/sweeps"

# Create destination directory on HPC
ssh "${DEST_USER}@${DEST_HOST}" "mkdir -p ${DEST_DIR}"

# Copy only the folder
echo "Copying: sweeps/LIDAR_TOP"
scp -r "$SRC" "${DEST_USER}@${DEST_HOST}:${DEST_DIR}/"



