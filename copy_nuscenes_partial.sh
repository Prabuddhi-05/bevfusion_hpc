#!/bin/bash

# Local base path (on the SSD)
SRC_BASE="/media/prabuddhi/Crucial X92/bevfusion-main/data/nuscenes"

# Destination on HPC cluster
DEST_USER="pwariyapperuma"
DEST_HOST="login.novel.hpc.network.uni"
DEST_BASE="~/bevfusion_final/data/nuscenes"

# Folders to exclude
EXCLUDE=("samples" "sweeps" "nuscenes_gt_database")

# Create destination directory on HPC
ssh "${DEST_USER}@${DEST_HOST}" "mkdir -p ${DEST_BASE}"

# Copy all folders except excluded ones
for item in "$SRC_BASE"/*; do
    name=$(basename "$item")

    # Skip excluded folders
    if [[ " ${EXCLUDE[*]} " == *" $name "* ]]; then
        echo "Skipping: $name"
        continue
    fi

    echo "Copying: $name"
    scp -r "$item" "${DEST_USER}@${DEST_HOST}:${DEST_BASE}/"
done

