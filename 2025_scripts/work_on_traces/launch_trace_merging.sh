#!/bin/bash

#SBATCH -t 10:00:00
#SBATCH --job-name trace_merge
#SBATCH -N 1
#SBATCH -p bigmem
#SBATCH -n 128
#SBATCH --mem 256G
#SBATCH --output /scratch/cros/trace_merging/trace_merge_%j_out.txt
#SBATCH --error /scratch/cros/trace_merging/trace_merge_%j_err.txt

unset XDG_RUNTIME_DIR
module purge
source ~/.bash_profile

# Load the exact MinIO client module version found by spider
module load mc/2024-05-03T11-21-07Z

# Centralize everything in the scratch directory
export SCRATCH_DIR="/scratch/cros/trace_merging"
cd "$SCRATCH_DIR"

echo "Using scratch directory: $SCRATCH_DIR"

# Force uv to use the local scratch drive for caching to avoid NFS corruption!
export UV_CACHE_DIR="$SCRATCH_DIR/uv_cache"
echo "Set UV cache to: $UV_CACHE_DIR"

# Setup MinIO client alias using the loaded mc module
mc alias set embl_anon https://s3.embl.de "" "" --api S3v4

# Helper function to check cache and copy if missing
download_if_missing() {
    local s3_path=$1
    local local_dir=$2
    local name=$3

    if [ -d "$local_dir" ]; then
        echo "[$name] Already cached at $local_dir. Skipping download."
    else
        echo "[$name] Downloading..."
        mc cp -r --quiet "$s3_path/" "$local_dir/"
    fi
}

# Sync S3 images to the local scratch space
download_if_missing \
    "embl_anon/platybrowser-2025/demo-v0/sbem-6dpf-1-whole-traces-david.n5" \
    "${SCRATCH_DIR}/sbem-6dpf-1-whole-traces-david.n5" \
    "David's Traces"

download_if_missing \
    "embl_anon/platybrowser/rawdata/sbem-6dpf-1-whole-traces.n5" \
    "${SCRATCH_DIR}/sbem-6dpf-1-whole-traces.n5" \
    "Kevin's Traces"

download_if_missing \
    "embl_anon/platybrowser/0.0.0/images/local/sbem-6dpf-1-whole-segmented-nuclei.n5" \
    "${SCRATCH_DIR}/sbem-6dpf-1-whole-segmented-nuclei.n5" \
    "Nuclei"

echo "Downloads complete. Launching merge script..."

# Run the python script
./merging_traces.py