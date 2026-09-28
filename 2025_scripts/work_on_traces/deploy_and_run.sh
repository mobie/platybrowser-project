#!/bin/bash

# The central scratch directory on the cluster
REMOTE_DIR="/scratch/cros/trace_merging"

# The path to your TSV file locally (adjust if your data folder is elsewhere)
LOCAL_TSV_PATH="../../data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-combined-traces/default.tsv"

echo "1. Creating remote directory..."
ssh cluster "mkdir -p $REMOTE_DIR"

echo "2. Transferring scripts and data to the cluster scratch space..."
scp merging_traces.py launch_trace_merging.sh "$LOCAL_TSV_PATH" cluster:$REMOTE_DIR/

echo "3. Submitting the job to SLURM..."
ssh cluster "cd $REMOTE_DIR && chmod +x merging_traces.py && sbatch launch_trace_merging.sh"

echo "Deployment complete!"