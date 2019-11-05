#!/bin/bash
#SBATCH -o "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/copandmir.out"
#SBATCH -e "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/copandmir.err"
#SBATCH --mem 16000
#SBATCH -c 4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p htc
#SBATCH -t 400
echo "starting job"
touch "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/copandmir.started"
START_TIME=$SECONDS
ulimit -c 0
hostname
head -1 /proc/meminfo
module load X11
module load Java
xvfb-run -a -e "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/copandmir.err" /g/almf/software/Fiji.app/ImageJ-linux64 -batch "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/ProSPr_copy_and_mirror.ijm"
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "job finished"
touch "/g/arendt/EM_6dpf_segmentation/platy-browser-data/registration/transfer_ProSPr_data/copandmir.finished"