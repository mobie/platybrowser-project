#! /bin/bash

module load Java/1.8.0_221 Maven
JGO=/g/arendt/EM_6dpf_segmentation/platy-browser-data/software/conda/miniconda3/envs/paintera/bin/jgo

# copy the jgo config to use the correct folders
cp install/.jgorc "$HOME/.jgorc"

echo "Starting paintera, this may take a while ..."
$JGO -Xmx120G -XX:+UseConcMarkSweepGC -Dprism.forceGPU=true org.janelia.saalfeldlab:paintera:0.24.1-SNAPSHOT+org.slf4j:slf4j-simple:1.7.25 $1
