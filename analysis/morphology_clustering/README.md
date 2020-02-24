## Snakemake workflow for morphology clustering analysis

### Config file
config.yaml contains all the relevant filepaths - to re-run you will need to modify these to match the version / file 
locations you want to use.  
The workdir: parameter controls where all the results are saved, with the 'dataset_id' controlling the naming of files.  
The dataset_id should be == the platybrowser version you are using.  
Column_subsets contains settings for different subsets of the morphology data - you can add more subsets by
modifying the file in scripts/subset_cols.py

### Installation and running
To run Snakemake - install as specified here: https://snakemake.readthedocs.io/en/stable/getting_started/installation.html
Then run the command: snakemake -k -j from the terminal, while in the folder containig the Snakefile.
-k lets the workflow continue with jobs that don't depend on failed jobs
-j allows the workflow to parallelise over all available cores

You can do a dry run to check everything is working with: snakemake -n

### Environments
The conda environment used is specified in envs/snakemake_morphology.yml 

### Snakefile
The Snakefile is the main file with commands - rule all contains lines to calculate all analysis steps.
If you only wish to run a subset of these, you can comment out lines under input: to only leave the desired results.

### Results
The Snakemake workflow runs the clustering shown in the main paper, but also many other analyses. Here I summarise the output
(but each of the rules are commented to describe what results they produce):  

morph_vs_genes:  
Various plots to assess morphology statistics in comparison to gene expression. e.g. all vs all correlation, scatter plots,
violin plots.  

QC:    
Various plots to assess morphology data. e.g. all vs all correlation, scatter plots of features vs eachother,
histograms of distribution.  

stat_tests:    
Graphs of the number of cells expressing each gene under overlap or vc assignment. (some more experimental tables for 
testing if morphology of cells is signficantly different between cells expressing a gene vs those that don't  - not fully
tested)  

tables:  
after_QC.csv - morphology statistics after various quality control measures. 
Rest of the files are the same but for region / xyz / genes. 

viz_tables:  
Similar to tables folder, but these contain additional columns so they can be visualised with the
Explore Objects Table plugin in Fiji (made by Christian Tischer - install via the EMBL-CBA update site)  

xyz_gradient:  
Plots to look at gradients in morphological features from anterior-posterior (AP) or dorsal-ventral (DV) in the ventral nerve cord.  

Subsets:  
Contains all results for different subsets of morphology data.
In each folder there are the subfolders:  

features_plotted_on_umap:  
Contains gene expression, morphology statistics, Platynereis regions and transformed xyz coordinates plotted on the UMAP. 

genes_per_morph_cluster:  
Heatmaps of genes per morphology cluster.  

louvain:  
All files for the final clustering (many different combinations of parameters are tried) - the final clusters are saved
in the file chosen_clusters.csv & the image of this overlaid on the UMAP is saved as chosen_louvain.png  

morph_heatmap: 
Heatmaps of morphology per morphology cluster.

morph_per_gene:  
Heatmaps of morphology vs expression of particular genes.  

UMAP:  
Files for UMAP under different parameter combinations. Final UMAP is saved as the file umap.csv & the image of the
UMAP as chosen_umap.png  

viz_tables:  
Clustering table with columns to visualise with Explore Objects Table plugin in Fiji. '_cell' means it is ordered by the
label id of cells, '_nuc' means it is ordered by label id of the nucleus. (There are also two tables here for outlier detection
based on morphology features)  

Note:  
UMAP currently produces many warnings - these have no impact on the results. See the
issue on their repository here: https://github.com/lmcinnes/umap/issues/252
