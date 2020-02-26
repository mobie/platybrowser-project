## Snakemake workflow for gene clustering analysis

### Config file
config.yaml contains all the relevant filepaths - to re-run you will need to modify these to match the version / file 
locations you want to use.  
The workdir: parameter controls where all the results are saved, with the 'dataset_id' controlling the naming of files.  
The dataset_id should be == the platybrowser version you are using.  
The cluster parameters / cluster_paramaters binary are for clustering different kinds of gene expression data. Overlap is by
overlap assignment, vc is by virtual cell assignment, 'binary' means gene expression data is first binarised with the thresholds
listed in the config file, 'unique' means that duplicate rows are removed before clustering.

### Installation and running
To run Snakemake - install as specified here: https://snakemake.readthedocs.io/en/stable/getting_started/installation.html
Then run the command: snakemake -k -j from the terminal, while in the folder containig the Snakefile.
-k lets the workflow continue with jobs that don't depend on failed jobs
-j allows the workflow to parallelise over all available cores

You can do a dry run to check everything is working with: snakemake -n

### Environments
The conda environment used is specified in envs/snakemake_genes.yml 

### Snakefile
The Snakefile is the main file with commands - rule all at the top is currently configured to run all analysis steps.
If you only wish to run a subset of these, you can comment out lines under input: to only leave the desired results.

### Results
The Snakemake workflow runs the clustering shown in the main paper (output in subsets > overlap), but also many other analyses. Here I summarise the output:  

QC_overlap folder:  
For overlap gene assignment, this contains graphs of the pearson & spearman correlation of all genes vs all genes.  

QC_vc folder:  
For virtual cell (vc) gene assignment, this contains graphs of the pearson & spearman correlation of all genes vs all genes.  

stat_tests folder:  
Graphs of the number of cells expressing each gene under overlap or vc assignment.  

tables folder:  
after_QC...csv - these files contain gene expression information for cells after various quality control measures. 
Some by overlap assignment (_overlap suffix), some by vc assignmment (_vc suffix). For some gene expression is binarised 
(_binary suffix). Finally, some have duplicate rows removed (_unique suffix).
The merged_ganglia.csv & merged_morph.csv tables are tables for head ganglia ids & morphology statistics respectively.  

viz_tables:  
Same naming scheme as tables folder, but these tables contain additional columns so they can be visualised with the
Explore Objects Table plugin in Fiji (made by Christian Tischer - install via the EMBL-CBA update site)

Subsets:  
Contains all results for different kinds of gene expression data - overlap / vc etc...
In each folder there are the subfolders:  

features_plotted_on_umap:  
Contains head ganglia, gene expression, morphology statistics and Platynereis regions plotted on the UMAP.  

louvain:  
All files for the final clustering (many different combinations of parameters are tried) - the final clusters are saved
in a file ending chosen_clusters.csv & the image of this overlaid on the UMAP is saved as a file ending in chosen_louvain.png  

stats_per_gene_cluster:  
Various statistics per gene cluster. Heatmaps of gene expression per cluster, heatmaps of morphology per cluster,
various specificity scores for genes & gene clusters to head ganglia.  

UMAP:  
Files for UMAP under different parameter combinations. Final UMAP is saved as a file ending umap.csv & the image of the
UMAP as chosen_umap.png  

viz_tables:  
Clustering table with columns to visualise with Explore Objects Table plugin in Fiji  

Note:  
UMAP currently produces many warnings - these have no impact on the results. See the
issue on their repository here: https://github.com/lmcinnes/umap/issues/252
