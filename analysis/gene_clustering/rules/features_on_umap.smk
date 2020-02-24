# plot various features on the umap

# plot gene expression of each gene on umap
rule plot_genes_on_umap:
    input:
        umap = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv",
        table = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/{gene_assignment}_{metric}_genes_on_umap_{gene_assignment}.log"

    params:
        gene_assignment = '{gene_assignment}'

    output:
        gene_dir = directory("subsets/{gene_assignment}/features_plotted_on_umap/gene_{gene_assignment}_{metric}")

    script:
        "../scripts/genes_on_umap.py"

def get_full_table(gene_assignment):

    root = gene_assignment
    if 'unique' in gene_assignment:
        # remove unique from the end of the name
        parts = gene_assignment.split('_')
        root = '_'.join(parts[0:len(parts) - 1])

    return "tables/after_QC_%s.csv" % root

# plot location of segmented regions on UMAP - for 'unique' tables (as multiple cells
# with identical gene expression are collapsed into one row) the majority region id is plotted. i.e. if one unique id
# corresponds to 4 cells in the head and 3 outside, then it will be assigned to the head.
rule plot_regions_on_umap:
    input:
        umap = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv",
        table = "tables/after_QC_{gene_assignment}.csv",
        region = config['table_paths']['cell_region_mapping'],
        # only necessary for input with 'unique_id'
        full_with_ids = lambda wildcards: get_full_table(wildcards.gene_assignment)

    params:
        gene_assign = "{gene_assignment}",
        label = False

    log:
        "logs/{gene_assignment}_{metric}_regions_on_umap.log"

    output:
        region_dir = directory("subsets/{gene_assignment}/features_plotted_on_umap/region_{gene_assignment}_{metric}")

    script:
        "../scripts/regions_on_umap.py"

# read in table of paired head ganglia, remove null values
rule merge_ganglia:
    input:
        ganglia = config['table_paths']['cell_ganglia_mapping']

    log:
        "logs/ganglia_merge.log"

    output:
        ganglia_merged = 'tables/merged_ganglia.csv'

    script:
        "../scripts/merge_ganglia.py"

# plot ganglia on umap (where possible) - for 'unique' tables (as multiple cells
# with identical gene expression are collapsed into one row) the majority ganglion id is plotted. i.e. if one unique id
# corresponds to 4 cells in the ganglion_1 and 3 in ganglion_2, then it will be assigned to ganglion_1.
rule plot_ganglia_on_umap:
    input:
        umap = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv",
        table = "tables/after_QC_{gene_assignment}.csv",
        region = 'tables/merged_ganglia.csv',
        # only necessary for input with 'unique_id'
        full_with_ids = lambda wildcards: get_full_table(wildcards.gene_assignment)

    params:
        gene_assign = "{gene_assignment}",
        label = True

    log:
        "logs/{gene_assignment}_{metric}_ganglia_on_umap.log"

    output:
        region_dir = directory("subsets/{gene_assignment}/features_plotted_on_umap/ganglia_{gene_assignment}_{metric}")

    script:
        "../scripts/regions_on_umap.py"

# plot morphology on umap (where possible) - can't be done for 'unique' tables as multiple cells
# with identical gene expression are collapsed into one row. Hence one point could come from multiple morphologies
rule plot_morph_on_umap:
    input:
        umap = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv",
        table = "tables/after_QC_{gene_assignment}.csv",
        merged_morph = "tables/merged_morph.csv"

    log:
        "logs/{gene_assignment}_{metric}_morph_on_umap.log"

    output:
        morph_dir = directory("subsets/{gene_assignment}/features_plotted_on_umap/morph_{gene_assignment}_{metric}")

    script:
        "../scripts/morph_on_umap.py"

