# plot various features on the umap

# plot morphology on umap
rule plot_morph_on_umap:
    input:
        umap = "subsets/{subset}/UMAP/umap.csv",
        table = "subsets/{subset}/filtered_cols.csv"

    log:
        "logs/{subset}_morph_on_umap.log"

    output:
        morph_dir = directory("subsets/{subset}/features_plotted_on_umap/morph_umap")

    script:
        "../scripts/morph_on_umap.py"


# plot gene expression of each gene on umap
rule plot_genes_on_umap:
    input:
        umap = "subsets/{subset}/UMAP/umap.csv",
        table = 'tables/QC_genes_{gene_assignment}.csv'

    log:
        "logs/{subset}_genes_on_umap_{gene_assignment}.log"

    params:
        gene_assign = "{gene_assignment}"

    output:
        gene_dir = directory("subsets/{subset}/features_plotted_on_umap/gene_{gene_assignment}_umap")

    script:
        "../scripts/genes_on_umap.py"

# plot location of segmented regions on UMAP
rule plot_regions_on_umap:
    input:
        umap = "subsets/{subset}/UMAP/umap.csv",
        table = 'tables/QC_region.csv'

    log:
        "logs/{subset}_regions_on_umap.log"

    output:
        region_dir = directory("subsets/{subset}/features_plotted_on_umap/regions_umap")

    script:
        "../scripts/regions_on_umap.py"

# plot corrected xyz coordinates on UMAP
rule xyz_on_umap:
    input:
        umap="subsets/{subset}/UMAP/umap.csv",
        table = 'tables/QC_xyz.csv'

    log:
        "logs/{subset}_xyz_on_umap.log"

    output:
        xyz_dir = directory("subsets/{subset}/features_plotted_on_umap/xyz_umap")

    script:
        "../scripts/xyz_on_umap.py"
