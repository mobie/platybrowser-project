# stats / calculations on a per gene basis, i.e. only taking cells with expression of that one gene

# umap based on morphology of cells expressing a certain gene (from binarised tables)
rule umap_per_gene:
    input:
        morph = 'subsets/{subset}/normalised.csv',
        genes = 'tables/QC_genes_{gene_assignment}_binary.csv'

    log:
        "logs/{subset}_umap_per_gene_{gene_assignment}.log"

    output:
        out_dir = directory("subsets/{subset}/morph_per_gene/umap_per_gene_{gene_assignment}")

    threads: 8

    script:
        "../scripts/umap_per_gene.py"

# heatmap of morphology for all cells expressing a certain gene
rule heatmap_per_gene:
    input:
        morph_names = 'subsets/{subset}/filtered_cols.csv',
        morph = 'subsets/{subset}/normalised.csv',
        genes = 'tables/QC_genes_{gene_assignment}_binary.csv'

    log:
        "logs/{subset}_heatmap_per_gene_{gene_assignment}.log"

    output:
        out_dir = directory("subsets/{subset}/morph_per_gene/heatmap_per_gene_{gene_assignment}")

#    threads: 8

    script:
        "../scripts/heatmap_per_gene.R"