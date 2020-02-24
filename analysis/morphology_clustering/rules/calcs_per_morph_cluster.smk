# calcs / stats per morphology cluster

# heatmaps of gene expression per morphology cluster
rule heatmaps_genes:
    input:
        clusters = "subsets/{subset}/louvain/chosen_clusters.csv",
        genes = 'tables/QC_genes_{gene_assignment}.csv'

    log:
        "logs/{subset}_heatmap_genes_{gene_assignment}.log"

    output:
        heatmap_dir = directory("subsets/{subset}/genes_per_morph_cluster/genes_{gene_assignment}_heatmap")

    script:
        "../scripts/gene_heatmaps.R"

# heatmaps of morphology per morphology cluster
rule heatmaps_morph:
    input:
        clusters = "subsets/{subset}/louvain/chosen_clusters.csv",
        morph_names = 'subsets/{subset}/filtered_cols.csv',
        morph = 'subsets/{subset}/normalised.csv'

    log:
        "logs/{subset}_heatmap_morph.log"

    output:
        heatmap_dir = directory("subsets/{subset}/morph_heatmap")

    script:
        "../scripts/morph_heatmaps.R"