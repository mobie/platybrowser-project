# calcs / stats per gene cluster

# heatmaps of gene expression per gene cluster (removes genes with no expression for heatmaps per cluster)
rule heatmaps_genes:
    input:
        clusters = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_chosen_clusters.csv",
        genes = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/{gene_assignment}_{metric}_heatmap_genes.log"

    output:
        heatmap_dir = directory("subsets/{gene_assignment}/stats_per_gene_cluster/heatmap_genes_{gene_assignment}_{metric}")

    script:
        "../scripts/gene_heatmaps.R"


# heatmaps for morphology per gene cluster
rule heatmaps_morph:
    input:
        morph = "subsets/{gene_assignment}/tables/{gene_assignment}_{metric}_morphology_normalised.csv"

    log:
        "logs/{gene_assignment}_{metric}_heatmap_morph.log"

    output:
        heatmap_dir = directory("subsets/{gene_assignment}/stats_per_gene_cluster/heatmap_morph_{gene_assignment}_{metric}")

    script:
        "../scripts/morph_heatmaps.R"

def get_full_table(gene_assignment):

    root = gene_assignment
    if 'unique' in gene_assignment:
        # remove unique from the end of the name
        parts = gene_assignment.split('_')
        root = '_'.join(parts[0:len(parts) - 1])

    return "tables/after_QC_%s.csv" % root

# Calculate specificty of gene clusters & individual genes for head ganglia (for individual genes a cutoff of 0.5
# overlap is used to determine expression i.e. over 0.5 == expression, below 0.5 == no expression
# produces some warnings for invalid values - this can occur when a gene is expressed in no cells under the 0.5
# threshold. Doesn't affect any of the calculations / graphs
rule ganglia_specificity:
    input:
        ganglia = 'tables/merged_ganglia.csv',
        full_with_ids = lambda wildcards: get_full_table(wildcards.gene_assignment),
        clustering = 'subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_chosen_clusters.csv'

    params:
        gene_assign = "{gene_assignment}"

    log:
        "logs/ganglia_specificity_{gene_assignment}_{metric}.log"

    output:
        specificity_dir = directory("subsets/{gene_assignment}/stats_per_gene_cluster/specificty_ganglia_{gene_assignment}_{metric}")

    script:
        "../scripts/ganglia_specificity.py"
