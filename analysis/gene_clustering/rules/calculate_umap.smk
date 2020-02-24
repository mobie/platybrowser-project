# calculate UMAP for specified k & provided metric (e.g. euclidean or jaccard) with default mindist
# Allows to screen different values of k to find a UMAP that looks good
rule calculate_UMAP_ks:
    input:
        table = "tables/after_QC_{gene_assignment}.csv"

    params:
        k = "{k}",
        metric = '{metric}'

    output:
        fig = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_default_mindist_k_{k}.png"

    log:
        "logs/{gene_assignment}_{metric}_{k}_calc_umap_ks.log"

    script:
        "../scripts/calculate_umap_ks.py"

def get_k_umap(gene_assignment):
    if 'binary' in gene_assignment:
        return config['cluster_parameters_binary'][gene_assignment]["chosen_umap_k"]

    else:
        return config['cluster_parameters'][gene_assignment]["chosen_umap_k"]

def get_mindist(gene_assignment):
    if 'binary' in gene_assignment:
        return config['cluster_parameters_binary'][gene_assignment]["chosen_umap_mindist"]

    else:
        return config['cluster_parameters'][gene_assignment]["chosen_umap_mindist"]

# calculate UMAP for chosen k, specified mindist & provided metric (e.g. euclidean or jaccard)
# Allows to screen different values for mindist to find a UMAP that looks good
rule calculate_UMAP_mindists:
    input:
        table = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/{gene_assignment}_{mindist}_{metric}_calc_umap_mindists.log"

    params:
        chosen_k = lambda wildcards: get_k_umap(wildcards.gene_assignment),
        mindist = "{mindist}",
        metric = '{metric}'

    output:
        fig = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_mindist_{mindist}.png"

    script:
        "../scripts/calculate_umap_mindists.py"

# calculate final UMAP with chosen parameters
rule save_chosen_umap:
    input:
        table = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/{gene_assignment}_{metric}_save_chosen_umap.log"

    params:
        chosen_k = lambda wildcards: get_k_umap(wildcards.gene_assignment),
        chosen_mindist = lambda wildcards: get_mindist(wildcards.gene_assignment),
        metric = '{metric}'

    output:
        umap_table = 'subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv',
        fig = "subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_chosen_umap.png"


    script:
        "../scripts/save_chosen_umap.py"