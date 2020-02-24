# calculate louvain partitioning

# Calculate KNN graph from table using specified k & distance metric - save as gpickle
rule calculate_KNN:
    input:
        table = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/{gene_assignment}_{metric}_{k}_calculate_KNN.log"

    params:
        k = "{k}",
        metric='{metric}'

    output:
        KNN = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_KNN_{k}.txt"

    script:
        "../scripts/calculate_louvain_KNN.py"

# Calculate louvain partitions from given KNN (with specified resolution)
# save as a pickled list where length == length of original input table, each entry is the partition that row
# is assigned to
rule calculate_louvain_partitions:
    input:
        KNN = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_KNN_{k}.txt"

    log:
        "logs/{gene_assignment}_{metric}_{k}_{res}_calculate_partitions.log"

    params:
        res = "{res}"

    output:
        clusterings = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_louvain_{k}_res_{res}.txt"

    script:
        "../scripts/calculate_louvain_partitions.py"

# plot given clustering on umap
rule plot_all_clusters_on_umap:
    input:
        umap = 'subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv',
        cluster = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_louvain_{k}_res_{res}.txt"

    log:
        "logs/{gene_assignment}_{k}_{res}_{metric}_plot_all_clusters.log"

    params:
        k = "{k}",
        res = "{res}"

    output:
        fig = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_louvain_{k}_res_{res}.png"

    script:
        "../scripts/plot_clusters_on_umap.py"

def get_cluster_filename (gene_assignment, metric):
    if 'binary' in gene_assignment:
        chosen_k = config['cluster_parameters_binary'][gene_assignment]['chosen_louvain_k']
        chosen_res = config['cluster_parameters_binary'][gene_assignment]['chosen_louvain_res']

    else:
        chosen_k = config['cluster_parameters'][gene_assignment]['chosen_louvain_k']
        chosen_res = config['cluster_parameters'][gene_assignment]['chosen_louvain_res']

    file_name = 'subsets/%s/louvain/%s_%s_louvain_%s_res_%s.txt' % (gene_assignment, gene_assignment, metric, chosen_k, chosen_res)
    return file_name

def get_cluster_parameters (gene_assignment, param):
    if 'binary' in gene_assignment:
        return config['cluster_parameters_binary'][gene_assignment]["chosen_louvain_" + param]

    else:
        return config['cluster_parameters'][gene_assignment]["chosen_louvain_" + param]

def get_filtered_tables(gene_assignment, param):

    root = gene_assignment
    if 'unique' in gene_assignment:
        # remove unique from the end of the name
        parts = gene_assignment.split('_')
        root = '_'.join(parts[0:len(parts) - 1])

    if param == 'viz':
        return "viz_tables/after_QC_%s_viz.csv" % root
    else:
        return "tables/after_QC_%s.csv" % root


# Save chosen louvain clustering as a UMAP, a table with clusters & a table for the Explore Object Tables plugin
# Also save tables of morphology statistics with column added for chosen gene clustering
rule save_chosen_louvain:
    input:
        umap = 'subsets/{gene_assignment}/UMAP/{gene_assignment}_{metric}_umap.csv',
        filtered = "tables/after_QC_{gene_assignment}.csv",
        cluster_file = lambda wildcards: get_cluster_filename(wildcards.gene_assignment, wildcards.metric),
        viz_table = lambda wildcards: get_filtered_tables(wildcards.gene_assignment, 'viz'),
        full_with_ids = lambda wildcards: get_filtered_tables(wildcards.gene_assignment, 'full'),
        merged_morph = "tables/merged_morph.csv"

    log:
        "logs/{gene_assignment}_{metric}_ save_chosen_louvain.log"

    params:
        chosen_k = lambda wildcards: get_cluster_parameters(wildcards.gene_assignment, 'k'),
        chosen_res = lambda wildcards: get_cluster_parameters(wildcards.gene_assignment, 'res'),
        gene_assign = "{gene_assignment}"

    output:
        fig = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_chosen_louvain.png",
        merged_table = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_chosen_clusters.csv",
        viz_table = "subsets/{gene_assignment}/viz_tables/{gene_assignment}_{metric}_louvain_clusters_cell.csv",
        # note morphology table filters any rows that fail any of the haralick calculations - so these are not included
        # in the morphology heatmap values
        morph_table = "subsets/{gene_assignment}/tables/{gene_assignment}_{metric}_morphology.csv",
        morph_table_normalised = "subsets/{gene_assignment}/tables/{gene_assignment}_{metric}_morphology_normalised.csv"

    script:
        "../scripts/save_chosen_louvain.py"


def get_all_cluster_filepaths (gene_assignment, metric):
    ks = config['louvain']['ks']
    res = config['louvain']['resolution']

    result = []
    for k, r in zip(ks, res):
            result.append("subsets/%s/louvain/%s_%s_louvain_%s_res_%s.txt" % (gene_assignment, gene_assignment, metric, k, r))

    return result

# make a summary of all clusterings for that subset (with all the different parameter combinations used)
# makes a summary table with one row per clustering with a column for the used k, the used res, the total number
# of clusters, then the assignments for each row in the original table labelled from 0 to nrow(table)
rule make_clusters_summary:
    input:
        clusters = lambda wildcards: get_all_cluster_filepaths(wildcards.gene_assignment, wildcards.metric)

    log:
        "logs/clusters_summary_{gene_assignment}_{metric}.log"

    output:
        summary_table = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_summary_table_clusters.csv"

    script:
        "../scripts/summary_clusters.py"

# Calculate various cluster metrics over summary table, and make graphs. Idea is to aid with choosing the k / res
# to use for the final plots
rule calculate_cluster_metrics:
    input:
        summary_table = "subsets/{gene_assignment}/louvain/{gene_assignment}_{metric}_summary_table_clusters.csv",
        raw_data = "tables/after_QC_{gene_assignment}.csv"

    log:
        "logs/cluster_metrics_{gene_assignment}_{metric}.log"

    output:
        metrics_table = "subsets/{gene_assignment}/louvain/metrics/{gene_assignment}_{metric}_metric_table.csv",
        plots = directory("subsets/{gene_assignment}/louvain/metrics/{metric}_plots")

    script:
        "../scripts/calculate_cluster_metrics.py"
