# calculate louvain partitioning

# Calculate KNN graph from table using specified k & distance metric - save as gpickle
rule calculate_KNN:
    input:
        table = "subsets/{subset}/normalised.csv"

    log:
        "logs/{subset}_{k}_calculate_KNN.log"

    params:
        k = "{k}"

    output:
        KNN = "subsets/{subset}/louvain/KNN_{k}.txt"

    script:
        "../scripts/calculate_louvain_KNN.py"

# Calculate louvain partitions from given KNN (with specified resolution)
# save as a pickled list where length == length of original input table, each entry is the partition that row
# is assigned to
rule calculate_louvain_partitions:
    input:
        KNN = "subsets/{subset}/louvain/KNN_{k}.txt"

    log:
        "logs/{subset}_{k}_{res}_calculate_partitions.log"

    params:
        res = "{res}"

    output:
        clusterings = "subsets/{subset}/louvain/louvain_{k}_res_{res}.txt"

    script:
        "../scripts/calculate_louvain_partitions.py"

# plot given clustering on umap
rule plot_all_clusters_on_umap:
    input:
        umap = "subsets/{subset}/UMAP/umap.csv",
        cluster = "subsets/{subset}/louvain/louvain_{k}_res_{res}.txt"

    log:
        "logs/{subset}_{k}_{res}_plot_all_clusters.log"

    params:
        k = "{k}",
        res = "{res}"

    output:
        fig = "subsets/{subset}/louvain/louvain_{k}_res_{res}.png"

    script:
        "../scripts/plot_clusters_on_umap.py"

def get_cluster_filename (subset):
    chosen_k = config['column_subsets'][subset]['chosen_louvain_k']
    chosen_res = config['column_subsets'][subset]['chosen_louvain_res']
    file_name = 'subsets/%s/louvain/louvain_%s_res_%s.txt' % (subset, chosen_k, chosen_res)
    return file_name

# Save chosen louvain clustering as a UMAP, a table with clusters & a table for the Explore Object Tables plugin
rule save_chosen_louvain:
    input:
        umap = "subsets/{subset}/UMAP/umap.csv",
        filtered = "tables/after_QC.csv",
        cluster_file = lambda wildcards: get_cluster_filename(wildcards.subset),
        viz_table_nuc = "viz_tables/after_QC_nuc_viz.csv",
        viz_table_cell = "viz_tables/after_QC_cell_viz.csv"
    log:
        "logs/{subset}_save_chosen_louvain.log"

    params:
        chosen_k = lambda wildcards: config['column_subsets'][wildcards.subset]["chosen_louvain_k"],
        chosen_res = lambda wildcards: config['column_subsets'][wildcards.subset]["chosen_louvain_res"]

    output:
        fig = "subsets/{subset}/louvain/chosen_louvain.png",
        merged_table = "subsets/{subset}/louvain/chosen_clusters.csv",
        viz_table_nuc = "subsets/{subset}/viz_tables/louvain_clusters_nuc.csv",
        viz_table_cell = "subsets/{subset}/viz_tables/louvain_clusters_cell.csv"

    script:
        "../scripts/save_chosen_louvain.py"
