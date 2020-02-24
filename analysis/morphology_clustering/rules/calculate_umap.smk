# calculate UMAP

# calculate UMAP for specified k with default mindist
# Allows to screen different values of k to find a UMAP that looks good
rule calculate_UMAP_ks:
    input:
        table = "subsets/{subset}/normalised.csv",

    params:
        k = "{k}"

    output:
        fig = "subsets/{subset}/UMAP/default_mindist_k_{k}.png"

    log:
        "logs/{subset}_{k}_calc_umap_ks.log"

    script:
        "../scripts/calculate_umap_ks.py"

# calculate UMAP for chosen k & specified mindist
# Allows to screen different values for mindist to find a UMAP that looks good
rule calculate_UMAP_mindists:
    input:
        table = "subsets/{subset}/normalised.csv"

    log:
        "logs/{subset}_{mindist}_calc_umap_mindists.log"

    params:
        chosen_k = lambda wildcards: config['column_subsets'][wildcards.subset]["chosen_umap_k"],
        mindist = "{mindist}"

    output:
        fig = "subsets/{subset}/UMAP/mindist_{mindist}.png"

    script:
        "../scripts/calculate_umap_mindists.py"

# calculate final UMAP with chosen parameters
rule save_chosen_umap:
    input:
        table = "subsets/{subset}/normalised.csv"

    log:
        "logs/{subset}_save_chosen_umap.log"

    params:
        chosen_k = lambda wildcards: config['column_subsets'][wildcards.subset]["chosen_umap_k"],
        chosen_mindist = lambda wildcards: config['column_subsets'][wildcards.subset]["chosen_umap_mindist"]

    output:
        fig = "subsets/{subset}/UMAP/chosen_umap.png",
        umap_table = "subsets/{subset}/UMAP/umap.csv"

    script:
        "../scripts/save_chosen_umap.py"
