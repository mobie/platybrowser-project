#!/bin/bash

# Define relative paths from 2025_scripts/work_on_traces to the tables directory
NEW_DIR="../../data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-combined-segmentation"
OLD_DIR="../../data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei"

echo "Processing broad_types_cluster_probability.tsv..."
awk -F'\t' -v OFS='\t' '
    # 1. Read the new default.tsv file first to collect active integer IDs
    NR==FNR { 
        if (FNR > 1) valid[$1] = 1
        next 
    }
    
    # 2. Print the header row of the cluster probability file exactly as-is
    FNR==1 { 
        print $0 
        next 
    }
    
    # 3. Clean floating IDs (e.g., 1.0 -> 1) and filter matches
    {
        clean_id = $1
        sub(/\.0$/, "", clean_id)
        if (clean_id in valid) {
            $1 = clean_id
            print $0
        }
    }
' "$NEW_DIR/default.tsv" "$OLD_DIR/broad_types_cluster_probability.tsv" > "$NEW_DIR/broad_types_cluster_probability.tsv"


echo "Processing detailed_cell_types_cluster_probability.tsv..."
awk -F'\t' -v OFS='\t' '
    # 1. Read active integer IDs from default.tsv
    NR==FNR { 
        if (FNR > 1) valid[$1] = 1
        next 
    }
    
    # 2. Print the header row verbatim
    FNR==1 { 
        print $0 
        next 
    }
    
    # 3. Match clean keys
    {
        clean_id = $1
        sub(/\.0$/, "", clean_id)
        if (clean_id in valid) {
            $1 = clean_id
            print $0
        }
    }
' "$NEW_DIR/default.tsv" "$OLD_DIR/detailed_cell_types_cluster_probability.tsv" > "$NEW_DIR/detailed_cell_types_cluster_probability.tsv"

echo "All cluster tables filtered, formatted, and written successfully to $NEW_DIR!"