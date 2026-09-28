#!/usr/bin/env -S uv run
# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "pandas"
# ]
# ///

import os
import pandas as pd
import hashlib

# --- Paths ---
PROJECT_ROOT = "../../"
BASE_DIR = os.path.join(PROJECT_ROOT, "data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-combined-traces")

PATH_BROAD = os.path.join(BASE_DIR, "broad_types_cluster_probability.tsv")
PATH_DETAILED = os.path.join(BASE_DIR, "detailed_cell_types_cluster_probability.tsv")
PATH_OUT = os.path.join(BASE_DIR, "assignment.tsv")

# --- Color Dictionary ---
BROAD_COLORS_HEX = {
    "Central nervous system": "#42D4F4",
    "Anterior nervous system": "#F032E6",
    "Developing neurons": "#911EB4",
    "Motile cilia": "#4363D8",
    "Enteric nervous system": "#800000",
    "Adult eye": "#3CB44B",
    "Gland cells": "#9A6324",
    "Glia": "#E6194B",
    "Somatic musculature": "#DCBEFF",
    "Cardiovascular system": "#FFE119",
    "Midgut": "#BFEF45",
    "Heme/chitin": "#F58231",
    "Epidermis": "#808000",
    "Macrophage": "#469990",
    "Stem cells": "#FABED4"
}

def hex_to_argb(hex_str):
    """Converts a standard #RRGGBB hex code to a MoBIE A-R-G-B string."""
    if pd.isna(hex_str) or not str(hex_str).startswith("#"):
        return "255-128-128-128" # Fallback gray
    
    hex_str = str(hex_str).strip().lstrip('#')
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return f"255-{r}-{g}-{b}"

def hash_string_to_argb(name):
    """Deterministically hashes a string into a random A-R-G-B color."""
    if pd.isna(name):
        return "0-0-0-0" # Transparent if completely missing
    
    name_str = str(name).strip()
    # Use MD5 to get a consistent pseudo-random byte hash
    h = hashlib.md5(name_str.encode('utf-8')).digest()
    
    # Take the first 3 bytes for R, G, and B
    return f"255-{h[0]}-{h[1]}-{h[2]}"

def get_cluster_col(df):
    """Safely extracts the cluster column handling space vs underscore naming variations."""
    if 'most_probable_cluster' in df.columns:
        return df['most_probable_cluster']
    elif 'most probable cluster' in df.columns:
        return df['most probable cluster']
    else:
        raise KeyError("Neither 'most_probable_cluster' nor 'most probable cluster' found in columns.")

def main():
    print(f"Reading broad types from: {PATH_BROAD}")
    df_broad = pd.read_csv(PATH_BROAD, sep='\t')
    
    print(f"Reading detailed subtypes from: {PATH_DETAILED}")
    df_detailed = pd.read_csv(PATH_DETAILED, sep='\t')

    # Safely extract the columns regardless of underscores or spaces
    try:
        df_broad['most_probable_broad_type'] = get_cluster_col(df_broad)
        df_detailed['most_probable_broad_subtype'] = get_cluster_col(df_detailed)
    except KeyError as e:
        print(f"\nERROR: {e}")
        return

    # Keep only the essential columns for merging
    df_broad = df_broad[['label_id', 'most_probable_broad_type']]
    df_detailed = df_detailed[['label_id', 'most_probable_broad_subtype']]

    print("Merging tables on label_id...")
    # Use an outer merge to ensure we don't drop cells that only appear in one table
    df_final = pd.merge(df_broad, df_detailed, on='label_id', how='outer')

    print("Assigning colors...")
    # 1. Broad Types
    mapped_hex = df_final['most_probable_broad_type'].map(BROAD_COLORS_HEX)
    df_final['broadTypeColourScheme'] = mapped_hex.apply(hex_to_argb)

    # 2. Subtypes
    df_final['subtypeColourScheme'] = df_final['most_probable_broad_subtype'].apply(hash_string_to_argb)

    # Ensure label_id is cast properly 
    df_final['label_id'] = df_final['label_id'].fillna(0).astype(int)
    
    # Drop rows where label_id is 0 
    df_final = df_final[df_final['label_id'] > 0]

    # Reorder columns to exactly match your requirement
    cols = [
        'label_id', 
        'most_probable_broad_type', 
        'most_probable_broad_subtype', 
        'broadTypeColourScheme', 
        'subtypeColourScheme'
    ]
    df_final = df_final[cols]

    print(f"Saving {len(df_final)} rows to {PATH_OUT}")
    df_final.to_csv(PATH_OUT, sep='\t', index=False)
    print("Done!")

if __name__ == "__main__":
    main()