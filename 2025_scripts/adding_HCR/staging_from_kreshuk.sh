#!/bin/bash

SRC_DIR="/g/kreshuk/buglakova/data/platy_registration/platybrowser-smfish-project/data/1.0.1/images/bdv-n5"
STAGING_BASE="/scratch/cros/platybrowser_staging"
COMBINED_STAGING="$STAGING_BASE/HCR_combined"
INDIVIDUAL_STAGING="$STAGING_BASE/HCR_individual"

# Create staging directories
mkdir -p "$COMBINED_STAGING" "$INDIVIDUAL_STAGING"

# Enable case-insensitive pattern matching for the _pl check
shopt -s nocasematch

for xml in "$SRC_DIR"/*.xml; do
    # Skip if the directory is empty and the glob fails
    [ -e "$xml" ] || continue 

    filename=$(basename "$xml")
    base="${filename%.xml}"

    # 1. Filter out segmentation and DAPI data
    if [[ "$filename" == *dapi* ]] || [[ "$filename" == *segm* ]]; then
        continue
    fi

    # 2. Dynamic Classification
    # If the filename contains "_pl" (case-insensitive), it's an individual replicate.
    # Otherwise, we assume it is a combined HCR file.
    if [[ "$filename" == *_pl* ]]; then
        DEST="$INDIVIDUAL_STAGING"
    else
        DEST="$COMBINED_STAGING"
    fi

    # 3. Symlink XML and its matching .n5 folder
    ln -sf "$xml" "$DEST/$filename"
    if [[ -d "$SRC_DIR/$base.n5" ]]; then
        ln -sf "$SRC_DIR/$base.n5" "$DEST/$base.n5"
    fi
done

# Turn off case-insensitive matching to keep the environment clean
shopt -u nocasematch

echo "Dynamic staging complete in $STAGING_BASE."