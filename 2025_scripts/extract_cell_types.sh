#!/bin/bash

# This script takes a bunch of manually saved Platybrowser JSON views.
# It merges them and returns a JSON, and extracts labelled cells into a TSV table.

set -e

################ Parsing

JSON_OUT=""
TSV_OUT=""
INPUT_FILES=()

usage() {
    echo "Usage: $0 -j <json_file> -t <tsv_file> [input_file1.json ...]"
    echo "       $0 --json_out <json_file> --tsv_out <tsv_file> [input_file1.json ...]"
    echo ""
    echo "Merges 'views' from multiple JSON files and extracts cell segmentation data to a TSV."
    echo ""
    echo "Options:"
    echo "  -j, --json_out <file>    Required. Path for the merged JSON output file."
    echo "  -t, --tsv_out <file>     Required. Path for the TSV output file."
    echo "  -h, --help               Display this help message and exit."
    exit 1
}

PARSED_ARGS=$(getopt -o j:t:h --long json_out:,tsv_out:,help --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    exit 2
fi

eval set -- "$PARSED_ARGS"

while true; do
    case "$1" in
        -j|--json_out)
            JSON_OUT="$2"
            shift 2
            ;;
        -t|--tsv_out)
            TSV_OUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift 
            break
            ;;
        *)
            echo "Internal error!" >&2
            exit 3
            ;;
    esac
done

for arg in "$@"; do
    INPUT_FILES+=("$arg")
done


if [[ -z "$JSON_OUT" ]] || [[ -z "$TSV_OUT" ]]; then
    echo "Error: --json_out and --tsv_out arguments are required." >&2
    usage
fi

if [ ${#INPUT_FILES[@]} -eq 0 ]; then
    echo "Error: At least one input JSON file is required." >&2
    usage
fi

################ Modifying JSON files in-place before processing

# Sometimes Detlev did not set the right options

echo "Preprocessing ${#INPUT_FILES[@]} input files to ensure correct settings..."
for file in "${INPUT_FILES[@]}"; do
    sed -i 's/"isExclusive": false/"isExclusive": true/g' "$file"
    sed -i 's/"sbem-segmentation"/"curated-cell-types"/g' "$file"
done
echo "Preprocessing complete."


################ Merging

echo "Merging ${#INPUT_FILES[@]} input files into $JSON_OUT..."
jq -s 'reduce .[] as $item ({views: {}}; .views += $item.views)' "${INPUT_FILES[@]}" > "$JSON_OUT"
echo "Merge complete."

################ Extracting into TSV of "label_id" to "curated_cell_type"

echo "Extracting cell data from $JSON_OUT into $TSV_OUT..."
jq -r '
    ["label_id", "curated_cell_type"],
    (
        .views | to_entries[] |
        . as $view_entry |
        $view_entry.value.sourceDisplays[]? |
        select(.segmentationDisplay.selectedSegmentIds) |
        .segmentationDisplay.selectedSegmentIds[] |
        [ (split(";")[-1]), $view_entry.key ]
    )
| @tsv
' "$JSON_OUT" | \
(head -n 1 && tail -n +2 | sort -n -k1,1) > "$TSV_OUT"

echo "Extraction complete. TSV saved to $TSV_OUT."
echo "Done."