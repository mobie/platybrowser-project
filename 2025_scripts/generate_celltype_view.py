#!/usr/bin/env -S uv run
# /// script
# ///

"""Generate a MoBIE view for selected cell types or differentiation programmes.

Filters nuclei (or cells) by probability threshold for the given
programmes/cell types, then produces a MoBIE view JSON with preselected
segments, either printed to stdout (dry run), written to a JSON file, or
merged directly into dataset.json.

Examples:
    # Dry-run: nuclei with clade6sub19 at ≥0.8 probability
    ./generate_celltype_view.py -n "clade6sub19 view" -t clade6sub19 --threshold 0.8

    # Cells (translated from nuclei), 3D, glasbey LUT, write to a standalone JSON file
    ./generate_celltype_view.py -n "my 3d view" -t clade6sub19 --cells --3d --outfile my_view.json

    # Broad programmes, explicit colours, edit dataset.json directly
    ./generate_celltype_view.py -n "neurons + glia" \\
        -t Neurons -t Glia --threshold 0.5 \\
        --color Neurons:red --color Glia:blue --edit ../data/platybrowser_6dpf/dataset.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

TABLES_DIR = Path(__file__).resolve().parent.parent / "data" / "platybrowser_6dpf" / "tables"
NUCLEI_TABLES = TABLES_DIR / "sbem-6dpf-1-whole-segmented-nuclei"
CELL_TABLES = TABLES_DIR / "sbem-6dpf-1-whole-segmented-cells"
CELLS_TO_NUCLEI = CELL_TABLES / "cells_to_nuclei.tsv"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--name", required=True,
                   help="Name for the new MoBIE view")
    p.add_argument("-t", "--type", dest="types", action="append", required=True,
                   help="Cell type or programme name (repeatable). "
                        "E.g. -t clade6sub19 -t nocladesub20 for fine-grained, "
                        "or -t Neurons -t Glia for broad programmes.")
    p.add_argument("--threshold", type=float, default=0.8,
                   help="Minimum probability to include a nucleus (default: 0.8). "
                        "A label_id is included if its probability for ANY of the"
                        " requested types meets this threshold.")
    p.add_argument("--cells", action="store_true",
                   help="Select cells instead of nuclei. Translates nuclei -> cells "
                        "via cells_to_nuclei.tsv.")
    p.add_argument("--3d", dest="show3d", action="store_true",
                   help="Enable showSelectedSegmentsIn3d on the segmentation display. "
                        "Raw EM is NOT shown in 3D (visual noise).")
    p.add_argument("--group", default=None,
                   help="uiSelectionGroup for the view (default: auto-generated).")
    p.add_argument("--lut", choices=["glasbey", "argbColumn"], default="glasbey",
                   help="LUT for the segmentation display (default: glasbey).")
    p.add_argument("--color", dest="colors", action="append", default=[],
                   help="Per-type colour: TYPE:COLOR (e.g. clade6sub19:red). "
                        "Sets lut=argbColumn and adds a colour column to the "
                        "additional table.")
    p.add_argument("--opacity", type=float, default=0.5,
                   help="Segmentation opacity (default: 0.5).")
    p.add_argument("--outfile", default=None,
                   help="Write the view JSON to this file instead of stdout.")
    p.add_argument("--edit", default=None,
                   help="Path to dataset.json to edit in-place. "
                        "Adds the view and writes the updated file back.")
    p.add_argument("--dataset-json", default=None,
                   help="Path to dataset.json (for reading only, e.g. to check "
                        "existing view names).")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Print JSON to stdout (default if --outfile and --edit not given).")
    return p.parse_args()


def read_tsv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def detect_table(types):
    """Detect whether types are broad programmes or detailed cell types."""
    broad = read_tsv(NUCLEI_TABLES / "broad_types_cluster_probability.tsv")
    detailed = read_tsv(NUCLEI_TABLES / "detailed_cell_types_cluster_probability.tsv")

    broad_cols = set(broad[0].keys())
    detailed_cols = set(detailed[0].keys())

    # Check where the requested types exist
    broad_match = any(t in broad_cols for t in types)
    detailed_match = any(t in detailed_cols for t in types)

    if broad_match and not detailed_match:
        return broad, "broad"
    elif detailed_match and not broad_match:
        return detailed, "detailed"
    elif broad_match and detailed_match:
        # Ambiguous – prefer detailed (more specific)
        return detailed, "detailed"
    else:
        sys.exit(f"None of the requested types {types} found in either table. "
                 f"Broad columns: {sorted(b for b in broad_cols if b not in ('label_id','zero','autofluorescence','most probable cluster'))[:10]}... "
                 f"Detailed columns: {sorted(d for d in detailed_cols if d not in ('label_id','zero','autofluorescence','most probable cluster'))[:10]}...")


def clean_label(raw):
    """Strip .0 suffix from TSV label_ids."""
    return str(int(float(raw))) if "." in str(raw) else str(raw)


def filter_nuclei(rows, types, threshold):
    """Return set of label_ids (as int strings) that meet threshold for any requested type."""
    result = set()
    for row in rows:
        label = clean_label(row["label_id"])
        for t in types:
            if t in row:
                try:
                    if float(row[t]) >= threshold:
                        result.add(label)
                        break
                except ValueError:
                    continue
    return result


def nuclei_to_cells(nucleus_ids):
    """Translate nucleus label_ids to cell label_ids."""
    mapping = read_tsv(CELLS_TO_NUCLEI)
    reverse = defaultdict(list)
    for row in mapping:
        nid = clean_label(row["nucleus_id"])
        cid = clean_label(row["label_id"])
        if nid != "0":
            reverse[nid].append(cid)
    result = []
    for nid in nucleus_ids:
        result.extend(reverse.get(str(nid), []))
    return result


def build_selected_ids(label_ids, source_name):
    """Format label_ids as MoBIE selectedSegmentIds strings."""
    return [f"{source_name};0;{lid}" for lid in sorted(label_ids, key=lambda x: int(x))]


def build_view(name, source_name, selected_ids, args, colour_column="colour"):
    """Build the MoBIE view dictionary."""
    view = {
        "uiSelectionGroup": args.group or name,
        "sourceDisplays": [
            {
                "imageDisplay": {
                    "sources": ["raw"],
                    "color": "white",
                    "contrastLimits": [0.0, 255.0],
                    "showImagesIn3d": False,
                    "name": "raw",
                    "opacity": 1.0,
                    "visible": True,
                }
            },
            {
                "segmentationDisplay": {
                    "sources": [source_name],
                    "selectedSegmentIds": selected_ids,
                    "showSelectedSegmentsIn3d": args.show3d,
                    "lut": args.lut,
                    "showTable": True,
                    "name": source_name,
                    "opacity": args.opacity,
                    "visible": True,
                }
            },
        ],
        "sourceTransforms": [],
        "viewerTransform": {
            "normalizedAffine": [
                -0.0015131437810072582,
                0.0017925984711191075,
                0.00014519509454128732,
                -0.04377762461627315,
                -0.0006805596447375475,
                -0.0007463593890945693,
                0.002122234960696727,
                -0.09278589214610529,
                0.0016647310485757498,
                0.0013242482319377198,
                0.000999565981942194,
                -0.5494189842840294,
            ],
            "timepoint": 0,
        },
        "isExclusive": True,
        "description": "",
    }

    # Handle colour columns
    if args.lut == "argbColumn":
        view["sourceDisplays"][1]["segmentationDisplay"]["colorByColumn"] = colour_column

    # Handle additional tables if colours were specified
    if args.colors:
        prefix = "broad" if args.cells else "broad"  # Use whichever was the data source
        # We'll generate a per-row colour table inline, but for the viewer
        # we need to write it to a file. For dry-run, we just set up the structure.
        view["sourceDisplays"][1]["segmentationDisplay"]["additionalTables"] = [
            f"{name}_colours.tsv"
        ]

    return view


def generate_colour_table(ids_by_type, colour_map, source_name):
    """Generate a TSV with label_id and colour column for argbColumn LUT."""
    lines = [f"label_id\tcolour"]
    colour_names = {
        "red": "255-255-0-0",
        "green": "255-0-255-0",
        "blue": "255-0-0-255",
        "magenta": "255-255-0-255",
        "cyan": "255-0-255-255",
        "yellow": "255-255-255-0",
        "orange": "255-255-165-0",
        "white": "255-255-255-255",
    }
    for label in sorted(ids_by_type.keys()):
        typ = ids_by_type[label]
        color_name = colour_map.get(typ, "white")
        argb = colour_names.get(color_name, "255-255-255-255")
        lines.append(f"{label}\t{argb}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()

    # Detect and read the appropriate table
    table_rows, table_kind = detect_table(args.types)

    # Filter nuclei
    nucleus_ids = filter_nuclei(table_rows, args.types, args.threshold)
    if not nucleus_ids:
        sys.exit(f"No nuclei found with probability ≥ {args.threshold} for types: {args.types}")

    # Translate to cells if requested
    source_name = "nuclei"
    if args.cells:
        cell_ids = nuclei_to_cells(nucleus_ids)
        source_name = "cells"
        selected_ids = build_selected_ids(cell_ids, "cells")
        print(f"# {len(nucleus_ids)} nuclei → {len(cell_ids)} cells", file=sys.stderr)
    else:
        selected_ids = build_selected_ids(nucleus_ids, "nuclei")
        print(f"# {len(nucleus_ids)} nuclei selected", file=sys.stderr)

    # Build the view
    view = build_view(args.name, source_name, selected_ids, args)

    # Add a colour column to the view if per-type colours requested
    if args.colors:
        colour_map = {}
        for spec in args.colors:
            if ":" in spec:
                typ, col = spec.split(":", 1)
                colour_map[typ] = col
        view["sourceDisplays"][1]["segmentationDisplay"]["lut"] = "argbColumn"
        view["sourceDisplays"][1]["segmentationDisplay"]["colorByColumn"] = "colour"

    view_json = json.dumps(view, indent=2)

    # Output
    if args.edit:
        # Edit dataset.json in place
        ds_path = Path(args.edit)
        with open(ds_path) as f:
            dataset = json.load(f)
        if args.name in dataset.get("views", {}):
            sys.exit(f"View '{args.name}' already exists in {ds_path}")
        dataset.setdefault("views", {})[args.name] = view
        with open(ds_path, "w") as f:
            json.dump(dataset, f, indent=2)
            f.write("\n")
        print(f"Added view '{args.name}' to {ds_path} "
              f"({len(selected_ids)} {source_name})", file=sys.stderr)
    elif args.outfile:
        with open(args.outfile, "w") as f:
            f.write(view_json)
            f.write("\n")
        print(f"Wrote view to {args.outfile}", file=sys.stderr)
    else:
        print(view_json)


if __name__ == "__main__":
    main()
