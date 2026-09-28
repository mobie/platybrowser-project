#!/usr/bin/env -S uv run
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Add per-probability-source views to dataset.json.

Reads the score columns from a cluster probability table (or takes an explicit
--subtypes list) and adds one image source ({name}_proba, bdv.n5 + bdv.n5.s3)
and one additive view per score column, under a configurable uiSelectionGroup
(--group, default nuclei_probabilities).

Views are named WITHOUT the _proba suffix (view key and display name = {name},
source = {name}_proba) and the group's views are kept naturally sorted so the
MoBIE dropdown is ordered. Idempotent: existing names are left untouched.
Dry-run by default; use --write to apply.

Usage:
    ./add_proba_sources_and_views.py --dataset-json <dataset.json> --table <tsv> \
        [--group coregulon_probabilities] [--s3-prefix images/bdv-n5-s3/coregulon_proba] \
        [--subtypes a,b,c] [--write]
"""

import argparse
import json
import re
import sys
from pathlib import Path

from generate_nuclei_proba_images import read_subtype_columns, safe_name

DEFAULT_UI_GROUP = "nuclei_probabilities"
DEFAULT_S3_PREFIX = "images/bdv-n5-s3/celltype_proba"


def natural_key(s):
    return tuple(
        (0, int(p)) if p.isdigit() else (1, p)
        for p in re.split(r"(\d+)", s) if p
    )


def source_definition(name, s3_prefix):
    return {
        "image": {
            "imageData": {
                "bdv.n5": {"relativePath": f"images/local/{name}.xml"},
                "bdv.n5.s3": {"relativePath": f"{s3_prefix}/{name}.xml"},
            }
        }
    }


def view_definition(subtype, ui_group):
    return {
        "uiSelectionGroup": ui_group,
        "sourceDisplays": [{
            "imageDisplay": {
                "sources": [f"{safe_name(subtype)}_proba"],
                "contrastLimits": [0.0, 1000.0],
                "name": subtype,
            }
        }],
    }


def reorder_group(views, ui_group):
    """Keep the group's views contiguous and naturally sorted (dropdown order)."""
    group = {k: v for k, v in views.items() if v.get("uiSelectionGroup") == ui_group}
    if not group:
        return
    for k in group:
        del views[k]
    for k in sorted(group, key=natural_key):
        views[k] = group[k]


def add_sources_and_views(dataset, subtypes, with_views=True,
                          ui_group=DEFAULT_UI_GROUP, s3_prefix=DEFAULT_S3_PREFIX):
    sources = dataset.setdefault("sources", {})
    views = dataset.setdefault("views", {})
    for subtype in subtypes:
        sname = f"{safe_name(subtype)}_proba"
        if sname not in sources:
            sources[sname] = source_definition(sname, s3_prefix)
        if with_views and subtype not in views:
            views[subtype] = view_definition(subtype, ui_group)
    if with_views:
        reorder_group(views, ui_group)
    return dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-json", required=True,
                   help="Path to data/platybrowser_6dpf/dataset.json")
    p.add_argument("--table", required=True,
                   help="Path to detailed_cell_types_cluster_probability.tsv")
    p.add_argument("--subtypes", default=None,
                   help="Comma-separated subtypes (default: all score columns from the table)")
    p.add_argument("--group", default=DEFAULT_UI_GROUP,
                   help=f"uiSelectionGroup for the views (default: {DEFAULT_UI_GROUP})")
    p.add_argument("--s3-prefix", default=DEFAULT_S3_PREFIX,
                   help=f"S3 prefix for the source XMLs (default: {DEFAULT_S3_PREFIX})")
    p.add_argument("--no-views", action="store_true",
                   help="Add only the sources, skip the views/dropdown items")
    p.add_argument("--write", action="store_true",
                   help="Apply changes to dataset.json (default: dry-run)")
    return p.parse_args()


def main():
    args = parse_args()
    subtypes = args.subtypes.split(",") if args.subtypes else read_subtype_columns(args.table)
    if not subtypes:
        sys.exit("No subtypes selected")
    ds_path = Path(args.dataset_json)
    with open(ds_path, encoding="utf-8") as f:
        dataset = json.load(f)
    before_sources = set(dataset.get("sources", {}))
    before_views = set(dataset.get("views", {}))
    add_sources_and_views(dataset, subtypes, with_views=not args.no_views,
                          ui_group=args.group, s3_prefix=args.s3_prefix)
    new_sources = set(dataset["sources"]) - before_sources
    new_views = set(dataset["views"]) - before_views
    if args.no_views:
        print(f"{len(new_sources)} new sources (views skipped: --no-views)",
              file=sys.stderr)
    else:
        print(f"{len(new_sources)} new sources, {len(new_views)} new views "
              f"(group '{args.group}')", file=sys.stderr)
    if args.write:
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
            f.write("\n")
        print(f"Wrote {ds_path}", file=sys.stderr)
    else:
        print("Dry run: no changes written. Use --write to apply.", file=sys.stderr)


if __name__ == "__main__":
    main()
