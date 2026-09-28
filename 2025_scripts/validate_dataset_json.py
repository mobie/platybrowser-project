#!/usr/bin/env python3
"""Structural validation for data/platybrowser_6dpf/dataset.json.

Checks that the file parses as JSON, has the required top-level keys, the
default view exists, and that every source referenced by views (in
sourceDisplays and selectedSegmentIds) exists in the sources section.
"""

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("data/platybrowser_6dpf/dataset.json")

REQUIRED_TOP_LEVEL = ("is2D", "defaultLocation", "sources", "views")
DISPLAY_TYPES = (
    "imageDisplay",
    "segmentationDisplay",
    "spotDisplay",
    "regionDisplay",
)


def validate(data):
    """Return a list of error strings; empty list means the dataset is valid."""
    errors = []
    if not isinstance(data, dict):
        return ["dataset.json root must be a JSON object"]

    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            errors.append(f"missing top-level key: {key}")

    sources = data.get("sources", {})
    if not isinstance(sources, dict):
        errors.append("top-level key 'sources' must be an object")
        sources = {}

    views = data.get("views", {})
    if not isinstance(views, dict):
        errors.append("top-level key 'views' must be an object")
        views = {}

    if "default" not in views:
        errors.append("missing required view: default")

    for name, view in views.items():
        if not isinstance(view, dict):
            errors.append(f"view {name}: must be an object")
            continue
        sds = view.get("sourceDisplays")
        if "sourceDisplays" in view and not isinstance(sds, list):
            errors.append(f"view {name}: sourceDisplays must be a list")
            sds = []
        for sd in sds or []:
            if not isinstance(sd, dict):
                errors.append(f"view {name}: sourceDisplay must be an object")
                continue
            present = [t for t in DISPLAY_TYPES if t in sd]
            if len(present) != 1:
                errors.append(
                    f"view {name}: sourceDisplay must have exactly one of "
                    f"{', '.join(DISPLAY_TYPES)}, found {present or 'none'}"
                )
                continue
            display = sd[present[0]]
            if not isinstance(display, dict):
                errors.append(f"view {name}: {present[0]} must be an object")
                continue
            srcs = display.get("sources")
            if "sources" in display and not isinstance(srcs, list):
                errors.append(f"view {name}: sources must be a list, got {srcs!r}")
                srcs = []
            for src in srcs or []:
                if src not in sources:
                    errors.append(f"view {name}: unknown source: {src}")
            selected = display.get("selectedSegmentIds")
            if "selectedSegmentIds" in display and not isinstance(selected, list):
                errors.append(
                    f"view {name}: selectedSegmentIds must be a list, got {selected!r}"
                )
                selected = []
            for entry in selected or []:
                parts = str(entry).split(";")
                if len(parts) != 3:
                    errors.append(
                        f"view {name}: bad selectedSegmentIds entry {entry!r} "
                        f"(expected source;timepoint;id)"
                    )
                elif parts[0] not in sources:
                    errors.append(
                        f"view {name}: selectedSegmentIds references unknown source: {parts[0]}"
                    )
            cl = display.get("contrastLimits")
            if cl is not None and (not isinstance(cl, list) or len(cl) != 2):
                errors.append(
                    f"view {name}: contrastLimits must be a 2-element list, got {cl!r}"
                )

    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate dataset.json structure."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="path to dataset.json (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    path = args.path
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 1

    errors = validate(data)
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"{path}: INVALID ({len(errors)} error(s))", file=sys.stderr)
        return 1
    print(f"{path}: valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
