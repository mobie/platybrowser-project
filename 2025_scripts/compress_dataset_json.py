#!/usr/bin/env python3
"""Compress dataset.json by removing viewer-default-valued keys.

Removes keys whose values equal the MoBIE viewer's Java defaults, per
AGENTS.md "Writing concise views (omit viewer defaults)". Idempotent:
running it on an already-compressed file is a no-op.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_PATH = Path("data/platybrowser_6dpf/dataset.json")

IMAGE_DISPLAY_DEFAULTS = {
    "opacity": 1.0,
    "visible": True,
    "invert": False,
    "showImagesIn3d": False,
    "blendingMode": "sum",
}

ANNOTATION_DISPLAY_DEFAULTS = {
    "visible": True,
    "opacity": 0.5,
    "showTable": True,
    "lut": "glasbey",
    "showAsBoundaries": False,
    "boundaryThickness": 1.0,
    "showScatterPlot": False,
    "scatterPlotAxes": ["anchor_x", "anchor_y"],
    "showSelectedSegmentsIn3d": False,
    "randomColorSeed": 42,
    "opacityNotSelected": 0.15,
    "blendingMode": "alpha",
}

WHITE_COLORS = ("white", "r=255,g=255,b=255,a=255")

DISPLAY_TYPES = (
    "imageDisplay",
    "segmentationDisplay",
    "spotDisplay",
    "regionDisplay",
)


def count_keys(obj):
    """Count every key in nested dicts (used for the removal summary)."""
    if isinstance(obj, dict):
        return len(obj) + sum(count_keys(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_keys(v) for v in obj)
    return 0


def _safe_unlink(path):
    """Best-effort temp-file cleanup; never raises."""
    if path is None:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


def _strip_defaults(display, defaults):
    for key, default in defaults.items():
        if display.get(key) == default:
            del display[key]
    # NB: `name` is NEVER stripped. It is the UI panel label shown by the
    # viewer (UserInterfaceHelper#createDisplayPanel renders getName() with
    # no fallback), and the display classes have no default for it — absent
    # means null, so views lose their cells/nuclei/traces/marker labels.


def compress_display(display_type, display):
    """Return a copy of a display dict with default-valued keys removed."""
    display = dict(display)
    if display_type == "imageDisplay":
        _strip_defaults(display, IMAGE_DISPLAY_DEFAULTS)
        if display.get("color") in WHITE_COLORS:
            del display["color"]
    else:
        _strip_defaults(display, ANNOTATION_DISPLAY_DEFAULTS)
    return display


def compress_view(view):
    """Return a copy of a view dict with default-valued keys removed."""
    view = dict(view)
    if view.get("isExclusive") is False:
        del view["isExclusive"]
    if view.get("sourceTransforms") == []:
        del view["sourceTransforms"]
    vt = view.get("viewerTransform")
    if isinstance(vt, dict) and vt.get("timepoint") == 0:
        vt = dict(vt)
        del vt["timepoint"]
        view["viewerTransform"] = vt
    sds = view.get("sourceDisplays")
    if isinstance(sds, list):
        new_sds = []
        for sd in sds:
            sd = dict(sd)
            for display_type in DISPLAY_TYPES:
                if display_type in sd:
                    sd[display_type] = compress_display(display_type, sd[display_type])
            new_sds.append(sd)
        view["sourceDisplays"] = new_sds
    return view


def compress_dataset(data):
    """Return a copy of the dataset with every view compressed."""
    data = dict(data)
    views = data.get("views")
    if isinstance(views, dict):
        data["views"] = {name: compress_view(v) for name, v in views.items()}
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compress dataset.json by removing viewer-default-valued keys."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="path to dataset.json (default: %(default)s)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the file would change; write nothing",
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="git add the file if it changed (for pre-commit auto-stage)",
    )
    args = parser.parse_args(argv)

    path = args.path
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        return 1
    try:
        with open(path) as f:
            before = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 1

    after = compress_dataset(before)
    if after == before:
        print(f"{path}: already compressed")
        return 0

    removed = count_keys(before) - count_keys(after)
    if args.check:
        print(
            f"{path}: not compressed ({removed} redundant keys found)",
            file=sys.stderr,
        )
        return 1

    # Atomic write: temp file in the same directory, then replace.
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(after, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
        if args.stage:
            subprocess.run(["git", "add", str(path)], check=True)
    except OSError as e:
        _safe_unlink(tmp)
        print(f"error: could not write {path}: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        _safe_unlink(tmp)
        print(f"error: git add failed for {path}: {e}", file=sys.stderr)
        return 1

    print(f"{path}: compressed (removed {removed} redundant keys)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
