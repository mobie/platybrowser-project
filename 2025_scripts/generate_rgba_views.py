#!/usr/bin/env -S uv run
# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Generate RGBA-coloured MoBIE views and 2x2 comparison PDFs.

For each group of cell types:
  1. Cluster types by Pearson correlation of probability vectors (thresholded nuclei)
  2. Assign each type an RGBA colour from a perceptual colormap in cluster order
  3. Write a TSV table mapping each nucleus to the colour of its argmax type
  4. Add an argbColumn view to dataset.json
  5. Produce a 2x2 PDF: threshold sweep (top row) + clustered correlation with
     coloured type labels (bottom row), per group.

Usage:
    ./generate_rgba_views.py
"""

import csv
import json
import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

DETAILED = Path(__file__).resolve().parent.parent / "data" / "platybrowser_6dpf" / "tables" / \
    "sbem-6dpf-1-whole-segmented-nuclei" / "detailed_cell_types_cluster_probability.tsv"
NUCLEI_TABLES = Path(__file__).resolve().parent.parent / "data" / "platybrowser_6dpf" / "tables" / \
    "sbem-6dpf-1-whole-segmented-nuclei"
DS = Path(__file__).resolve().parent.parent / "data" / "platybrowser_6dpf" / "dataset.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "tmp_plots"

GROUPS = OrderedDict({
    "midgut_endoderm": {
        "types": [
            "clade1sub2", "clade1sub3", "clade1sub4", "clade1sub7",
            "clade1sub11", "clade1sub12", "clade1sub13", "clade1sub14",
            "clade1sub16", "clade1sub17", "clade1sub18", "clade1sub19",
        ],
        "threshold": 0.8,
        "view_name": "midgut/endoderm RGBA",
    },
    "first_view": {
        "types": [
            "clade6sub19", "nocladesub20", "nocladesub26",
            "nocladesub12", "nocladesub10", "nocladesub2", "nocladesub22",
        ],
        "threshold": 0.9,
        "view_name": "first_view RGBA",
    },
})

UI_GROUP = "test-new-views-Detlev-2026"
VIEWER_XF = {
    "normalizedAffine": [
        -0.0015131437810072582, 0.0017925984711191075,
        0.00014519509454128732, -0.04377762461627315,
        -0.0006805596447375475, -0.0007463593890945693,
        0.002122234960696727, -0.09278589214610529,
        0.0016647310485757498, 0.0013242482319377198,
        0.000999565981942194, -0.5494189842840294,
    ],
    "timepoint": 0,
}


def read_data():
    with open(DETAILED) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_matrix(rows, types):
    data = np.zeros((len(rows), len(types)))
    for i, row in enumerate(rows):
        for j, t in enumerate(types):
            data[i, j] = float(row.get(t, 0))
    return data


def cluster_order(corr):
    """Return leaf order from hierarchical clustering of correlation matrix."""
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    return leaves_list(Z)


def assign_colours(types, order):
    """Assign RGBA colours to types in cluster order using turbo colormap."""
    n = len(types)
    # Sample n evenly-spaced points from turbo (excludes dark low end)
    cmap = plt.cm.turbo
    samples = np.linspace(0.15, 1.0, n)  # skip darkest region
    colours = {}
    for i, idx in enumerate(order):
        r, g, b, _ = cmap(samples[i])
        # MoBIE argbColumn format: alpha-red-green-blue
        colours[types[idx]] = f"{int(r*255)}-{int(g*255)}-{int(b*255)}-255"
    return colours


def write_rgba_table(rows, types, colours, out_path):
    """Write TSV with label_id and colour column (argbColumn format)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["label_id", "colour"])
        for row in rows:
            lid = str(int(float(row["label_id"]))) if "." in row["label_id"] else row["label_id"]
            # Find argmax type
            best_type = max(types, key=lambda t: float(row.get(t, 0)))
            writer.writerow([lid, colours[best_type]])
    return out_path


def add_view(dataset, view_name, table_name, types, threshold):
    """Add an argbColumn view to the dataset dict."""
    view = {
        "uiSelectionGroup": UI_GROUP,
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
                    "sources": ["nuclei"],
                    "lut": "argbColumn",
                    "colorByColumn": "colour",
                    "additionalTables": [table_name],
                    "showTable": True,
                    "name": "nuclei",
                    "opacity": 0.5,
                    "visible": True,
                }
            },
        ],
        "sourceTransforms": [],
        "viewerTransform": VIEWER_XF,
        "isExclusive": True,
        "description": "",
    }
    dataset["views"][view_name] = view


def make_pdf(group_key, group_info, types, threshold, colours, order):
    """Generate the 2x2 PDF for one group."""
    rows_all = read_data()
    data = build_matrix(rows_all, types)
    max_prob = data.max(axis=1)

    mask = max_prob >= threshold
    data_sel = data[mask]

    # --- Figure: 2 rows × 2 cols ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Row 1: threshold sweep (spans full width for readability)
    # Actually the user said 2x2: rows = groups, cols = raw proba vs colored
    # But we're doing one PDF per group, so 2x2 per group:
    # Top-left: threshold sweep
    # Top-right: viridis probability heatmap (all nuclei)
    # Bottom-left: clustered correlation heatmap
    # Bottom-right: legend of colours

    # Top-left: threshold sweep
    thresholds_arr = np.arange(0, 1.01, 0.05)
    counts = [(max_prob >= t).sum() for t in thresholds_arr]
    ax1.plot(thresholds_arr, counts, "o-", color="steelblue", markersize=3, linewidth=1.5)
    ymax = counts[np.where(thresholds_arr == 0.2)[0][0]]
    ax1.set_ylim(0, ymax * 1.05)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Nuclei retained")
    ax1.set_title(f"{group_key}: threshold vs. nuclei retained\n"
                  f"({len(data_sel)} nuclei at ≥{threshold})")
    ax1.grid(True, alpha=0.3)

    # Top-right: viridis heatmap (max probability per nucleus above threshold)
    n_cells = min(200, len(data_sel))
    if len(data_sel) > n_cells:
        idx = np.random.RandomState(42).choice(len(data_sel), n_cells, replace=False)
        display_data = data_sel[idx]
    else:
        display_data = data_sel

    # Show max probability as a row color strip
    ax2.imshow(display_data.max(axis=1, keepdims=True).T, aspect="auto",
               cmap="viridis", vmin=0, vmax=1)
    ax2.set_yticks([])
    ax2.set_xlabel("Nuclei (sampled)")
    ax2.set_title("Max probability per nucleus (viridis)")

    # Bottom-left: clustered correlation heatmap
    corr = np.corrcoef(data_sel.T)
    corr = corr[order][:, order]
    short_names = [t if len(t) <= 15 else t[:7] + "…" + t[-7:] for t in types]
    short_names = [short_names[i] for i in order]

    im = ax3.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(short_names)))
    ax3.set_yticks(range(len(short_names)))
    # Color the tick labels with the assigned colours
    type_colours = [colours[types[i]] for i in order]
    for i, (name, color_str) in enumerate(zip(short_names, type_colours)):
        r, g, b = [int(x) / 255 for x in color_str.split("-")[1:]]
        ax3.get_xticklabels()[i].set_color((r, g, b))
        ax3.get_yticklabels()[i].set_color((r, g, b))
    ax3.set_xticklabels(short_names, rotation=90, fontsize=7)
    ax3.set_yticklabels(short_names, fontsize=7)
    ax3.set_title(f"{group_key}: clustered Pearson r\n"
                  f"({len(data_sel)} nuclei ≥ {threshold})")
    plt.colorbar(im, ax=ax3, shrink=0.8, label="Pearson r")

    # Bottom-right: small color legend
    ax4.axis("off")
    ax4.set_title("Assigned colours")
    for i, idx in enumerate(order):
        t = types[idx]
        color_str = colours[t]
        r, g, b = [int(x) / 255 for x in color_str.split("-")[1:]]
        short = t if len(t) <= 20 else t[:10] + "…" + t[-9:]
        ax4.add_patch(plt.Rectangle((0.05, 0.92 - i * (0.85 / len(order))),
                                     0.15, 0.06, color=(r, g, b), transform=ax4.transAxes))
        ax4.text(0.23, 0.95 - i * (0.85 / len(order)), short,
                 transform=ax4.transAxes, fontsize=7, va="center")

    fig.tight_layout()
    pdf_path = OUT_DIR / f"{group_key}_rgba.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}", file=sys.stderr)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_data()
    print(f"Loaded {len(rows)} nuclei", file=sys.stderr)

    with open(DS) as f:
        dataset = json.load(f)

    for group_key, info in GROUPS.items():
        types = info["types"]
        threshold = info["threshold"]
        view_name = info["view_name"]

        # Build matrix and filter
        data = build_matrix(rows, types)
        max_prob = data.max(axis=1)
        mask = max_prob >= threshold
        data_sel = data[mask]

        # Correlation and clustering
        corr = np.corrcoef(data_sel.T)
        order = cluster_order(corr)

        # Assign colours
        colours = assign_colours(types, order)

        # Write RGBA TSV table
        table_name = f"{group_key}_rgba.tsv"
        out_tsv = NUCLEI_TABLES / table_name
        write_rgba_table(rows, types, colours, out_tsv)
        print(f"Wrote {out_tsv} ({len(types)} types, {len(colours)} colours)",
              file=sys.stderr)

        # Add view to dataset
        add_view(dataset, view_name, table_name, types, threshold)
        print(f"Added view '{view_name}'", file=sys.stderr)

        # Generate 2x2 PDF
        make_pdf(group_key, info, types, threshold, colours, order)

        # Show colour assignments
        for i in order:
            t = types[i]
            print(f"  {t} -> {colours[t]}", file=sys.stderr)
        print(file=sys.stderr)

    # Write dataset.json
    with open(DS, "w") as f:
        json.dump(dataset, f, indent=2)
        f.write("\n")
    print(f"Updated {DS}", file=sys.stderr)


if __name__ == "__main__":
    main()
