#!/usr/bin/env -S uv run
# /// script
# dependencies = ["matplotlib", "numpy", "scipy"]
# ///

"""Plot threshold sensitivity and cell type correlations for a MoBIE view.

Reads the detailed_cell_types_cluster_probability.tsv table and produces a PDF
with two panels:
  1. Number of nuclei retained as the probability threshold varies from 0 to 1
  2. Pairwise Pearson correlation between the selected cell types

Usage:
    ./plot_threshold_analysis.py -n "midgut/endoderm" \
        clade1sub2 clade1sub3 clade1sub4 clade1sub7 \
        clade1sub11 clade1sub12 clade1sub13 clade1sub14 \
        clade1sub16 clade1sub17 clade1sub18 clade1sub19

    ./plot_threshold_analysis.py -n "first_view" \
        clade6sub19 nocladesub20 nocladesub26 nocladesub12 \
        nocladesub10 nocladesub2 nocladesub22
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

DETAILED = Path(__file__).resolve().parent.parent / "data" / "platybrowser_6dpf" / "tables" / \
    "sbem-6dpf-1-whole-segmented-nuclei" / "detailed_cell_types_cluster_probability.tsv"
OUT_DIR = Path(__file__).resolve().parent.parent / "tmp_plots"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--name", required=True,
                   help="View name (used for output PDF filename)")
    p.add_argument("types", nargs="+",
                    help="Cell type columns to analyse (from detailed TSV)")
    p.add_argument("-o", "--outdir", default=str(OUT_DIR),
                   help=f"Output directory (default: {OUT_DIR})")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Correlation computed only on nuclei with max probability "
                        "≥ threshold (default: 0 = all nuclei). "
                        "Use the same threshold as the view.")
    return p.parse_args()


def read_data():
    with open(DETAILED) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def clean_label(raw):
    return str(int(float(raw))) if "." in str(raw) else str(raw)


def build_matrix(rows, types):
    """Build an n_nuclei × n_types matrix of probabilities."""
    data = np.zeros((len(rows), len(types)))
    for i, row in enumerate(rows):
        for j, t in enumerate(types):
            data[i, j] = float(row.get(t, 0))
    return data


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_data()
    print(f"Loaded {len(rows)} nuclei with {len(rows[0])} columns", file=sys.stderr)

    # Check types exist
    available = set(rows[0].keys())
    missing = [t for t in args.types if t not in available]
    if missing:
        sys.exit(f"Types not found in table: {missing}")

    data = build_matrix(rows, args.types)
    max_prob = data.max(axis=1)  # max probability per nucleus

    # --- Figure ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])    # threshold sweep (top, full width)
    ax2 = fig.add_subplot(gs[1, 0])    # correlation heatmap
    ax3 = fig.add_subplot(gs[1, 1])    # colour legend

    # Panel 1: threshold sweep
    thresholds = np.arange(0, 1.01, 0.05)
    counts = [(max_prob >= t).sum() for t in thresholds]

    ax1.plot(thresholds, counts, "o-", color="steelblue", markersize=4, linewidth=1.5)
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Nuclei retained")
    ax1.set_title(f"{args.name}: threshold vs. nuclei retained")
    ax1.grid(True, alpha=0.3)
    # Limit y-axis to the count at threshold 0.2 (meaningful range)
    ymax = counts[np.where(thresholds == 0.2)[0][0]]
    ax1.set_ylim(0, ymax * 1.05)
    # Annotate the 0.5 crossing
    idx50 = np.searchsorted(thresholds, 0.5) - 1
    ax1.axhline(counts[idx50], color="gray", linestyle="--", alpha=0.4)
    ax1.axvline(0.5, color="gray", linestyle="--", alpha=0.4)
    ax1.annotate(f"n={counts[idx50]} at 0.5", (0.52, counts[idx50]),
                 fontsize=8, color="gray")

    # Panel 2: correlation heatmap (only on nuclei passing threshold)
    if args.threshold > 0:
        mask = max_prob >= args.threshold
        data_corr = data[mask]
        n_corr = data_corr.shape[0]
        title_note = f"Pearson r across {n_corr} nuclei with max(P) ≥ {args.threshold}"
    else:
        data_corr = data
        n_corr = data_corr.shape[0]
        title_note = f"Pearson r across all {n_corr} nuclei"

    corr = np.corrcoef(data_corr.T)
    # Cluster by hierarchical clustering (1 - r as distance)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    corr = corr[order][:, order]

    short_names_all = [t if len(t) <= 15 else t[:7] + "…" + t[-7:] for t in args.types]
    short_names = [short_names_all[i] for i in order]

    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(short_names)))
    ax2.set_yticks(range(len(short_names)))
    ax2.set_xticklabels(short_names, rotation=90, fontsize=7, color="black")
    ax2.set_yticklabels(short_names, fontsize=7, color="black")
    ax2.set_title(f"{args.name}: cell type correlations\n{title_note}", fontsize=10)
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8, label="Pearson r")
    cbar.ax.tick_params(labelsize=7)

    # Colour legend in third panel
    ax3.axis("off")
    ax3.set_title("Colour key", fontsize=9)
    cmap = plt.cm.turbo
    n = len(order)
    samples = np.linspace(0.15, 1.0, n)
    for i in range(n):
        r, g, b, _ = cmap(samples[i])
        y = 0.93 - i * (0.85 / n)
        ax3.add_patch(plt.Rectangle((0.1, y), 0.15, 0.06, color=(r, g, b),
                                     transform=ax3.transAxes))
        label = short_names[i]
        ax3.text(0.28, y + 0.03, label, transform=ax3.transAxes,
                 fontsize=6.5, va="center")

    fig.tight_layout()

    safe_name = args.name.replace("/", "_").replace(" ", "_")
    pdf_path = outdir / f"{safe_name}.pdf"
    fig.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
