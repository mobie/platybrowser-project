# AGENTS.md — PlatyBrowser Project

## Project overview

This is a fork of the [PlatyBrowser](https://github.com/mobie/platybrowser-project) MoBIE project — a resource for exploring a full EM volume of a 6-day-old *Platynereis* larva combined with a gene expression atlas and tissue, cellular, and ultra-structure segmentations.

The upstream is `mobie/platybrowser-project`. This fork currently uses the GitHub repo URL `https://github.com/cyrilcros/platybrowser-project-2025` as the MoBIE project entry point. Current work focuses on the `platybrowser_6dpf` dataset only.

## Active vs legacy

**Only `data/platybrowser_6dpf/` is actively maintained.** Everything else is legacy from the original 2019–2020 publication and should not be modified:

| Directory | Author | Last touched | Status |
|-----------|--------|-------------|--------|
| `segmentation/` | Constantin Pape | 2019 | Legacy — original paper |
| `registration/` | Constantin Pape | 2019 | Legacy — original paper |
| `mmpb/` | Constantin Pape | Feb 2020 | Legacy — original backend library |
| `analysis/` (all subdirs) | Constantin Pape | 2020 | Legacy — original paper analyses |
| `misc/` | Constantin Pape | 2019–2020 | Legacy — original bookmarks |
| `software/` | Constantin Pape | 2020 | Legacy — original conda envs |
| `data/0.0.0` through `data/1.0.1` | Constantin Pape | 2019–2021 | Legacy — old versioned datasets |
| `data/platybrowser_6dpf/` | Cyril Cros | Aug 2025 | **Active** — current dataset |

**Snakemake** was used for the original 2020 publication analyses (gene clustering, morphology clustering). It is no longer used. The Snakefiles, rules, and scripts in `analysis/` are kept for reference only — do not run or modify them.

Do not edit the legacy directories unless explicitly asked. Treat them as read-only historical artifacts.

## Data structure

### MoBIE project layout

The project follows the [MoBIE specification v0.3.0](https://mobie.github.io/specs/mobie_spec.html). A MoBIE project is organized around four key concepts:

**Project** — top-level container grouping datasets. Defined by `data/project.json`, which lists available datasets, the default dataset, a description, references, and the spec version. The current `project.json` lists all historical versioned datasets but defaults to `platybrowser_6dpf`.

**Dataset** — a self-contained collection of sources, views, images, and tables. Defined by `dataset.json` in the dataset root. Must contain `images/` (BDV XML metadata), `tables/` (derived tabular data), and optionally `misc/` (bookmarks, extra views).

**Source** — a single image, segmentation, spots, or regions dataset. Four types exist:
- `image` — intensity images (EM, light microscopy, ProSPr gene expression)
- `segmentation` — label masks with integer-labeled objects, plus an associated table directory
- `spots` — point-like data (e.g., gene detections), defined purely by a table
- `regions` — region annotations referencing a region table

**View** — a complete viewer state. Defined in `dataset.json` under `views` (or in `misc/views/` as separate JSON files). Contains:
- `sourceDisplays` — arrays of `imageDisplay`, `segmentationDisplay`, `spotDisplay`, or `regionDisplay` specifying which sources to show with what color maps, opacity, LUT, contrast limits, and table settings
- `sourceTransforms` — affine, crop, mergedGrid, transformedGrid, and timepoints transformations applied to sources
- `viewerTransform` — the initial viewer camera position, rotation, and timepoint
- `uiSelectionGroup` — which UI menu group to show this view under (e.g. `"bookmark"`)
- `isExclusive` — `true` = **supersedes** the viewer state. Use only for polished, self-contained figure panels (publication figures, curated cell-type views). An exclusive view must provide its own `sourceDisplays`, camera position, and visual styling. `false` = additive (layers on top of current view)

A dataset must contain a `default` view. Supported image data formats: `bdv.n5`, `bdv.n5.s3`, `bdv.hdf5`, `bdv.ome.zarr`, `bdv.ome.zarr.s3`, `ome.zarr`, `ome.zarr.s3`, `openOrganelle.s3`.

## Sources in `platybrowser_6dpf`

The dataset has sources of four types: image (intensity data — EM, gene expression, HCR probes), segmentation (labeled object masks), spots (point-like table data), and regions (region annotations). Every source specifies both a `bdv.n5` (local) and a `bdv.n5.s3` path, so the viewer can load from either storage backend. We use an EMBL Minio S3 install with two path prefixes:

| Prefix | Purpose |
|--------|---------|
| `images/bdv-n5-s3/vergara_2021/` | Original Vergara et al. 2021 data (raw EM, ProSPr gene expression, all segmentations) |
| `images/bdv-n5-s3/paper_2025/` | New HCR-spotiflow data for 2025 paper |

Molecular data includes ProSPr gene expression (200+ genes), HCR-spotiflow probes, and associated tables (gene expression, clustering, UMAP). Stainings include EdU pulse-chase labelings. Segmentation sources cover cells, nuclei, chromatin, tissue, plus anatomical structures (foregut, midgut, VNC, neuropil, shell, midline, glands, ganglia) and organelles (cilia).

### Source definition pattern

Every source follows this structure:

```json
{
  "source-name": {
    "image": {                                        // or "segmentation"
      "imageData": {
        "bdv.n5": {
          "relativePath": "images/local/file.xml"      // local N5
        },
        "bdv.n5.s3": {
          "relativePath": "images/bdv-n5-s3/PREFIX/file.xml"  // S3
        }
      }
    }
  }
}
```

**Segmentation sources** additionally include `tableData`:

```json
"tableData": {
  "tsv": {
    "relativePath": "tables/sbem-6dpf-1-whole-segmented-cells"
  }
}
```

### S3 path conventions

Two S3 prefixes are in use:

| Prefix | Purpose |
|--------|---------|
| `images/bdv-n5-s3/vergara_2021/` | Original Vergara et al. 2021 data (genes, segmentations, raw EM) |
| `images/bdv-n5-s3/paper_2025/` | New HCR-spotiflow data for 2025 paper |

### Source naming

- ProSPr gene expression images: short gene names (`ache`, `pax6`, `rx`)
- SBEM segmentations: full naming convention (`sbem-6dpf-1-whole-segmented-cells`)
- 2025 paper sources: descriptive with metadata (`"MYH1 (striated muscle) | XLOC_045336 : HCR-spotiflow (AP_004)"`)

### Key segmentation sources

| Source name | Type | Table path |
|-------------|------|------------|
| `cells` | segmentation | `tables/sbem-6dpf-1-whole-segmented-cells` |
| `nuclei` | segmentation | `tables/sbem-6dpf-1-whole-segmented-nuclei` |
| `virtual-cells` | segmentation | `tables/prospr-6dpf-1-whole-virtual-cells` |
| `tissue` | segmentation | `tables/sbem-6dpf-1-whole-segmented-tissue` |
| `chromatin` | segmentation | `tables/sbem-6dpf-1-whole-segmented-chromatin` |

## Views in `platybrowser_6dpf`

The dataset has views across 12 UI selection groups:

| Group | Purpose |
|-------|---------|
| `prospr` | Individual gene expression views (additive) |
| `Figures Vergara2021` | Figure panels from the original 2021 paper |
| (no group) | Ungrouped views |
| `HCR_combined` | Combined HCR-spotiflow stainings |
| `stainings-2025-paper` | 2025 paper: individual stainings |
| `sbem` | EM image overlaid views |
| `prospr-mask` | Gene expression with mask overlays |
| `sbem-segmentation` | Segmentation-only views (no EM) |
| `traces` | Neuron trace views |
| `anatomical-views` | Canonical orientations (coronal, registration) |
| `Figures Pape2023` | Figure from Pape 2023 paper |
| `prospr-segmentation` | Segmentation in ProSPr space |

### View patterns

**Simple navigation view** (no displays, just camera position):
```json
{
  "uiSelectionGroup": "Figures Vergara2021",
  "viewerTransform": {
    "normalizedAffine": [0.018, -0.039, 0.0, 3.35, 0.039, 0.018, 0.0, -7.74, 0.0, 0.0, 0.043, -2.32]
  },
  "isExclusive": false
}
```

**Image display views** (gene expression, additive):
```json
{
  "uiSelectionGroup": "prospr",
  "sourceDisplays": [{
    "imageDisplay": {
      "sources": ["ache"],
      "color": "randomFromGlasbey",
      "contrastLimits": [0.0, 1000.0],
      "opacity": 1.0,
      "visible": true,
      "showImagesIn3d": false,
      "name": "ache"
    }
  }],
  "isExclusive": false
}
```

### 10 diverse view examples

**1. `default`** — simplest exclusive view, sets the initial state:
```json
{
  "uiSelectionGroup": "Figures Vergara2021",
  "sourceDisplays": [{"imageDisplay": {
    "sources": ["raw"], "color": "white", "contrastLimits": [0.0, 255.0],
    "opacity": 1.0, "name": "raw", "visible": true, "showImagesIn3d": false
  }}],
  "isExclusive": true
}
```

**2. `ache`** — additive gene expression, using random glasbey color:
```json
{
  "uiSelectionGroup": "prospr",
  "sourceDisplays": [{"imageDisplay": {
    "sources": ["ache"], "color": "randomFromGlasbey",
    "contrastLimits": [0.0, 1000.0], "opacity": 1.0, "name": "ache", "visible": true
  }}],
  "isExclusive": false
}
```

**3. `cells`** — segmentation-only additive view (no raw EM behind it):
```json
{
  "uiSelectionGroup": "sbem-segmentation",
  "sourceDisplays": [{"segmentationDisplay": {
    "sources": ["cells"], "lut": "glasbey", "opacity": 0.5,
    "showTable": true, "visible": true, "name": "cells",
    "opacityNotSelected": 0.15, "randomColorSeed": 42
  }}],
  "isExclusive": false
}
```

**4. `Figure 7C: Virtual cell assignment: gene expression level`** — numerical LUT with value limits and additional tables:
```json
{
  "uiSelectionGroup": "Figures Vergara2021",
  "sourceDisplays": [
    {"imageDisplay": {"sources": ["raw"], "color": "white", "contrastLimits": [0.0, 255.0], "opacity": 1.0, "name": "raw", "visible": true}},
    {"segmentationDisplay": {
      "sources": ["cells"], "lut": "viridis",
      "colorByColumn": "expression_sum", "valueLimits": [0.0, 40.0],
      "additionalTables": ["vc_assignments.tsv"],
      "opacity": 0.5, "name": "cells", "showTable": true, "visible": true
    }}
  ],
  "isExclusive": true,
  "viewerTransform": {"normalizedAffine": [...]}
}
```

**5. `Figure 3B: Morphology clustering full body`** — ARGB column color scheme (per-row colors from table):
```json
{
  "uiSelectionGroup": "Figures Vergara2021",
  "sourceDisplays": [
    {"imageDisplay": {"sources": ["raw"], "color": "r=255,g=255,b=255,a=255", "contrastLimits": [0.0, 255.0], "opacity": 1.0, "name": "raw", "visible": true}},
    {"segmentationDisplay": {
      "sources": ["cells"], "lut": "argbColumn",
      "colorByColumn": "morphologyColourScheme",
      "additionalTables": ["morphology_clusters.tsv"],
      "opacity": 0.5, "name": "cells", "showTable": true, "visible": true
    }}
  ],
  "isExclusive": true,
  "viewerTransform": {"normalizedAffine": [...]}
}
```

**6. `Fig2_prediction_brain_ACh_SSN_bsx_Dlx`** — two segmentation displays on one view, with divergent color maps and selected segments:
```json
{
  "uiSelectionGroups": ["2025-paper-cell-type-predictions"],
  "sourceDisplays": [
    {"imageDisplay": {"sources": ["raw"], "color": "r=255,g=255,b=255,a=255", "blendingMode": "sum", ...}},
    {"segmentationDisplay": {
      "sources": ["cells"], "lut": "blueWhiteRed", "colorByColumn": "source",
      "selectedSegmentIds": ["cells;0;4817", "cells;0;5507", ...],  // 57 preselected segments
      "opacity": 0.5, "name": "cells", "randomColorSeed": 50, ...
    }},
    {"segmentationDisplay": {
      "sources": ["nuclei"], "lut": "viridisZeroTransparent",
      "colorByColumn": "clade11sub48", "valueLimits": [10.0, 1.0],
      "additionalTables": ["master_top_10_ranks.tsv"],
      "opacity": 0.5, "name": "nuclei", ...
    }}
  ],
  "sourceTransforms": [],
  "viewerTransform": {"normalizedAffine": [...], "timepoint": 0},
  "isExclusive": true
}
```

**7. `coronal`** — anatomical orientation (transverse and coronal planes), uses `normalVector` instead of `normalizedAffine`:
```json
{
  "uiSelectionGroup": "anatomical-views",
  "viewerTransform": {"normalVector": [0.7, 0.56, 0.43]},
  "isExclusive": false
}
```

**8. `allglands`** — mask overlay view with explicit color:
```json
{
  "uiSelectionGroup": "prospr-mask",
  "sourceDisplays": [{"imageDisplay": {
    "sources": ["allglands"], "color": "magenta",
    "contrastLimits": [0.0, 1000.0], "opacity": 1.0, "name": "allglands", "visible": true
  }}],
  "isExclusive": false
}
```

**9. `david_cells`** — 3D view with `showImagesIn3d` and `selectedSegmentIds`:
```json
{
  "uiSelectionGroups": ["curated-cell-types"],
  "sourceDisplays": [
    {"imageDisplay": {
      "sources": ["raw"], "color": "r=255,g=255,b=255,a=255",
      "blendingMode": "sum", "showImagesIn3d": true, ...
    }},
    {"segmentationDisplay": {
      "sources": ["cells"], "lut": "glasbey",
      "selectedSegmentIds": ["cells;0;8380", "cells;0;8413", ...],  // 128 preselected cells
      "opacity": 0.5, "name": "cells", ...
    }}
  ],
  "viewerTransform": {"normalizedAffine": [...], "timepoint": 0},
  "isExclusive": true
}
```

**10. `Fig3: coregulons_raw_middle_view`** — nuclei-based clustering with `glasbey` LUT and `colorByColumn` using a text column:
```json
{
  "uiSelectionGroups": ["2025-paper-cell-type-predictions"],
  "sourceDisplays": [
    {"imageDisplay": {"sources": ["raw"], "blendingMode": "sum", ...}},
    {"segmentationDisplay": {
      "sources": ["nuclei"], "lut": "glasbey",
      "colorByColumn": "most probable cluster",
      "additionalTables": ["cluster_probability.tsv"],
      "opacity": 1.0, "name": "nuclei", ...
    }}
  ],
  "viewerTransform": {"normalizedAffine": [...], "timepoint": 0},
  "isExclusive": true
}
```

### All available LUT values

| LUT | Use case |
|-----|----------|
| `glasbey` | Categorical (random distinct colors), default for segmentations |
| `argbColumn` | Per-row ARGB colors from a table column (format: `alpha-red-green-blue`) |
| `viridis` | Numeric continuous colormap, requires `valueLimits` |
| `viridisZeroTransparent` | Like viridis but value 0 = invisible |
| `blueWhiteRed` | Divergent numeric colormap, requires `valueLimits` |
| `glasbeyZeroTransparent` | Like glasbey but label_id 0 = invisible |

### Color values for `imageDisplay`

| Value | Meaning |
|-------|---------|
| `"white"` | Fixed white (greyscale for EM) |
| `"r=255,g=255,b=255,a=255"` | Explicit RGBA |
| `"magenta"` | Named color |
| `"randomFromGlasbey"` | Random glasbey-distinct color |

### Full list of view properties

**imageDisplay properties:**
`sources`, `color`, `contrastLimits`, `opacity`, `visible`, `showImagesIn3d`, `invert`, `blendingMode` (`"sum"` or `"alpha"`), `name`

**segmentationDisplay properties:**
`sources`, `lut`, `colorByColumn`, `valueLimits` [min, max], `opacity`, `opacityNotSelected` (default 0.15), `visible`, `showTable`, `showAsBoundaries`, `boundaryThickness`, `showScatterPlot`, `scatterPlotAxes` [xCol, yCol], `showSelectedSegmentsIn3d`, `selectedSegmentIds` (format: `"sourceName;timePoint;label_id"`), `additionalTables` (array of TSV filenames), `randomColorSeed`, `name`

**sourceTransforms types:**
`affine` (12-param BDV matrix), `crop` (min/max bounding box), `mergedGrid` (tiled sources), `transformedGrid` (grid-spaced arrangement). Currently all sourceTransforms arrays in this dataset are empty; they are defined but unused.

**viewerTransform variations:**
- `normalizedAffine` — 12-element array (3×4 matrix in column-major order, BDV convention)
- `normalVector` — 3-element array defining view plane normal (for anatomical orientations)
- Both optionally include `timepoint`

### Writing concise views (omit viewer defaults)

The viewer (`mobie/mobie-viewer-fiji`, `src/main/java/org/embl/mobie/lib/serialize/`) deserializes views with Gson using each class's no-arg constructor: **any field absent from the JSON keeps its Java default**. Unknown/extra fields are silently ignored. So a field only needs to be written when its value differs from the default — omitting it is behavior-identical and keeps `dataset.json` small (a 2025 sweep removed ~3,500 redundant keys / 121 KB / 19.4%).

**Rule — reduce every new or edited view.** Whenever a view is added to or modified in `dataset.json` — by hand, by the MoBIE Fiji plugin's save, or by a generator script — check whether it was written with explicit default values, and reduce it to the concise form below whenever possible. Any key whose value matches a default in the tables below is redundant and should be removed before committing. This also applies when syncing handcrafted views (see "Syncing handcrafted views").

The `2025_scripts/compress_dataset_json.py` script performs this reduction
automatically, and the pre-commit hook runs it (auto-staging the result)
plus `2025_scripts/validate_dataset_json.py` on every commit. GitHub Actions
enforces both on `main`.

**imageDisplay** — omit when the value equals the default:

| Field | Default when absent |
|---|---|
| `opacity` | 1.0 |
| `visible` | true |
| `invert` | false |
| `showImagesIn3d` | false |
| `blendingMode` | `"sum"` |
| `color` | white (absent = white, same as `"white"` or `"r=255,g=255,b=255,a=255"`) |
| `contrastLimits` | source type min/max (not applied) — omit only when the source's natural range is intended |
| `name` | **required — never omit.** UI panel label; the viewer renders `display.getName()` directly with no fallback (`UserInterfaceHelper#createDisplayPanel`), and display classes have no default for it (absent = null → blank label). Always write it, normally `= sources[0]`. |

**segmentationDisplay / spotDisplay / regionDisplay** — omit when equal to the default:

| Field | Default when absent |
|---|---|
| `visible` | true |
| `opacity` | 0.5 |
| `showTable` | true |
| `lut` | `"glasbey"` |
| `colorByColumn` | null (per-segment color) |
| `valueLimits` | null (categorical coloring) |
| `showAsBoundaries` | false |
| `boundaryThickness` | 1.0 |
| `showScatterPlot` | false |
| `scatterPlotAxes` | `["anchor_x", "anchor_y"]` |
| `showSelectedSegmentsIn3d` | false |
| `selectedSegmentIds` | null (no preselection; `[]` is the same as absent) |
| `additionalTables` | null |
| `randomColorSeed` | 42 |
| `opacityNotSelected` | 0.15 |
| `blendingMode` | `"alpha"` (annotation displays default to alpha, unlike imageDisplay) |
| `name` | **required — never omit.** UI panel label; same rule as `imageDisplay` (`= sources[0]` normally, custom label allowed) |

**View level:**

| Field | Default when absent |
|---|---|
| `isExclusive` | false — stripped automatically by `2025_scripts/compress_dataset_json.py` (pre-commit hook) |
| `sourceTransforms` | no transform — omit entirely; empty `[]` arrays are dead weight |
| `uiSelectionGroup(s)` | `"views"` group |
| `viewerTransform.timepoint` | timepoint unchanged (BDV default 0; this dataset is single-timepoint, so `timepoint: 0` is always omit-able) |

**sourceTransforms objects:** `crop.centerAtOrigin` = false, `mergedGrid.name` = `"merged image"`, `mergedGrid.centerAtOrigin` = false, `grid`/`transformedGrid.centerAtOrigin` = true, grid `margin` = 0.1, grid `positions` = null → auto-grid.

**Minimal examples.** An additive gene-expression view collapses to:

```json
{
  "uiSelectionGroup": "prospr",
  "sourceDisplays": [{
    "imageDisplay": {
      "sources": ["ache"],
      "color": "randomFromGlasbey",
      "contrastLimits": [0.0, 1000.0],
      "name": "ache"
    }
  }]
}
```

(no `opacity`, `visible`, `invert`, `showImagesIn3d`, `blendingMode` — all default). A plain segmentation view:

```json
{
  "sourceDisplays": [{
    "segmentationDisplay": {
      "sources": ["cells"],
      "name": "cells"
    }
  }]
}
```

**Caveats**

- `contrastLimits: [0.0, 255.0]` equals the default only for 8-bit sources — check the source bit depth before omitting.
- The MoBIE Fiji plugin re-writes views with explicit values when saving a view, and the `2025_scripts/` generators emit verbose views — a stripped `dataset.json` becomes verbose again on regeneration (see "Syncing handcrafted views"). Stripping is a one-time cleanup, not a format guarantee.
- These defaults were verified against `mobie/mobie-viewer-fiji` (`org.embl.mobie.lib.serialize`). Re-verify there if the viewer version changes.

### Active dataset: `data/platybrowser_6dpf/`

```
data/platybrowser_6dpf/
├── dataset.json          # Sources, views, defaultLocation
├── images/
│   └── local/            # BDV XML metadata for local images
├── tables/               # TSV tables keyed by source name
│   ├── sbem-6dpf-1-whole-segmented-cells/    # Cells (default table + genes, regions, morphology, clusters, etc.)
│   ├── sbem-6dpf-1-whole-segmented-nuclei/   # Nuclei
│   ├── sbem-6dpf-1-whole-segmented-ganglia/  # Ganglia
│   ├── prospr-6dpf-1-whole-virtual-cells/   # ProSPr virtual cells
│   └── ...
└── misc/
    ├── dynamic_segmentations.json
    └── ...
```

### Table conventions

- Every table directory must contain a `default.tsv` (or `default.csv`) with at minimum: `label_id`, `anchor_x`, `anchor_y`, `anchor_z`, `bb_min_x/y/z`, `bb_max_x/y/z`.
- Additional tables must contain `label_id` and may contain arbitrary extra columns. Label IDs in additional tables must be a subset of those in the default table.
- Label ID `0` is reserved for background and must not appear in tables.
- Use TSV (tab-separated) by preference per the MoBIE spec, though CSV is also supported.

### Cell type predictions and type propagation

Nuclei carry cell type predictions from scRNA-seq cluster probability tables:

| File | Purpose |
|------|---------|
| `tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv` | Per-nucleus probabilities for ~260 detailed subtypes |
| `tables/sbem-6dpf-1-whole-segmented-nuclei/broad_types_cluster_probability.tsv` | Per-nucleus probabilities for 17 broad cell types |

Each probability table has one row per nucleus (`label_id` column, float) and one column per cluster, plus a `most probable cluster` column naming the winning type.

**Type propagation to other segmentations.** Cell type annotations (`most_probable_subtype`, `most_probable_broad_type`) are propagated from nuclei into related segmentation tables to enable coloring by cell type directly in their default views. The propagation chains are:

| Segmentation | Lookup chain |
|---|---|
| `nuclei` | Direct: `label_id` → probability tables → `most_probable_subtype`, `most_probable_broad_type` |
| `chromatin` | `nucleus_id` → nuclei `default.tsv` → subtype/broad_type |
| `cilia` | `cell_id` → `cells_to_nuclei.tsv` → `nucleus_id` → nuclei `default.tsv` → subtype/broad_type |
| `traces` | `nucleus_id` → nuclei `default.tsv` → subtype/broad_type |

**Column ordering convention.** Every `default.tsv` follows the same column order: `label_id` first, then lookup/relation columns (`nucleus_id`, `cell_id`, `most_probable_subtype`, `most_probable_broad_type`, etc.), then the mandatory spatial columns (`anchor_x/y/z`, `bb_min_*`, `bb_max_*`), then remaining metadata. This ordering matches how cells `default.tsv` is already structured and ensures the viewer picks up the type columns for coloring.

**label_id is immutable.** `label_id` is the segmentation mask identifier and must never be modified. Any operation that touches a `default.tsv` must preserve `label_id` values exactly. `label_id`, `nucleus_id`, and `cell_id` are distinct ontologies — they do not need to align and often don't. A nucleus `label_id` of 5000 has no relationship to a cell `label_id` of 5000. The mapping between them lives in `cells_to_nuclei.tsv`.

### Trace ID conventions

Two people did separate connectomics traces in different parts of the worm
(Kevin: head, David: motorneurons / 2nd segment), and both independently
numbered their traces 1, 2, 3, 4, … — there are **no unique IDs** across the
two sets. The "Rosetta stone" that combines them is
`tables/sbem-6dpf-1-whole-combined-traces/default.tsv`:

- `label_id` is the **nucleus ID** (traces are made by picking a nucleus and
  tracing from it). A row may also have a `cell_id` matching a segmented cell.
- `david_motorneuron_2nd_segment_traces_id` is 0 or a `label_id` from
  `tables/sbem-6dpf-1-whole-traces-MNs-David-Puga/default.tsv`.
- `kevin_head_traces_id` is 0 or a `label_id` from
  `tables/sbem-6dpf-1-whole-traces/default.tsv`.
- Each row has 0 in one of those columns — the two trace sets are disjoint.

**Inferring the ID used for a trace** (e.g. from a `selectedSegmentIds` value):
- all values < 1000 → one of the old per-person trace IDs (not used);
- all values < 12000 → a nucleus ID;
- cell IDs are generally not used for traces (they may appear above 12000).

**The canonical trace source is `sbem-6dpf-1-whole-combined-traces`.** If a
view references `sbem-6dpf-1-whole-traces` or
`sbem-6dpf-1-whole-traces-MNs-David-Puga` directly, warn the user — those are
the legacy per-person sources and IDs in them are ambiguous.

### Image naming convention

```
MODALITY-STAGE-ID-REGION
```

Examples: `sbem-6dpf-1-whole-segmented-cells`, `prospr-6dpf-1-whole`

### Coordinate system note

The two main viewing planes are transverse and coronal. The X/Y axes do not fully match across data modalities (SBEM vs ProSPr have a registration mismatch in the lateral plane). The Z axis is mostly consistent across datasets.

## Active branches

All recent work branches diverge from `main` at `9fcf57b` (May 2026):

| Branch | Author | Last active | Purpose |
|--------|--------|-------------|---------|
| `adding_views` | Cyril Cros | Jul 2026 | Current: views, skill, AGENTS.md |
| `traces` | Cyril Cros | Jun 2026 | Neuron traces |
| `adding_alyona` | Cyril Cros | May 2026 | HCR probe data |
| `david_experimental` | Cyril Cros | May 2026 | David Puga's traces + cell types |
| `variations_david` | Cyril Cros | May 2026 | Trace color schemes |

## Important: always pull before editing

**MoBIE may modify `dataset.json` on the remote** — for example, when a user saves a new view through the Fiji plugin, it gets committed and pushed to the branch. Before any local edit to `dataset.json`, always:

```
git pull --rebase
```

This avoids conflicts and prevents overwriting views added by collaborators through MoBIE.

## Handcrafted view conventions

Handcrafted cell-type views in `2026_views_curated/` follow a strict naming pattern:

```
{subclade}__{descriptive_name}
```

For example: `clade11sub51__brain_ACh_MN_mnx_phox2_Lhx15`

(The `family_types` prefix was dropped from curated view names on 2026-08-06
while the cell types are being renamed; the `subclade` part is kept.)

These views must be:
- **Non-exclusive** (`isExclusive: false`) — additive toggles that layer on top of existing state
- **No camera transform** — `viewerTransform` removed, so toggling them doesn't change the view angle
- **Segmentation only** — no `imageDisplay` (raw EM or gene markers), only `segmentationDisplay` with preselected cells
- **Glasbey LUT** by default
- **No `raw` source** — non-exclusive views must not include `raw` as an imageDisplay source; the user typically already has EM open, and a second `raw` layer makes it look overexposed

Each curated cell type produces two view variants, shown in two separate
dropdowns (uiSelectionGroups), with a producer suffix `_detlev` or `_sam`:
- `{name}_detlev` / `{name}_sam` — naked views in group `curated_views`:
  segmentation only (`cells` + `nuclei` + traces where available), no camera,
  for quickly toggling cell positions
- `{name}_illustrated_detlev` — illustrated views in group
  `curated_views_probes_on`: marker `imageDisplay`s (no `raw`) for exploring
  the markers used to annotate the cell type

Unresolved/questionable views live in the `status_unclear_curated_views` group.

The `subclade` in the view name comes from the cell type master list (`cell_types_masterlist.tsv`). Views that match the master list are in `2026_views_curated/`. Views with unresolved matches (double assignments, missing CSV entries) are in `2025_scripts/detlev_handcrafted_views_questionable/`.

See `2025_scripts/README_handcrafted_views.md` for full documentation.

## Syncing handcrafted views

The canonical source for handcrafted views lives in `2026_views_curated/`. When a JSON file in this directory is updated, the corresponding view in `dataset.json` must be synced to match. Do not edit these views directly in `dataset.json` — edit the JSON file, then sync. When syncing, reduce the view to its concise form (omit any key that equals the viewer default — see "Writing concise views (omit viewer defaults)") so it matches the one-key-per-non-default convention of the rest of `dataset.json`.

## Do not touch

- **Raw image data on network drives or S3** — never modify, delete, or move actual image data (N5, HDF5) on network drives (`W:/`, `Z:/`) or the S3 bucket. These are read-only references. Only metadata in git may be edited.
- **Legacy directories** — `segmentation/`, `registration/`, `mmpb/`, `analysis/`, `misc/`, `software/`, and old `data/` version directories (`0.0.0`–`1.0.1`) are historical artifacts. Do not edit.
- **MoBIE-breaking structural changes** — validate against the [MoBIE spec](https://mobie.github.io/specs/mobie_spec.html) before any changes to `data/project.json`, `dataset.json`, or table/image structure.
- **Upstream divergence** — this is a fork of `mobie/platybrowser-project`. Avoid changes that would make future merging impossible.

## When you can edit

- `data/platybrowser_6dpf/dataset.json` — add or modify sources, views, sourceTransforms, sourceDisplays
- `data/platybrowser_6dpf/tables/` — add new tables, update table contents following the MoBIE table spec
- `data/platybrowser_6dpf/images/local/` — add new BDV XML metadata files for new image sources
- `data/platybrowser_6dpf/misc/` — add bookmarks, additional view JSON files
- `data/project.json` — only if adding a new dataset to the list

## Verification

- MoBIE structural validity: the viewer will fail to load if `dataset.json`, table columns, or BDV XML references are broken. Validate manually with the MoBIE Fiji plugin.
- Table integrity: ensure `label_id` columns are consistent across all tables for a source, and that `default.tsv` always has the mandatory spatial columns.
