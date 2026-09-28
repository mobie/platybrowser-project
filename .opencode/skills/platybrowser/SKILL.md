---
name: platybrowser
description: Use when working on the PlatyBrowser MoBIE project. Covers how to install MoBIE, access this repo, how MoBIE tables and views work, and the platybrowser_6dpf dataset-specific conventions (cells/nuclei mapping, table structure, source and view patterns).
---

# PlatyBrowser Skill

## Setup: how to install MoBIE and access this project

The project is based on the Vergara et al. (2021) *Platynereis dumerilii* multimodal atlas. See [`DATASET.md`](DATASET.md) for details on how the data was generated, segmented, and its scientific significance.

1. Install [Fiji](https://imagej.net/software/fiji/downloads)
2. Install the MoBIE Fiji plugin:
   - In Fiji, go to `Help > Update...`, click `Manage update sites`
   - Check `MoBIE` and click `Close`, then `Apply changes`
   - Restart Fiji
3. Open the project:
   - In Fiji, go to `Plugins > MoBIE > Open MoBIE Project...`
   - Enter the GitHub repo URL: `https://github.com/cyrilcros/platybrowser-project-2025`
   - This loads `data/project.json` and defaults to the `platybrowser_6dpf` dataset

The project uses an EMBL Minio S3 install for image data. The BDV XML metadata files reference both local (`bdv.n5`) and S3 (`bdv.n5.s3`) paths, so the viewer can load from either backend.

The `default` view shows the raw EM data. Subsequent views are additive (gene expression overlays, segmentations, etc.) or exclusive (figure panels that replace the current state).

## How tables work in MoBIE

Every segmentation source has an associated table directory under `data/platybrowser_6dpf/tables/`. Each directory must contain a `default.tsv` with spatial columns:

| Column | Required | Description |
|--------|----------|-------------|
| `label_id` | yes | Integer object ID (0 reserved for background) |
| `anchor_x`, `anchor_y`, `anchor_z` | yes | Reference point (viewer centers on this when selected) |
| `bb_min_x/y/z`, `bb_max_x/y/z` | yes | Bounding box in physical units |

Additional tables contain `label_id` plus arbitrary extra columns. Label IDs in additional tables must be a subset of those in `default.tsv`. They are loaded on demand via `additionalTables` in a view's `segmentationDisplay`.

The viewer displays numeric columns via LUTs (`viridis`, `blueWhiteRed` with `valueLimits`), categorical columns via `glasbey` or `argbColumn` (per-row ARGB values).

## How views work in MoBIE

Views are defined in `dataset.json` under the `views` key. Each view is a complete or partial viewer state. Key properties:

- `isExclusive`: controls how the view interacts with the current viewer state
  - `true` — **supersedes** the current viewer state entirely. Use this only for polished, self-contained figure panels (e.g. publication figures, curated cell-type views). An exclusive view wipes everything else, so it must set up its own `sourceDisplays`, camera position, and visual styling.
  - `false` — **additive**: layers on top of whatever the user already has open. This is the default for exploration views (gene overlays, stainings, segmentations) that the user toggles on and off to browse data.
- `sourceDisplays`: array of `imageDisplay` and/or `segmentationDisplay`
- `viewerTransform`: camera position (`normalizedAffine` 12-element BDV matrix, or `normalVector`)
- `uiSelectionGroup`: which menu group the view appears under
- `sourceTransforms`: optional affine/grid transforms (all empty in this dataset)

**imageDisplay** shows intensity data: EM, gene expression, stainings. Set `color`, `contrastLimits`, `opacity`.

**segmentationDisplay** shows labeled object masks. Set `lut`, `colorByColumn`, `valueLimits`, `additionalTables`, `selectedSegmentIds`.

## The platybrowser_6dpf dataset

### Cells and nuclei

The two main segmentation sources are:

| Source | Table directory | Objects |
|--------|----------------|---------|
| `cells` | `tables/sbem-6dpf-1-whole-segmented-cells/` | ~32,699 cells |
| `nuclei` | `tables/sbem-6dpf-1-whole-segmented-nuclei/` | ~11,497 nuclei |

**Cell → nucleus mapping** is done via `cells_to_nuclei.tsv`:

```
label_id    nucleus_id
1           0
2           0
3           5421
...
```

- Each cell's `label_id` maps to a `nucleus_id`
- `nucleus_id = 0` means the cell has no assigned nucleus
- To look up a nucleus from a cell: find the cell's `label_id` in `cells_to_nuclei.tsv`, get the `nucleus_id`, then look it up in the nuclei `default.tsv`

**Cell tables** (`tables/sbem-6dpf-1-whole-segmented-cells/`):

| File | Purpose |
|------|---------|
| `default.tsv` | Spatial data (anchor, bounding box, `n_pixels`) + `cells`, `trace_id`, `cell_type` columns |
| `cells_to_nuclei.tsv` | Maps cell `label_id` → `nucleus_id` |
| `genes.tsv` | Gene expression per cell |
| `gene_clusters.tsv` | Gene expression clustering results |
| `gene_umap.tsv` | UMAP coordinates for gene expression |
| `morphology.tsv` | Morphological features per cell |
| `morphology_clusters.tsv` | Morphology clustering results |
| `morphology_umap.tsv` | UMAP coordinates for morphology |
| `ganglia_ids.tsv` | Ganglion assignment per cell |
| `regions.tsv` | Anatomical region assignment per cell |
| `vc_assignments.tsv` | Virtual cell (ProSPr) assignments |
| `symmetric_cells.tsv` | Bilateral cell pair annotations |
| `cell-types-manual-curation.tsv` | Curated cell type labels |
| `david_assigned_vcs.tsv` | David's virtual cell assignments |
| `extrapolated_intensity_correction.tsv` | Intensity correction values |
| `ganglion_9_gene_clusters.tsv` | Gene clusters within ganglion 9 |

**Nuclei tables** (`tables/sbem-6dpf-1-whole-segmented-nuclei/`):

| File | Purpose |
|------|---------|
| `default.tsv` | Spatial data + `n_pixels` |
| `morphology.tsv` | Morphological features per nucleus |
| `master_top_10_ranks.tsv` | Top-10 cluster probability ranks |
| `broad_types_cluster_probability.tsv` | Broad cell type cluster probabilities |
| `detailed_cell_types_cluster_probability.tsv` | Detailed cell type cluster probabilities |
| `extrapolated_intensity_correction.tsv` | Intensity correction values |

### Source patterns

Every source defines both `bdv.n5` (local) and `bdv.n5.s3` (S3) paths, so the viewer can load from either backend. S3 data comes from an EMBL Minio install with two prefixes:

| Prefix | Data origin |
|--------|-------------|
| `images/bdv-n5-s3/vergara_2021/` | Vergara et al. 2021 — raw EM, ProSPr gene expression, all segmentations |
| `images/bdv-n5-s3/paper_2025/` | New HCR-spotiflow data for the 2025 paper |

ProSPr gene expression sources use short gene symbols (`ache`, `pax6`, `rx`). 2025 paper sources use descriptive names with probe and stain metadata (`"MYH1 (striated muscle) | XLOC_045336 : HCR-spotiflow (AP_004)"`).

### Data types in the dataset

**Molecular data** — gene expression and probe detections:
- ProSPr gene expression images (200+ genes, individual views in the `prospr` group)
- HCR-spotiflow probes (individual views in `stainings-2025-paper`, combined views in `HCR_combined`)
- Gene expression tables (`genes.tsv`, `gene_clusters.tsv`, `gene_umap.tsv`)
- Cluster probability tables on nuclei for cell type predictions

**Stainings**:
- EdU pulse-chase labelings (`edu42to48` and others)

**Segmentations of anatomical structures and organelles**:
- `cells` — full cell segmentation (~32,700 cells)
- `nuclei` — nuclear segmentation (~11,500 nuclei)
- `tissue` — tissue mask
- `chromatin` — chromatin segmentation
- Anatomical masks: `foregut`, `midgut`, `vnc`, `neuropil`, `shell`, `midline`, `inside`, `pygidium`, `crypticsegment`
- Glandular structures: `glands`, `allglands`
- ProSPr-based segmentations: `virtual-cells`, `outside`, `restofanimal`
- Neuron traces: `sbem-6dpf-1-whole-traces`, `sbem-6dpf-1-whole-combined-traces`
- Ganglia: `sbem-6dpf-1-whole-segmented-ganglia`

**Coordinate system**: The two main viewing planes are transverse and coronal. X/Y axes do not fully match across data modalities (SBEM vs ProSPr differ in the lateral plane). Z is mostly consistent.

### View patterns

Views are organized by purpose:
- **Exploration views** (additive, `isExclusive: false`): gene overlays (`prospr` group), HCR probes, stainings, segmentation toggles, mask overlays, anatomical orientations. These layer on top of whatever the user already has open.
- **Figure views** (exclusive, `isExclusive: true`): publication-quality panels from Vergara 2021, Pape 2023, and the 2025 paper. These supersede the current viewer state and are self-contained with their own source setup and camera position.
- **Curated cell-type views** (exclusive): views with preselected cells and specific LUT settings for cell type prediction figures.

See `AGENTS.md` for detailed view examples covering all features.

### When editing

- Modify `data/platybrowser_6dpf/dataset.json` to add or change views/sources
- Add new tables under `data/platybrowser_6dpf/tables/` following MoBIE conventions
- Add new BDV XML files under `data/platybrowser_6dpf/images/local/`
- Never modify raw image data on S3 or network drives
- Validate changes with the MoBIE Fiji plugin before committing

### Generating views from cell type probabilities

Use [`../../2025_scripts/generate_celltype_view.py`](../../2025_scripts/generate_celltype_view.py) to produce MoBIE views by filtering nuclei (or cells) from the scRNAseq cluster probability tables. See [`../../2025_scripts/README.md`](../../2025_scripts/README.md) for usage.
