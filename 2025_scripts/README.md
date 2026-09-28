# 2025 preprint of the 6dpf cell type atlas

We are adding a few scripts to help process our outputs. I also am reformatting Platybrowser legacy code. I am using BASH Shell or Python.
You should only need `uv` (`module load uv`  works at EMBL), and dependencies will be installed via [inline-script metadata](Inline script metadata).

## Reformatting tables

First I need to remove symlinked tables, so I don't affect other datasets. It is also better practice...

    cd ./data/platybrowser_6dpf/tables/
    find . -type l -exec sh -c 'cp --remove-destination -aL "$1" "$1.tmp" && mv "$1.tmp" "$1"' _ {} \;

The tables have a bunch of `1.0` / `2.0` / `3.0` columns due to bad CSV handling.
We replace all the affected columns.

    ./clean_up_integer.py

## Adding up what Alyona has registered so far

We assume we have valid views in a local folder like `data-tmp`.
For now I am symlinking (randomly...) representative images from Alyona's repo to this `data-tmp` and running this on it. The actual path for what Alyona registered is `/g/cba/exchange/buglakova/platybrowser-smfish-project/data/1.0.1/images/bdv-n5`.

We want to:

1. convert from `bdv-n5` to `bdv-n5-s3`
2. upload to a S3 bucket
3. add `bdv-n5-s3` as sources using a naming scheme from [`2025_staining_naming.csv`](./2025_staining_naming.csv)
4. use the naming scheme to create corresponding views

### From local bdv.n5 XML to bdv.n5.s3 XML and a populated S3 bucket

Let's say we we have a bunch of n5 files and XML folders with identical names as part of `data/platybrowser_6dpf/images/bdv-n5`. I want to populate 
`data/platybrowser_6dpf/images/bdv-n5-s3` by uploading to S3 and converting my XML stanza to `<ImageLoader format="bdv.n5.s3">`
with the right bucket. At this point I am not yet touching `data/platybrowser_6dpf/dataset.json`. I am also assuming the resolution, scale factor, trnasofrmation, unit, etc... are correct, I am just adjusting from a working local version to a remote one.

You need to use the `.env` file (cf copy from and fill `.env.example`) for S3 access.

    ./upload_Alyona_local_n5_to_s3.py -i ../data-tmp/ -o ../data/platybrowser_6dpf/images/bdv-n5-s3/paper_2025 \
    -e "https://s3.embl.de" -r "us-west-2"  -b "platybrowser-2025" -p "demo-v0" --dry-run

Remove the `--dry-run`  to upload. Files are sequentially uploaded which is slow but acceptable.

### From bdv.n5.s3 XML files to new sources and view in dataset.json

We are altering the `dataset.json`

    ./add_sources_and_views_to_n5_s3_data.py ../data/platybrowser_6dpf/dataset.json \ ../data/platybrowser_6dpf/images/bdv-n5-s3/paper_2025 2025_staining_naming.csv stainings-2025-paper

## Getting cell types from views into a table

!!!!!!!!!!!!
*REMEMBER TO CHECK IF VIEWS ARE EXCLUSIVE AND IN curated-cell-types*
!!!!!!!!!!

**Use case:** Detlev manually saves a view after turning up markers of interest and manually selecting cells.
We end up with a bunch of JSON files, one view each, with the name of cell type. See [this example file](./detlev_handcrafted_views/fg_GABA_SN_Dbx_Ptf1a.json).
We want a merged JSON to be possibly further edited, and a CSV table label to cell type for later.

Use `extract_cell_types.sh` (`chmod +x` may be needed) as in 

    2025_scripts/extract_cell_types.sh -j 2025_scripts/fused_views.json -t data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-cells/cell-types-manual-curation.tsv 2025_scripts/detlev_handcrafted_views/*.json

## Generating views from cell types or differentiation programmes

`generate_celltype_view.py` creates MoBIE views by filtering nuclei (or cells)
by probability threshold from the scRNAseq cluster probability tables. It reads
from `broad_types_cluster_probability.tsv` (programmes) or
`detailed_cell_types_cluster_probability.tsv` (fine-grained types), auto-detecting
based on the requested type names.

Nuclei with probability ≥ threshold for any of the requested types are preselected
via `selectedSegmentIds`. You get an exclusive view with raw EM as background.

### Usage

    # Dry-run: nuclei assigned to clade6sub19 at ≥0.8
    ./generate_celltype_view.py -n "clade6sub19" -t clade6sub19 --threshold 0.8

    # Cells (translated via cells_to_nuclei.tsv), 3D rendering
    ./generate_celltype_view.py -n "3d cells" -t clade6sub19 --cells --3d

    # Multiple types, broad programmes
    ./generate_celltype_view.py -n "neurons + glia" \
        -t Neurons -t Glia --threshold 0.5

    # Write to a standalone JSON file
    ./generate_celltype_view.py -n "my view" -t clade6sub19 --outfile my_view.json

    # Edit dataset.json in-place (adds view to the existing file)
    ./generate_celltype_view.py -n "my view" -t clade6sub19 \
        --edit ../data/platybrowser_6dpf/dataset.json

    # Per-type colours (sets lut=argbColumn, generates a colour table)
    ./generate_celltype_view.py -n "coloured types" \
        -t clade6sub19 -t nocladesub20 \
        --color clade6sub19:red --color nocladesub20:blue

### Options

| Flag | Description |
|------|-------------|
| `-n`, `--name` | View name (used as `uiSelectionGroup` unless `--group` given) |
| `-t`, `--type` | Cell type or programme name (repeatable) |
| `--threshold` | Minimum probability to include a nucleus (default: 0.8) |
| `--cells` | Select cells instead of nuclei (translates via `cells_to_nuclei.tsv`) |
| `--3d` | Enable `showImagesIn3d` and `showSelectedSegmentsIn3d` |
| `--group` | Explicit `uiSelectionGroup` (defaults to view name) |
| `--lut` | `glasbey` (default) or `argbColumn` |
| `--color` | Per-type colour: `TYPE:COLOR` (repeatable, sets `argbColumn`) |
| `--opacity` | Segmentation opacity (default: 0.5) |
| `--outfile` | Write view JSON to file |
| `--edit` | Path to `dataset.json` to edit in-place |
| `--dry-run` | Print JSON to stdout (default if no `--outfile` or `--edit`)

## Per-subtype nuclei probability sources (image overlays)

This generates **one image source per detailed cell type** (`clade*`/`nocladesub*`
column of `detailed_cell_types_cluster_probability.tsv`, 274 in total) for showing
nucleus assignment probabilities as a MoBIE image overlay. Each source is a copy
of the nuclei segmentation grid (same pyramid, same group attributes) whose
nucleus pixels are repainted with `round(p × 1000)` (uint16, 0 = background).
Output N5s use gzip + `fillvalue 0`, so each image is mostly black and small
(≈15 MB on average; ~4 GB total for all 274).

Nuclei with no probability row (115 of the 11,497 mask labels; they are absent
from both probability tables) map to 0 and stay invisible in every overlay.

The sources were generated with:

    ./generate_nuclei_proba_images.py \
        --mask <path-to>sbem-6dpf-1-whole-segmented-nuclei.n5 \
        --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
        --stage-dir tmp_celltype_proba \
        --local-xml-dir data/platybrowser_6dpf/images/local \
        --workers 32 --gzip-level 1

Then uploaded to S3 (bucket `platybrowser-2025`, prefix
`images/bdv-n5-s3/celltype_proba/`):

    mc mirror --overwrite tmp_celltype_proba/ <S3-alias>/platybrowser-2025/images/bdv-n5-s3/celltype_proba/

And wired into `dataset.json` as image sources (both `bdv.n5` + `bdv.n5.s3`),
sources only (views are added separately with their own naming):

    ./add_proba_sources_and_views.py \
        --dataset-json data/platybrowser_6dpf/dataset.json \
        --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
        --no-views --write

### Broad-type ("coregulon") probability overlay dropdown

The same pipeline produces the `coregulon_probabilities` dropdown (15 images),
one per broad-type column of `broad_types_cluster_probability.tsv`, uploaded to
S3 prefix `images/bdv-n5-s3/coregulon_proba/`:

    ./generate_nuclei_proba_images.py \
        --mask <path-to>sbem-6dpf-1-whole-segmented-nuclei.n5 \
        --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/broad_types_cluster_probability.tsv \
        --stage-dir tmp_coregulon_proba \
        --local-xml-dir data/platybrowser_6dpf/images/local \
        --subtypes "Cardiovascular system,...,Adult eye" --workers 16 --gzip-level 1

    ./add_proba_sources_and_views.py \
        --dataset-json data/platybrowser_6dpf/dataset.json \
        --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/broad_types_cluster_probability.tsv \
        --group coregulon_probabilities \
        --s3-prefix images/bdv-n5-s3/coregulon_proba \
        --subtypes "Cardiovascular system,...,Adult eye" --write

Notes:

- Score columns containing `/` or spaces (e.g. `Heme/chitin`, `Adult eye`) are
  sanitized to `_` in file and source names (`Heme_chitin_proba`,
  `Adult_eye_proba`) but kept verbatim in view names. Sanitization is required:
  MoBIE fetches source XMLs from raw.githubusercontent.com without URL-encoding,
  so a space in a relative path fails with HTTP 400.
- `add_proba_sources_and_views.py` accepts `--group` and `--s3-prefix` for any
  probability dropdown (defaults: `nuclei_probabilities` / `celltype_proba`).

### Notes

- The generated N5s must carry the mask's **group-level attributes** too
  (`setup0/attributes.json` with `dataType` + per-level `downsamplingFactors`,
  and `setup0/timepoint0/attributes.json` with `multiScale` + `resolution`) —
  MoBIE's `N5S3ImageLoader` reads `dataType` from `setup0` and fails with an NPE
  when it is missing. The generator mirrors them automatically.
- `imageDisplay` cannot render a colormap from `dataset.json` (its `color` field
  is a single color); apply a LUT (e.g. viridis) interactively in the viewer.
- Large runs: the block-fill pass is parallelized with a process pool
  (`--workers`); peak open file descriptors ≈ `workers × (subtypes + 1)`, so
  large worker counts need a high `ulimit -n`. The 274-image full run was done
  on the EMBL HPC as a Slurm job (~30 min on 32 cores).
