# Nuclei Probability Dropdown (`proba_as_dropdown`)

Date: 2026-08-25 (rev. 2 — image-source approach restored)
Status: Approved design (pending spec review)
Branch: `proba_as_dropdown`

## Goal

Let users browse cell-type assignment probabilities for nuclei in MoBIE through a
dropdown: one entry per detailed cell type (clade subtype), shown as a mostly-black
image in which each nucleus is painted with its probability for that subtype. The
overlays are recolored interactively in the viewer (LUT), and **multiple cell types
can be opened additively at the same time**.

## Approach

Generate **one new N5 image per subtype** by repainting the nuclei segmentation
mask: every voxel of a nucleus keeps the mask geometry but its value becomes the
nucleus's probability for that subtype (quantized to 0.001). Background (non-nucleus)
stays 0. These are plain image sources, loaded additively like the gene-expression
overlays, and recolored in the viewer.

### Why images and not segmentation displays

The user wants per-cell-type *image* sources so that several different cell types can
be shown simultaneously as independent overlays (a dropdown where you tick multiple
types at once), with each recolored separately in the viewer. This is not possible
with a single segmentation display, and multiple segmentation displays re-coloring
the same `nuclei` source were not what the user wants.

## Data source

`data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv`

- 11,382 nuclei (one row per `label_id`)
- **274 subtype probability columns** (all `clade*` / `nocladesub*` columns;
  excludes `label_id`, `zero`, `autofluorescence`, `most probable cluster`)
- Values are floats in [0, 1]

Mask geometry (from the `nuclei` source `sbem-6dpf-1-whole-segmented-nuclei`):

- 3438 × 3240 × 2854 voxels, voxel size 0.08 / 0.08 / 0.1 µm
- uint32 label IDs; label 0 = background

## Deliverables

### 1. 274 probability images (N5)

For each subtype column `s`:

- Grid, voxel size, and affine identical to the nuclei mask → exact overlay on EM/nuclei
- dtype **uint16**, value per nucleus voxel = `round(p(nucleus, s) × 1000)` ∈ [0, 1000]
  (0.001 precision; `p < 0.0005` → 0)
- Background = 0; N5 written with gzip + `fill_value: 0` → the image is *mostly
  black* and compresses to almost nothing (only nuclei with non-negligible
  probability for that subtype are non-zero)
- File/source name: `{subtype}_proba` (e.g. `clade1sub1_proba`)

### 2. Storage layout

- S3 (EMBL Minio, bucket `platybrowser-2025`): `images/bdv-n5-s3/celltype_proba/`
  — N5 datasets + one BDV XML per subtype there
- Local mirror: `data/platybrowser_6dpf/images/local/{subtype}_proba.xml` (bdv.n5
  path), matching the dataset convention of every source carrying both `bdv.n5` and
  `bdv.n5.s3` paths
- **Only adds data under the new `celltype_proba` prefix; never modifies existing S3 content**

### 3. `dataset.json`

- **274 image sources** `{subtype}_proba`, each with `imageData` → `bdv.n5` (local
  XML) + `bdv.n5.s3` (S3 XML)
- **274 additive views** in uiSelectionGroup `nuclei_probabilities`, concise form:

```json
{
  "uiSelectionGroup": "nuclei_probabilities",
  "sourceDisplays": [
    {
      "imageDisplay": {
        "sources": ["clade1sub1_proba"],
        "contrastLimits": [0.0, 1000.0],
        "name": "clade1sub1_proba"
      }
    }
  ]
}
```

- `isExclusive` omitted (false) → additive; `viewerTransform` omitted → camera
  unchanged; `color` omitted → white/grayscale default (the user recolors in the
  viewer); no `raw` source (handcrafted-view convention)
- Because each view uses a **distinct source**, any number of them can be toggled on
  together (sum blending, each with its own contrast limits / recolor)

### 4. Rendering / "recolor" note (viridis)

Verified in `mobie/mobie-viewer-fiji` (`ImageDisplay.java` serialization fields,
`ColorHelper.getARGBType`) and the MoBIE spec v0.3.0: `imageDisplay` has **no `lut`
field** — `color` accepts single colors only (ARGB strings, `java.awt.Color` names,
`randomFromGlasbey`). A colormap such as `viridis` therefore cannot be the
dataset.json default for an image source.

Consequence: images ship as grayscale (white) overlays; the **viridis LUT is applied
interactively in the viewer** per source. The pilot must validate this workflow:
recolor one pilot image with a LUT, and check whether the recolor persists across
sessions (or must be reapplied).

### 5. Scripts (`2025_scripts/`, uv inline-script pattern)

- `generate_nuclei_proba_images.py` — reads the nuclei mask N5 (path argument) +
  the probability TSV; builds a per-subtype `label_id → round(p×1000)` map; writes
  one uint16 N5 per subtype (gzip, fill_value 0) + a BDV XML per subtype from the
  nuclei XML template. Single pass over the mask (per-chunk relabel into all 274
  outputs).
- S3 upload — reuse/extend the existing `upload_Alyona_local_n5_to_s3.py` pattern to
  populate `images/bdv-n5-s3/celltype_proba/` (`.env` credentials).
- `add_proba_sources_and_views.py` — appends the 274 sources + 274 views to
  `data/platybrowser_6dpf/dataset.json` (dry-run default), following the pattern of
  `add_sources_and_views_to_n5_s3_data.py`.

## Workflow

1. **Pilot (5 subtypes first).** Representative selection: one abundant
   (`clade6sub19`), one rare, one `nocladesubX`, a CNS subtype, and a midgut
   subtype (e.g. `clade1sub2`). Generate locally (no S3 yet), wire 5 sources + 5
   views into `dataset.json`. User validates in Fiji:
   - images load (local XMLs) and overlay the EM exactly
   - mostly-black rendering with nucleus intensities
   - **recolor with a LUT works; check persistence across sessions**
   - multiple pilot views open additively, each independently recolored
   - **checkpoint:** measure per-image size and generation time at full
     resolution; if impractical for 274 images, generate downsampled-only images
     (value is constant per nucleus, so a 2–4× coarser grid is visually
     equivalent; voxel size adjusted to keep the physical extent identical)
2. **Full run (274 images).** Generate all, upload to S3, commit XMLs + sources +
   views in one commit.
3. **Post-commit checks.** Pre-commit hook runs `compress_dataset_json.py`
   (strips default keys, auto-stages) + `validate_dataset_json.py`; GitHub Actions
   enforces both on `main`.

## Verification

- `python 2025_scripts/validate_dataset_json.py` passes after edits.
- `git diff` on `dataset.json` shows only added sources/views; no existing content
  altered.
- Value sanity: every `label_id` present in the mask is also in the probability
  table (no unassigned nuclei → no phantom zeros/values).
- Manual MoBIE validation by the user in Fiji (pilot step above).
- Confirm counts: 274 sources + 274 views under `nuclei_probabilities` (pilot: 5).

## Constraints

- No modifications to existing raw image data or S3 content; only additive under
  `celltype_proba`.
- Legacy directories and other datasets untouched (AGENTS.md rules).
- `label_id` values in tables are immutable; the mask is only read, never written.
- The `nuclei` source definition, default table, and probability table are unchanged
  (the table is only read).

## Rejected alternatives (record)

1. **Segmentation views with `lut: viridis` + `colorByColumn`** (MoBIE-native,
   no new images) — rejected by the user; wants independent per-cell-type image
   sources.
2. **ARGB images with baked-in viridis colors** — rejected by the user; wants
   numeric pixel values and interactive recolor.
3. **Binary 0/1 rounding** (round to 0 or 1) — superseded by raw probability at
   0.001 precision.
4. **Broad types (17) only** — rejected; detailed subtypes (~274) wanted.
