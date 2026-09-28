# Shell half-objects (sagittal + coronal) — Design

Date: 2026-09-11
Status: Approved (design); revised 2026-09-11 to principal-axis planes
Branch: `shell_halves` (off `main`)

## Goal

The `shell` source renders as a full closed mask around the animal and blocks
part of the 3D view. Produce **four duplicate half-shell objects** derived from
the existing shell N5 mask, so the user can toggle any one half on to see the
interior while keeping a shell surface. Two cut families, two halves each:

- **Left/right** cut: plane through the shell centroid with normal = the
  measured left–right axis `PC2` (≈ `(1,−1,0)`, the bilateral-symmetry normal).
- **Front/back** cut: plane through the centroid with normal = the measured
  dorso-ventral axis `PC3`.

Both planes contain the shell's true main (antero-posterior) axis `PC1`, which
is tilted ~32° from Z. Naming uses `left` / `right` / `front` / `back` halves.

## Context / constraints

- `shell` is modelled in `dataset.json` as an **`image`** source (not a
  table-backed segmentation), rendered with `contrastLimits [0, 1]`.
- Physical N5: `platybrowser/rawdata/sbem-6dpf-1-whole-segmented-shell.n5` on
  `https://s3.embl.de` (read-only reference). Locally only the XML metadata
  exists (`data/rawdata/` holds XMLs, no `.n5`; `.gitignore` ignores `*.n5`).
- Shell volume: **uint8**, size `860 810 714` (x, y, z), voxel size
  `0.32 0.32 0.4` µm, pyramid levels `[1,1,1] [2,2,2] [4,4,4] [8,8,8] [16,16,16]`.
- The mask is sparse binary: values `0` and `255` (2,372,162 shell voxels); it
  is a hollow, closed ovoid enclosing the animal.
- The cut planes are defined from the ovoid's measured principal axes, not the
  index origin (planes through the origin miss the object). See
  `2026-09-11-shell-ovoid-orientation.md` for the measurements.
- Do **not** modify or delete any existing S3 object. Only new keys under the
  new prefix `images/bdv-n5-s3/shell_halves/` in bucket `platybrowser-2025`.
- `mc` first alias `EmblArendtS3` (`~/.mc/config.json`) is used for both buckets.

## Geometry / masking definition

Masking is done in index space (`x`, `y`, `z` = array indices). For every kept
voxel the value is unchanged; every removed voxel becomes `0`.

The shell's centroid `c` and principal axes are computed at run time from the
mask (measured values are recorded in `2026-09-11-shell-ovoid-orientation.md`):

| axis | measured direction `(x, y, z)` | role |
|---|---|---|
| `PC1` | `(−0.359, −0.388, 0.849)` | main / antero-posterior (tilted ~32° from Z) |
| `PC2` | `(−0.633, 0.770, 0.084)` | left–right / bilateral-symmetry normal |
| `PC3` | `(0.686, 0.507, 0.522)` | dorso-ventral / flattening axis |

Both cut planes pass through `c` and contain `PC1`. `LR` = the cross-sectional
axis with the larger extent (`PC2`); `DV` = the smaller (`PC3`). Signs are fixed
deterministically (`PC1·ẑ ≥ 0`, `LR·(1,−1,0) ≥ 0`, `DV·(1,1,0) ≥ 0`).

| Source name | Plane normal | Keep | Zero out | Half |
|---|---|---|---|---|
| `shell_left`  | `LR` | `(p−c)·LR ≥ 0` | `< 0` | left  |
| `shell_right` | `LR` | `(p−c)·LR ≤ 0` | `> 0` | right |
| `shell_front` | `DV` | `(p−c)·DV ≥ 0` | `< 0` | front |
| `shell_back`  | `DV` | `(p−c)·DV ≤ 0` | `> 0` | back  |

The plane itself (`dot = 0`) is kept by both halves of a pair — a single-voxel
plane, visually negligible. The `left/right` and `front/back` labels are viewer
conventions and can be swapped by renaming. Naming caveat: the second pair
splits along the dorso-ventral axis, so `front/back` here is that split, not an
antero-posterior one.

## Approach (approved: A — local staging → mask → `mc mirror` up)

1. **Branch.** `shell_halves` off `main`.
2. **Verify write access.** Probe `EmblArendtS3/platybrowser-2025` with a tiny
   temporary object (create then delete) before generating anything.
3. **Stage the source.** `mc mirror` the shell N5 from
   `EmblArendtS3/platybrowser/rawdata/sbem-6dpf-1-whole-segmented-shell.n5`
   into `tmp_shell_halves_src/` (gitignored, kept separate from the upload
   staging so the source volume is never uploaded).
4. **Mask + write N5.** A new script `2025_scripts/generate_shell_halves.py`
   reads the level-0 volume (uint8, 860×810×714), applies the four masks, and writes four BDV N5
   volumes to `data/rawdata/shell_halves/` (gitignored upload staging):
   - uint8, chunk `96³`, gzip compression, fill value `0`;
   - full 5-level pyramid `[1,1,1] … [16,16,16]`;
   - group attributes mirrored from the source:
     - `setup0/attributes.json`: `dataType`, `downsamplingFactors`
     - `setup0/timepoint0/attributes.json`: `multiScale`, `resolution`
       `[0.32, 0.32, 0.4]`.
5. **Upload.** `mc mirror --overwrite data/rawdata/shell_halves/ \
   EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/`.
6. **Metadata.**
   - Local XML per source in `data/platybrowser_6dpf/images/local/<name>.xml`,
     `<ImageLoader format="bdv.n5">` with
     `<n5 type="relative">../../../rawdata/shell_halves/<name>.n5</n5>`.
   - S3 XML per source in
     `data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves/<name>.xml`,
     `<ImageLoader format="bdv.n5.s3">` with
     `<Key>images/bdv-n5-s3/shell_halves/<name>.n5</Key>`,
     `<SigningRegion>us-west-2</SigningRegion>`,
     `<ServiceEndpoint>https://s3.embl.de</ServiceEndpoint>`,
     `<BucketName>platybrowser-2025</BucketName>`.
   - Four new `image` sources in `dataset.json`, each with both `bdv.n5` and
     `bdv.n5.s3` imageData, matching the existing `shell` source shape.
   - Four non-exclusive views in `uiSelectionGroup: "sbem"`, each an
     `imageDisplay` on one source with `contrastLimits: [0.0, 1.0]` and
     `name` equal to the source name.
7. **Reduce + validate + commit.** Run `2025_scripts/compress_dataset_json.py`
   (also a pre-commit hook) and `2025_scripts/validate_dataset_json.py`; commit
   on `shell_halves`.

## Deliverables

- `2025_scripts/generate_shell_halves.py` (uv inline deps: `z5py`, `numpy`;
  reuse the N5 attribute-mirroring approach from
  `2025_scripts/generate_nuclei_proba_images.py`).
- Four N5 volumes in S3 under `platybrowser-2025/images/bdv-n5-s3/shell_halves/`.
- Eight XML files (4 local + 4 S3) in the dataset tree.
- Four sources + four views in `data/platybrowser_6dpf/dataset.json`.
- This spec.

## Verification

- **S3 write probe** succeeded (create + delete a tiny object).
- `mc ls` shows all four `.n5` folders with expected object counts and
  non-trivial sizes.
- Read back each `setup0/attributes.json` and
  `setup0/timepoint0/attributes.json` from S3 and confirm `dataType=uint8`,
  `downsamplingFactors`, `multiScale`, `resolution`.
- Spot-check masking: for each output, sample voxel indices and confirm the
  zero/nonzero pattern matches the keep condition (e.g. via a small z5py read).
- Confirm the full shell volume is unchanged on S3 (same key, not rewritten).
- `compress_dataset_json.py` produces no diff beyond the intended additions;
  `validate_dataset_json.py` passes; pre-commit hook passes.
- MoBIE manual load of the four views (user-facing, final gate).

## Non-goals / out of scope

- No change to the original `shell` source or view.
- No hardcoded plane offsets; the planes are derived from the measured axes.
- No modification/deletion of any existing S3 object.
- No new dataset/version; work stays on the active `platybrowser_6dpf` dataset.

## Risks / open points

- **Runtime principal-axis computation** relies on the mask point cloud; a
  materially different mask would move the axes. The measured axes are recorded
  in `2026-09-11-shell-ovoid-orientation.md` so drift can be detected.
- **Local staging disk** ~0.5 GB in, ~2.3 GB uncompressed out (smaller with
  gzip). Staging dir is gitignored and can be removed after upload.
- **Pyramid downsampling**: each level is masked with its own global indices
  (scale-invariant for a plane through the origin-of-axes); edges are hard at
  every level, which is intended for a half-cut.
