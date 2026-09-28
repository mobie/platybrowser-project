# Shell halves and foregut lumen

Scripts and notes for the four half-shell image sources and the foregut-lumen
source in the `platybrowser_6dpf` dataset.

## What this produces

The original `shell` source is a thin, hollow ovoid around the animal. It blocks
the 3D view, so it is split into four halves by two planes through the shell
centroid, and the near-axis "foregut lumen" protrusion is separated out:

| Source | What it is |
|---|---|
| `shell_left_half`   | shell kept on the left of the left/right plane, **foregut lumen zeroed** |
| `shell_right_half`  | shell kept on the right of the left/right plane, foregut lumen zeroed |
| `shell_front_half`  | shell kept on the front of the front/back plane, foregut lumen zeroed |
| `shell_back_half`   | shell kept on the back of the front/back plane, foregut lumen zeroed |
| `shell_foregut_lumen` | the near-axis foregut-lumen mass (removed from the halves above) |

Views (all in the `sbem` selection group, display identical to the plain `shell`
view — `contrastLimits [0,1]`, no colour/opacity override, no camera transform):

`shell_left_half_faint_dark_mesh`, `shell_right_half_faint_dark_mesh`,
`shell_front_half_faint_dark_mesh`, `shell_back_half_faint_dark_mesh`,
`shell_foregut_lumen_faint_dark_mesh`.

## Geometry

The cut planes pass through the shell centroid and contain the shell's main
(antero-posterior) axis; their normals are the measured `LR` and `DV` axes from
a PCA of the shell voxel cloud. The foregut lumen is selected as a 25 µm-radius
cylinder along the segment mouth `(157.0, 152.2, 47.1)` µm → back
`(176.5, 157.9, 141.7)` µm (plus its mirror), then connected-component filtered
to keep only components within 150 voxels (`|x − y|`) of the sagittal plane —
this drops the disjoint lateral parapodia/antennae. See
`docs/shell-ovoid-orientation.md` for the measurements and
`docs/shell-halves-design.md` for the design.

## Pipeline

1. `generate_shell_halves.py` — cuts the shell into the four halves and writes
   BDV N5 pyramids plus the local + S3 XMLs.
2. `make_foregut_lumen.py` — extracts the foregut lumen as its own N5 + XMLs.
3. `subtract_mask.py` — zeroes the foregut lumen from each half (`out = in AND
   NOT mask`, level by level).

All scripts use `uv` inline dependencies (`numpy`, `z5py`, `scipy`) and mirror
the source pyramid (levels, shapes, chunks, group/dataset attributes; uint8,
gzip, fillvalue 0).

### Example

```bash
# 1. halves (from a local mirror of the shell N5)
uv run --python 3.12 generate_shell_halves.py \
  --mask <shell.n5> --stage-dir data/rawdata/shell_halves \
  --local-xml-dir data/platybrowser_6dpf/images/local \
  --s3-xml-dir data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves

# 2. foregut lumen
uv run --python 3.12 make_foregut_lumen.py \
  --shell-mask <shell.n5> \
  --out-n5 data/rawdata/shell_halves/shell_foregut_lumen.n5 \
  --local-xml-dir data/platybrowser_6dpf/images/local \
  --s3-xml-dir data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves \
  --segment-p0 157.0,152.2,47.1 --segment-p1 176.5,157.9,141.7 \
  --radius 25 --mirror --keep-near-plane 150

# 3. zero the lumen out of each half
for h in shell_left_half shell_right_half shell_front_half shell_back_half; do
  uv run --python 3.12 subtract_mask.py \
    --in-n5 data/rawdata/shell_halves/$h.n5 \
    --mask data/rawdata/shell_halves/shell_foregut_lumen.n5 \
    --out-n5 data/rawdata/shell_halves/${h}_tmp.n5
  mv data/rawdata/shell_halves/${h}_tmp.n5 data/rawdata/shell_halves/$h.n5
done
```

## Data location

The N5 volumes are **not** in git (`.gitignore` ignores `*.n5`). They live on
S3 at `platybrowser-2025/images/bdv-n5-s3/shell_halves/`:

```
shell_left_half.n5  shell_right_half.n5  shell_front_half.n5
shell_back_half.n5  shell_foregut_lumen.n5
```

Upload with `mc cp --recursive <dir>/<name>.n5 <alias>/platybrowser-2025/images/bdv-n5-s3/shell_halves/`
(do **not** rely on `mc mirror --overwrite`: it silently skips objects whose
name and size already match, which can leave stale data in place).

## Tests

```bash
uv run --python 3.12 --with pytest --with z5py --with numpy --with scipy \
  python -m pytest test_generate_shell_halves.py -v
```

## Docs

- `docs/shell-halves-design.md` — design/spec.
- `docs/shell-ovoid-orientation.md` — measured shell orientation and cut planes.
- `docs/shell-halves-plan.md` — original implementation plan (historical).
