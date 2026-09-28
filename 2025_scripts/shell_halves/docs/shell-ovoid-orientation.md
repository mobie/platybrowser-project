# Shell ovoid orientation — measured geometry

Date: 2026-09-11
Status: Reference (measurements used to define the shell-half cut planes)
Related: `2026-09-11-shell-halves-design.md`

This note records the measurements behind the half-shell cut planes, so the
choice of plane normals can be re-derived and audited later.

## Why

The first attempt cut the shell with `x = y` and `x = -y` planes through the
N5 index origin. The coronal plane `x = -y` never intersects the shell, so it
produced `front` = the whole shell and `back` = empty. Rather than guess an
offset, the shell's own principal axes were measured and the cut planes are now
defined from them.

## Data

- Source: `platybrowser/rawdata/sbem-6dpf-1-whole-segmented-shell.n5`
  (`https://s3.embl.de`), level `s0`.
- N5 array axes are `(z, y, x)`; shape `(714, 810, 860)`.
- The mask is **sparse binary**: values are `0` and `255` only
  (`2,372,162` shell voxels; `495,000,238` background). It is **not** a 0/1
  mask, which is why the raw sum of values (`604,901,310 = 255 × 2,372,162`)
  exceeds the voxel count.
- Bounding box of nonzero voxels: `x ∈ [122, 764]`, `y ∈ [71, 745]`,
  `z ∈ [0, 711]`.
- Cross-sections are closed rings: the object is a **hollow, closed ovoid**
  (a shell enclosing the animal), not a solid or an open curved surface.

## Method

1. **Principal axes.** Chunked pass over `s0` collecting the nonzero voxel
   coordinates; compute the centroid and the 3×3 covariance of the point cloud
   (coordinates in `(x, y, z)`), then eigendecompose. For a uniform ellipsoid
   the eigenvectors are the ellipsoid axes and the eigenvalues scale with the
   squared semi-axes.
2. **Symmetry-plane search.** Independently, reflect a random 300k subsample of
   shell voxels across candidate planes through the centroid with in-plane
   normals `n(θ) = (cosθ, sinθ, 0)` for `θ ∈ [0°, 180°)`, and measure the
   fraction of reflected points that land on shell voxels. The peak marks the
   bilateral-symmetry plane; its normal is the left–right axis.

## Measurements

Centroid: `c = (479.92, 438.42, 356.95)`.

| axis | eigenvalue | extent (std) | projection length | raw direction `(x, y, z)` | oriented axis (used) |
|---|---|---|---|---|---|
| **PC1 — AP / main** | 46,365 | 215.3 | 845 | `(−0.359, −0.388, 0.849)` | `(−0.359, −0.388, 0.849)` |
| **PC2 — LR** | 26,052 | 161.4 | 779 | `(−0.633, 0.770, 0.084)` | `(0.633, −0.770, −0.084)` |
| **PC3 — DV** | 9,521 | 97.6 | 377 | `(0.686, 0.507, 0.522)` | `(0.686, 0.507, 0.522)` |

Raw eigenvector signs are arbitrary (an axis is a line, not an arrow); the
"oriented axis" column is the sign-fixed version the generator actually uses,
per the conventions below.

Axis-aligned standard deviations for comparison: `σx=144.5`, `σy=157.6`,
`σz=190.3`.

Key dot products:

- `AP · ẑ = 0.849` → the main (antero-posterior) axis is **~32° off Z**,
  tilted toward `−x, −y`. The animal is *not* Z-aligned.
- `LR · (1,−1,0)/√2 = 0.993` → the left–right axis is essentially the
  `(1,−1,0)` diagonal, i.e. the user's stated bilateral-symmetry plane
  `x = y`. The symmetry-plane reflection search peaks at `θ = 131°`, normal
  `(−0.656, 0.755, 0)`, which agrees with `PC2` (dot `0.998`).
- `DV · (1,1,0)/√2 = 0.844` → the dorso-ventral (flattening) axis is tilted
  out of the XY plane, a consequence of the tilted main axis.
- Cross-section `LR : DV ≈ 161 : 98` → the ovoid is **dorso-ventrally flattened**
  (~1.6:1), as expected for a worm.

Caveat on the symmetry search: the absolute reflection overlap is low
(`0.099`), as expected for a 1-voxel-thick surface where exact mirror voxels are
rare; the reliable signal is the *location* of the peak, which matches `PC2`.

## Observed shape (max-projections and mid-slices)

Pooled ASCII maps (`@` densest, blank empty). Max projections look filled;
the mid-slices confirm the hollow ring.

```
XY projection (x right, y up) [max over z]     XY slice z=356 (ring)
       .-                                            ++#::::
       *%%=-#%=                                      % :
      #+%%%%%%%*                                     +    =- .
    .=#%%%%%%%%%=                                    -    :  .
     #%%%%%%%%%%%                                    .     :
     .#%%%%%%%%%%.                                   .   %
     +#%%%%%%%%%%:                                   :   =
    .#%%%%%%%%%%%:                                   .--.:
    .*%%%%%%%%%%%                                      ==
      #%%%%%%%%%:
      +%%%%%%%%%.
      -%%%%%%%%%
       -%%%%%%%%.
        #%%%%%%*=
        =%%%%%%:*
        :++%+%+:=
          =-.+
```

```
XZ slice y=438 (x right, z up)          YZ slice x=480 (y right, z up)
        :---                                    .---.
     .--.   :                                   :   :
    =*      :                                --:     .
      -      :                               :       ..
      ..     .                                -.      :
       :      :                                :       ::
       :       .                              .-        :
       :       :                              :         .
       -       :                              .        :
       ..      +                               :.      :
        :      #                                :       %
        :      :                                 :       :
         :     -                                  .     :
         :      -                                 :  :   :
         : =%   ..                                ..-+-  :
          ::%    :                                 : #.  :
          :-:   :                                  .::   :
            :   :                                     :  ..
            .---                                      .--:
```

The ovoid leans: in XZ the top is at smaller `x` than the bottom, matching
`PC1` having `−x` with `+z`.

## Resulting cut planes

Both planes pass through the centroid `c` and contain the main axis `PC1`:

| half | plane normal | keep condition |
|---|---|---|
| `shell_left`  | `LR = PC2` | `(p − c) · LR ≥ 0` |
| `shell_right` | `LR = PC2` | `(p − c) · LR ≤ 0` |
| `shell_front` | `DV = PC3` | `(p − c) · DV ≥ 0` |
| `shell_back`  | `DV = PC3` | `(p − c) · DV ≤ 0` |

Sign conventions (deterministic, for reproducibility): `PC1 · ẑ ≥ 0`,
`LR · (1,−1,0) ≥ 0`, `DV · (1,1,0) ≥ 0`. The `left/right` and `front/back`
labels are viewer conventions only and may be swapped by renaming.

Note the naming caveat: the second pair splits along the **dorso-ventral**
axis (the ovoid's flattened cross-axis), so `front/back` here means the two
sides of that cut, not an antero-posterior split.

## Reproduce

The measurement is deterministic and re-runnable with
`uv run --python 3.12 --with z5py --with numpy` over a local mirror of the
shell `s0`; the generator (`2025_scripts/generate_shell_halves.py`) computes the
same centroid and axes at run time from the mask.
