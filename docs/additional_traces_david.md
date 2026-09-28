# David Puga traces → N5 — handoff / context summary (2026-09)

Compressed context for a future session. Goal: convert **all of David Puga's
nmx tracings** into a PlatyBrowser-compatible N5 segmentation, with a
per-trace provenance table, and expose it in the 6dpf MoBIE project.

## 1. Where things live

| Artifact | Location |
|---|---|
| N5 + XML + mapping + this doc | `/g/arendt/David_Puga_Consolidated_data/processed/n5/` (cluster) |
| S3 copy of the N5 | `EmblArendtS3/platybrowser-2025/demo-v0/david_all_traces.n5` |
| Repo branch | `add-david-all-traces` (worktree `~/platybrowser-project-2025/.worktrees/add-david-all-traces`, base `main`) |
| MoBIE source | `data/platybrowser_6dpf/…` source `david_all_traces`, view "David all traces" (`uiSelectionGroup: traces`) |
| Trace library (inputs) | `/g/arendt/David_Puga_Consolidated_data/processed/traces/` (2,025 readable nmx) |
| Original code/env (read-only) | `/g/arendt/EM_6dpf_segmentation/platy-browser-data/` (mmpb + `software/conda`) |
| Docker env (for the pipeline) | `~/transfer_check/platybrowser-docker-release/` (image + sources + tests) |

## 2. What was produced

- **284 traces** (one per physical cell), labels **1–284**, int16.
- Grid = the **nuclei/traces grid**: array (z,y,x) = (2854, 3240, 3438),
  voxel **100/80/80 nm** (resolution attr `[0.08, 0.08, 0.1]` µm), chunks 96³.
- Pyramid **s0–s7** (4×2×2×2… 7 downsamplings, max-pool) — same depth and
  shapes as `nuclei`/`cells`.
- Started from **2,025 readable nmx** (1 corrupt skipped: `Nk6.473.nmx`).
  Fingerprint dedupe (first node = identity) collapsed autosave chains/session
  copies into **298 distinct cells**; then **14 degenerate stubs
  (n_points ≤ 10)** were dropped → 284 (they cannot be meshed by MoBIE).

## 3. Finished / unfinished, by type

"Finished" = member of David's finished commissural set (`in_finished167`).
"Present" = already in PlatyBrowser before this work (`in_platybrowser`).

| type (comment) | n | finished | already in dataset | nucleus clear | nucleus far |
|---|---|---|---|---|---|
| commissural (finished group) | 166 | 166 | 166 | 152 | 0 |
| non-commissural: big-chain extra cell | 61 | 0 | 3 | 44 | 8 |
| non-commissural: p0 / V0 class | 17 | 0 | 0 | 14 | 2 |
| non-commissural: Dbx neuron chain | 12 | 0 | 0 | 11 | 0 |
| non-commissural: sim_hox4 (TF-biased) | 10 | 0 | 0 | 7 | 3 |
| non-commissural: Hb9 motoneuron / 2_Motoneurons | 6 | 0 | 0 | 1 | 3 |
| DP-only: comm_sec_seg all chain | 4 | 0 | 0 | 4 | 0 |
| non-commissural: eve (paper) | 3 | 0 | 0 | 0 | 3 |
| DP-only: bilateral counterpart | 2 | 0 | 0 | 2 | 0 |
| training/test | 1 | 0 | 0 | 1 | 0 |
| DP-only: Mouse tracings | 1 | 0 | 0 | 1 | 0 |
| non-commissural: Pyg | 1 | 0 | 0 | 0 | 1 |
| **total** | **284** | **166** | **169** | **237** | **20** |

- **Unfinished / other sets** = everything except the 166 finished commissural
  cells (118 traces): the extra cells traced inside the big chains
  (`Nk6_all`, `Commissures V2`, `Remaining_comm…`), p0/V0, Dbx, Hb9, sim_hox4,
  eve/Pyg, DP-only continuation/bilateral/mouse, training/test.
- **"Mouse" is NOT another species**: those files are `<experiment name="Platy1607"/>`
  (same EM volume, same tracer). They are Platynereis neurons kept for David's
  cross-species comparison (`…/comparison vert droso/…/mouse/`). All seeds fall
  on the volume's own cell segmentation.

## 4. Which traces have a clear nucleus

Measured with a KD-tree over the **11,497 6dpf nuclei anchors** (µm),
distance from the **closest node anywhere on the trace**:

- David's own 166 finished traces calibrate the range:
  **p50 0.88 µm, p90 2.15 µm, max 4.86 µm**.
- New/other traces: **237 clear (≤2.15 µm)**, 27 plausible (2.15–4.86), 20 far (>4.86).
- **No nucleus within range** (soma nucleus not in the 11,497 set, or simulated/test):
  Hb9 motoneurons (3/6 far), eve (3/3), Pyg (1), sim_hox4 (3/10),
  some big-chain extras (8/61), p0/V0 (2/17).
- Extremity-only distances are looser (David: p50 3.06 / p90 4.79 / max 7.30 µm);
  the any-node test is the reliable one. `closest_nucleus_id` +
  `min_dist_nucleus_um` are recorded per label in `origin_mapping.tsv`.

## 5. Methods (as run)

1. **Environment** — the original py3.7 stack (elf 0.2.2, pybdv 0.4.1, nifty,
   z5py, skimage 0.16) from `platybrowser-new`, incl. the four `pip -e` dev
   checkouts (`elf`, `pybdv`, `mobie-utils-python`, `cluster_tools`) that live
   outside the env. Backed up self-contained (`/g/arendt/Cyril/backups/…`),
   dockerized (release folder + 6-check test suite), and used for the conversion.
2. **Grid conventions** (side-by-side):
   - raw EM: 2 TB, uint8, full-res grid 20/20/25 nm (no s0; s1 = full res)
   - cells: uint64, 20/20/25 nm (s0), 256·256·32 chunks
   - nuclei & traces: **same lower-res grid** (2854,3240,3438), 80/80/100 nm,
     128³ / 96³ chunks → traces were written to match nuclei exactly.
3. **Parsing** — nmx = zip of nml; each `<thing>` = one skeleton; nodes
   `[z,y,x]` in **nm**. Identity = **first-node fingerprint** (stable across
   autosave snapshots); keep the most complete instance per cell.
4. **Rasterization** — per node, a radius-2 disk (`skimage.draw.circle`) on its
   z-plane, written into an int16 N5; labels renumbered 1..N.
5. **Pyramid** — custom boundary-safe 2×2×2 **max-pool** downsampler
   (replaced `pybdv.converter.make_scales`, which crashes on odd volume edges);
   7 levels to match nuclei; BDV XML + N5 multiscale metadata (`write_n5_metadata`).
6. **Nucleus proximity** — `scipy.spatial.cKDTree` over nuclei anchors; per
   trace the minimum distance over all nodes + nearest nucleus id.
7. **MoBIE integration** — new source `david_all_traces` (local `bdv.n5` +
   `bdv.n5.s3`), `default.tsv`, `origin_mapping.tsv`, additive view.
8. **Git** — worktree branch off `main`, pushed to `origin` (HTTPS + `gh` token;
   the machine has no GitHub SSH key configured).

## 6. Table schema

`tables/david_all_traces/default.tsv` (MoBIE):
`label_id, anchor_x/y/z, bb_min_x/y/z, bb_max_x/y/z, n_points, origin_dir,
in_platybrowser` (True/False).

`tables/david_all_traces/origin_mapping.tsv` (provenance):
`label_id, aka, comment, source_file, first_node_zyx_nm, n_nodes,
nmx_in_traces, nmx_in_data, cell_group_dir, min_dist_nucleus_um,
closest_nucleus_id, in_platybrowser`
- `aka` = David's original numbering (e.g. `6.002`, `0.07`)
- `comment` = neuron type/group (see table above; mouse/others generic)
- `source_file` = nmx path with `/` → `_`

## 7. Bugs hit & fixes (watch for these again)

- **`sed` over `lib/` corrupted `.so`** (rewrote embedded paths) → segfaults.
  Only ever rewrite shebangs + `*.egg-link`/`*.pth`, never binaries.
- **`pybdv` 0.4.1 `make_scales` boundary bug** on odd volume sizes → replaced
  with our own block-max pyramid.
- **Single-node traces** → MoBIE "Could not create mesh"; filtered n_points ≤ 10.
- **`default.tsv` bb columns had x/z swapped** (generator wrote `[z,y,x]` into
  `x/y/z`) → MoBIE read empty crops; fixed.
- Dev `elf`/`pybdv` APIs differ from released: pybdv uses `timepoint`,
  `elf.skeleton.io.read_nml` returns `(coords, extras)` tuples.

## 8. Open items / next steps

- **`comment` curation**: mouse/others rows are generic; refine as you inspect.
- **PR**: branch pushed; PR not opened (link in chat history).
- **PlatyBrowser schema wiring**: `cell_id`/`nucleus_id` columns for new traces
  can be derived from `closest_nucleus_id` (via `cells_to_nuclei`) where clear.
- **Overlap with existing source**: 169 flagged `True` via anchor match
  (tolerance 0.3 µm) — verify no double-labelling if merged with `traces`/`traces_MN_David_Puga`.
- **Optional**: register the S3 XML for `david_all_traces` (already done),
  upload/replace if the N5 changes; consider per-file/per-thing labelling mode.
- **Visual QC**: MoBIE view renders; spot-check the 20 "far" traces.

## 9. Scripts (in `~/transfer_check/nmx2n5/`)

`convert_all_david_traces.py` (full conversion; `MIN_NODES`, `N_SCALES` env),
`append_trace_scales.py`, `trace_extremity_proximity.py`,
`trace_anynode_proximity.py`, `convert_nmx_to_n5.py` (single-folder),
`david_all_traces.xml`. Cluster copies under `/scratch/cros/phase1/`.

---

## Update — MoBIE views added (by neuron type)

New menu group **`additional_traces`** with 12 additive views (each shows only its
traces via `opacityNotSelected=0`; where a nucleus was found the paired
**nucleus + its cell** are selected too):

| view | traces | paired nuclei (unique) |
|---|---|---|
| additional traces: commissural (finished) | 166 | 106 |
| additional traces: big-chain extras | 61 | 41 |
| additional traces: p0/V0 | 17 | 15 |
| additional traces: Dbx | 12 | 10 |
| additional traces: sim hox4 | 10 | 6 |
| additional traces: Hb9 motoneurons | 6 | 3 |
| additional traces: commissural 2nd-segment | 4 | 4 |
| additional traces: eve | 3 | 0 |
| additional traces: bilateral counterparts | 2 | 2 |
| additional traces: training/test | 1 | 1 |
| additional traces: mouse set | 1 | 1 |
| additional traces: Pyg | 1 | 0 |

Pairing rule: `min_dist_nucleus_um <= 4.86` (David's own worst case) and a valid
`closest_nucleus_id`; cell = the cell whose `nucleus_id` equals it.

`origin_mapping.tsv` gained columns: **`neuron_type`** (stable type for grouping),
**`overlaps_existing`** (True/False) and **`overlap_with`** (existing
PlatyBrowser trace ids). 10 new traces are flagged as likely continuations of an
existing trace; their `comment` reads
"may be a continuation of PlatyBrowser trace(s) <ids>". The 2nd-segment
continuations (168.2-171.2) are new cells (not in the existing table; 13-22 um
from any existing start) and are typed `commissural 2nd-segment`.
