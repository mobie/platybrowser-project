# 2026 Views Curated

Canonical source JSONs for the curated views added to `dataset.json` (2026
work). Each file is synced to a view in `dataset.json`; edit the JSON here,
then sync (see AGENTS.md "Syncing handcrafted views").

## Contents

55 curated views, all present in `dataset.json`:

| File | dataset.json view | Group |
|---|---|---|
| `clade*__*_detlev.json` (26 files) | `clade*__*_detlev` | `curated_views` (naked) |
| `clade*__*_illustrated_detlev.json` (23 files) | `clade*__*_illustrated_detlev` | `curated_views_probes_on` |
| `clade*_sam.json` (6 files) | `clade*_sam` | `curated_views` (naked, Sam) |

Producer suffix: `_detlev` (Detlev Arendt) / `_sam` (Sam).

## Unassigned / questionable views

These exist as source JSONs in
`2025_scripts/detlev_handcrafted_views_questionable/` and are exposed in
`dataset.json` under the `status_unclear_curated_views` group. They have no
clear cell-type assignment yet.

| View (dataset.json) | Source file | Cells | Markers | What it could be | Problem |
|---|---|---|---|---|---|
| `brain_5HT_IN_Phox2_Vsx` | `brain_5HT_IN_Phox2_Vsx.json` | 87 | sert, trph, chx10, hnf6, pitxb, phox2b | Serotonergic interneuron | Not in masterlist at all; earlier cross-reference matched it to `brain_NA_IN_Phox2_Isl_Coe` (5HT vs NA — likely different cell types) |
| `brain_ACh_MN_mnx_phox2` | `brain_ACh_MN_mnx_phox2.json` | 36 | hb9, phox2b, chat | Cholinergic motorneuron | Masterlist has the old-name twice (N_msom03/clade11sub37 and N_msom04/clade11sub66) — ambiguous which |
| `brain_deepbrainPRC_rx_foxq2` | `brain_deepbrainPRC_rx_foxq2.json` | 2 | — | Deep-brain PRC photoreceptor? | Only 2 cells; no masterlist match |
| `brain_PC` | `brain_LLE-PC.json` (legacy name) | 2 | fezf, nk21, ap2, dlx | Photoreceptor cell? | Only 2 cells; no masterlist match; file name differs from view key |
| `david_cells` | `davids_cells.json` (legacy name) | 132 | — | David's curated cell set | 132 cells but no markers and no masterlist match; file name differs from view key |
| `dbx_traced_cells` | `dbx_traced_cells.json` | 4 | — | Dbx-traced cells | No masterlist match |
| `fg_GABA_MN_lmx1_sim_mnx` | `fg_GABA_MN_lmx1_sim_mnx.json` | 8 | lmx1, hb9, sim1 | Foregut GABA MN | No masterlist match; overlaps neither N_mpro03 view nor the old dubious set |
| `traced_evx` | `traced_evx.json` | 4 | — | Evx-traced cells | No masterlist match |

### Masterlist entries with no view yet

| Masterlist old-name | Family / subclade | Problem |
|---|---|---|
| `brain_GABA_IN_Dbx_Ptf1a` | N_ipro04 / clade11sub14subsub9 | No curated view exists |
| `ls_pyg_Glu_cIN_Evx_Pou4` | N_spro02 / clade11sub13subsub6 | No curated view exists |

### Old per-person trace names

`sbem-6dpf-1-whole-traces` and `sbem-6dpf-1-whole-traces-MNs-David-Puga`
still exist as data sources but their IDs are ambiguous; the canonical source
is `sbem-6dpf-1-whole-combined-traces` (see AGENTS.md "Trace ID
conventions"). No views reference the old per-person sources directly.
