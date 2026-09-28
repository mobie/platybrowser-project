# Nuclei Probability Dropdown — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 274 per-subtype nuclei probability N5 images (nuclei mask repainted with `round(p × 1000)`, pyramid mirroring the nuclei N5), wire 274 image sources + additive views into `dataset.json` under the `nuclei_probabilities` dropdown, pilot 5 subtypes first with a file-size report, then full run + S3 upload.

**Architecture:** A uv-inline generator script (`2025_scripts/generate_nuclei_proba_images.py`) reads the nuclei segmentation N5 block-wise, introspects its pyramid (levels, shapes, chunks, attributes), and for every subtype column of the detailed cluster-probability table writes a uint16 N5 whose voxels are `round(p × 1000)` relabeled through the mask. Outputs use gzip + `fillvalue 0` so they are sparse (mostly black). A second script (`2025_scripts/add_proba_sources_and_views.py`) wires sources + concise views into `dataset.json`. The existing `upload_Alyona_local_n5_to_s3.py` converts/upload to S3 under `images/bdv-n5-s3/celltype_proba/`.

**Tech Stack:** Python 3 (uv inline-script metadata, like existing `2025_scripts/`), numpy, z5py (N5 I/O), stdlib `csv`/`xml.etree`/`json`, unittest (repo test style).

## Global Constraints

- Work on branch `proba_as_dropdown`. Before editing `data/platybrowser_6dpf/dataset.json`, run `git pull --rebase` (MoBIE may have pushed view changes).
- Only modify: `2025_scripts/`, `data/platybrowser_6dpf/dataset.json`, `data/platybrowser_6dpf/images/local/` (new XMLs), `data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/` (new XMLs). All legacy dirs (`segmentation/`, `registration/`, `mmpb/`, `analysis/`, `misc/`, `software/`, `data/0.0.0`–`1.0.1`) are read-only. Never modify raw image data on S3 — only add under the new `celltype_proba` prefix.
- Subtype columns = every header cell of `data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv` starting with `clade` or `nocladesub` (currently 274; excludes `label_id`, `zero`, `autofluorescence`, `most probable cluster`).
- Value encoding: uint16, `round(p × 1000)` ∈ [0, 1000] (0.001 precision), label 0 / background = 0, missing label = 0.
- Output N5 must mirror the mask's pyramid **exactly**: same level names (`s0`, `s1`, … present in the mask), same per-level shape, chunks, and `resolution`/`downsamplingFactors`/`offset` attributes; dtype uint16; gzip compression + `fillvalue 0`. **Group-level attributes must also be mirrored** — `setup0/attributes.json` (`dataType`, nested per-level `downsamplingFactors`) and `setup0/timepoint0/attributes.json` (`multiScale`, `resolution`) — because MoBIE's `N5S3ImageLoader` reads `dataType` from `setup0/attributes.json` and NPEs when it is missing.
- Naming: source/view name `{subtype}_proba` (e.g. `clade1sub1_proba`); uiSelectionGroup `nuclei_probabilities`.
- View shape (concise — every key not listed here is omitted): `{"uiSelectionGroup": "nuclei_probabilities", "sourceDisplays": [{"imageDisplay": {"sources": ["X_proba"], "contrastLimits": [0.0, 1000.0], "name": "X_proba"}}]}`. No `isExclusive`, no `viewerTransform`, no `raw`, no `color`.
- Source shape: `{"image": {"imageData": {"bdv.n5": {"relativePath": "images/local/X_proba.xml"}, "bdv.n5.s3": {"relativePath": "images/bdv-n5-s3/celltype_proba/X_proba.xml"}}}}`.
- Staging dir for generated N5s: `tmp_celltype_proba/` (matches `tmp*` and `*.n5` in `.gitignore` — never committed).
- Tests: unittest (repo style, `sys.path.insert(0, parent)`), run from repo root with `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py"`.
- Pre-commit hook runs `2025_scripts/compress_dataset_json.py` (auto-strips defaults) + `2025_scripts/validate_dataset_json.py`; CI enforces on `main`.
- `*.n5` is gitignored: N5 files are never committed. Only XMLs + `dataset.json` + scripts + tests are committed.
- The nuclei mask N5 is NOT present at `data/0.0.0/images/local/` in this checkout — the user must provide its path (`--mask`) before the pilot/full runs.

---

### Task 1: Generator — table reading and value tables

**Files:**
- Create: `2025_scripts/generate_nuclei_proba_images.py`
- Test: `2025_scripts/tests/test_generate_nuclei_proba_images.py`

**Interfaces:**
- Produces:
  - `read_subtype_columns(tsv_path: str | Path) -> list[str]` — subtype columns in table order
  - `read_probabilities(tsv_path: str | Path, subtype: str) -> dict[int, float]`
  - `build_value_table(probs: dict[int, float], scale: int = 1000) -> np.ndarray` (uint16, indexed by label, `[0] == 0`)

- [ ] **Step 1: Write the failing tests**

```python
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from generate_nuclei_proba_images import (
    read_subtype_columns,
    read_probabilities,
    build_value_table,
)


def make_fixture_tsv(path: Path) -> Path:
    path.write_text(
        "label_id\tclade1sub1\tclade6sub19\tnocladesub3\tzero\tautofluorescence\tmost probable cluster\n"
        "1.0\t0.95\t0.0003\t0.0\t0.0\t0.0\tclade1sub1\n"
        "2.0\t0.4996\t0.9996\t0.5\t0.0\t0.0\tclade6sub19\n"
        "3.0\t0.0\t0.0\t1.0\t0.0\t0.0\tnocladesub3\n",
        encoding="utf-8",
    )
    return path


class TestReadSubtypeColumns(unittest.TestCase):
    def test_returns_only_clade_and_nocladesub_columns(self):
        with tempfile.TemporaryDirectory() as d:
            tsv = make_fixture_tsv(Path(d) / "proba.tsv")
            cols = read_subtype_columns(tsv)
            self.assertEqual(cols, ["clade1sub1", "clade6sub19", "nocladesub3"])


class TestReadProbabilities(unittest.TestCase):
    def test_returns_label_to_float(self):
        with tempfile.TemporaryDirectory() as d:
            tsv = make_fixture_tsv(Path(d) / "proba.tsv")
            probs = read_probabilities(tsv, "clade1sub1")
            self.assertEqual(probs[1], 0.95)
            self.assertEqual(probs[3], 0.0)


class TestBuildValueTable(unittest.TestCase):
    def test_rounds_to_0_001_and_maps_zero_for_missing_and_background(self):
        table = build_value_table({1: 0.95, 2: 0.4996, 3: 0.0003})
        self.assertEqual(table.dtype, np.uint16)
        self.assertEqual(table[0], 0)      # background
        self.assertEqual(table[1], 950)    # round(0.95 * 1000)
        self.assertEqual(table[2], 500)    # round(0.4996 * 1000)
        self.assertEqual(table[3], 0)      # round(0.0003 * 1000) -> 0
        self.assertEqual(len(table), 4)    # max_label + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_nuclei_proba_images'`

- [ ] **Step 3: Write the script skeleton + the three functions**

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Generate per-subtype nuclei probability N5 images.

For each detailed cell type (subtype) column in the detailed cluster
probability table, repaint the nuclei segmentation mask: every nucleus voxel
keeps its geometry but its value becomes round(p * 1000) for that subtype.
Background stays 0. Output N5s mirror the mask's pyramid exactly (levels,
shapes, chunking, resolution/downsamplingFactors/offset attributes) and use
gzip + fillvalue 0 so images are sparse and small.

Usage:
    ./generate_nuclei_proba_images.py --mask <nuclei.n5> --table <tsv> \
        --stage-dir <dir> --local-xml-dir <dir> [--subtypes a,b,c]
"""

import argparse
import csv
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import z5py

SUBSTYPE_PREFIXES = ("clade", "nocladesub")
EXCLUDED_COLUMNS = {"label_id", "zero", "autofluorescence", "most probable cluster"}
SCALE = 1000
DEFAULT_XML_TEMPLATE = Path(__file__).resolve().parent.parent / \
    "data" / "platybrowser_6dpf" / "images" / "local" / \
    "sbem-6dpf-1-whole-segmented-nuclei.xml"


def read_subtype_columns(tsv_path):
    """Return subtype probability columns in table order (clade*/nocladesub* only)."""
    with open(tsv_path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f, delimiter="\t"))
    return [c for c in header
            if c.startswith(SUBSTYPE_PREFIXES) and c not in EXCLUDED_COLUMNS]


def read_probabilities(tsv_path, subtype):
    """Return {label_id: probability} for one subtype column."""
    probs = {}
    with open(tsv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label = int(float(row["label_id"]))
            probs[label] = float(row[subtype])
    return probs


def build_value_table(probs, scale=SCALE):
    """uint16 lookup table indexed by label: value = round(p * scale), label 0 -> 0."""
    max_label = max(probs.keys(), default=0)
    table = np.zeros(max_label + 1, dtype=np.uint16)
    for label, p in probs.items():
        table[label] = int(round(p * scale))
    return table


def main():
    print("generate_nuclei_proba_images: table helpers loaded", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_nuclei_proba_images.py 2025_scripts/tests/test_generate_nuclei_proba_images.py
git commit -m "Add probability table helpers for nuclei proba images"
```

---

### Task 2: Generator — mask introspection and relabel block

**Files:**
- Modify: `2025_scripts/generate_nuclei_proba_images.py`
- Test: `2025_scripts/tests/test_generate_nuclei_proba_images.py`

**Interfaces:**
- Consumes: `build_value_table` (Task 1)
- Produces:
  - `mirror_level_info(mask_path: str | Path) -> list[dict]` — one dict per `setup0/timepoint0/s{k}` level present in the mask: `{"name": str, "shape": tuple[int], "chunks": tuple[int], "attrs": dict}`
  - `relabel_block(mask_block: np.ndarray, value_table: np.ndarray) -> np.ndarray` (uint16, same shape; labels beyond table length → 0)

- [ ] **Step 1: Write the failing tests**

```python
    def test_mirror_level_info_reads_pyramid(self):
        with tempfile.TemporaryDirectory() as d:
            mask = make_mask_n5(Path(d) / "mask.n5")
            levels = mirror_level_info(mask)
            self.assertEqual([lv["name"] for lv in levels], ["s0", "s1"])
            self.assertEqual(levels[0]["shape"], (16, 16, 16))
            self.assertEqual(levels[0]["chunks"], (8, 8, 8))
            self.assertEqual(levels[1]["shape"], (8, 8, 8))
            self.assertEqual(levels[1]["attrs"]["downsamplingFactors"], [2, 2, 2])

    def test_relabel_block(self):
        mask_block = np.zeros((2, 2, 2), dtype=np.uint32)
        mask_block[0, 0, 0] = 1
        mask_block[0, 0, 1] = 2
        mask_block[1, 1, 1] = 99  # not in table -> 0
        table = build_value_table({1: 0.95, 2: 0.4996})
        out = relabel_block(mask_block, table)
        self.assertEqual(out.dtype, np.uint16)
        self.assertEqual(int(out[0, 0, 0]), 950)
        self.assertEqual(int(out[0, 0, 1]), 500)
        self.assertEqual(int(out[1, 1, 1]), 0)
```

Also add the shared fixture `make_mask_n5` at module level (used by Task 3 tests too):

```python
def make_mask_n5(path: Path) -> Path:
    """Minimal 2-level nuclei-like mask: s0 16^3, s1 8^3, labels {1, 2}."""
    import z5py
    with z5py.File(str(path), "a") as f:
        g = f.create_group("setup0").create_group("timepoint0")
        s0 = g.create_dataset("s0", shape=(16, 16, 16), chunks=(8, 8, 8),
                              dtype="uint32", compression="gzip", fillvalue=0)
        s0.attrs["resolution"] = [0.08, 0.08, 0.1]
        s0.attrs["downsamplingFactors"] = [1, 1, 1]
        s0.attrs["offset"] = [0.0, 0.0, 0.0]
        a0 = np.zeros((16, 16, 16), dtype=np.uint32)
        a0[2:5, 2:5, 2:5] = 1
        a0[10:13, 10:13, 10:13] = 2
        s0[...] = a0
        s1 = g.create_dataset("s1", shape=(8, 8, 8), chunks=(8, 8, 8),
                              dtype="uint32", compression="gzip", fillvalue=0)
        s1.attrs["resolution"] = [0.08, 0.08, 0.1]
        s1.attrs["downsamplingFactors"] = [2, 2, 2]
        s1.attrs["offset"] = [0.0, 0.0, 0.0]
        a1 = np.zeros((8, 8, 8), dtype=np.uint32)
        a1[1:3, 1:3, 1:3] = 1
        a1[5:7, 5:7, 5:7] = 2
        s1[...] = a1
    return path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: FAIL with `ImportError: cannot import name 'mirror_level_info'` (and `relabel_block`)

- [ ] **Step 3: Implement the two functions** (append after `build_value_table`)

```python
def mirror_level_info(mask_path):
    """Introspect the mask pyramid: level name, shape, chunks, attrs per s-level."""
    with z5py.File(str(mask_path), "r") as f:
        tp = f["setup0/timepoint0"]
        levels = []
        for key in sorted(tp.keys()):
            ds = tp[key]
            levels.append({
                "name": key,
                "shape": tuple(ds.shape),
                "chunks": tuple(ds.chunks),
                "attrs": dict(ds.attrs),
            })
    return levels


def relabel_block(mask_block, value_table):
    """Map a uint32 label block through the value table (labels beyond table -> 0)."""
    out = np.zeros(mask_block.shape, dtype=np.uint16)
    idx = mask_block <= value_table.shape[0] - 1
    out[idx] = value_table[mask_block[idx].astype(np.intp)]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_nuclei_proba_images.py 2025_scripts/tests/test_generate_nuclei_proba_images.py
git commit -m "Add mask pyramid introspection and relabel block"
```

---

### Task 3: Generator — N5 writer mirroring the mask pyramid

**Files:**
- Modify: `2025_scripts/generate_nuclei_proba_images.py`
- Test: `2025_scripts/tests/test_generate_nuclei_proba_images.py`

**Interfaces:**
- Consumes: `mirror_level_info`, `relabel_block`, `build_value_table` (Tasks 1–2)
- Produces: `mirror_group_attrs(mask_path) -> dict` (`{"setup0": {...attrs}, "timepoint0": {...attrs}}`) and `write_outputs(mask_path: str | Path, value_tables: dict[str, np.ndarray], stage_dir: Path, levels: list[dict], group_attrs: dict) -> list[Path]` — writes `{subtype}_proba.n5` per subtype, mirroring the mask's per-level AND group-level attributes; block-wise single mask pass; skips all-zero blocks (fillvalue covers them); returns written paths

- [ ] **Step 1: Write the failing tests**

```python
class TestWriteOutputs(unittest.TestCase):
    def test_writes_pyramid_mirroring_mask_with_relabeled_values(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask = make_mask_n5(d / "mask.n5")
            levels = mirror_level_info(mask)
            value_tables = {
                "clade1sub1": build_value_table({1: 0.95, 2: 0.4996}),
                "nocladesub3": build_value_table({1: 0.0, 2: 1.0}),
            }
            stage = d / "out"
            written = write_outputs(mask, value_tables, stage, levels)
            self.assertEqual(len(written), 2)
            self.assertTrue((stage / "clade1sub1_proba.n5").is_dir())

            with z5py.File(str(stage / "clade1sub1_proba.n5"), "r") as f:
                s0 = f["setup0/timepoint0/s0"]
                self.assertEqual(s0.shape, (16, 16, 16))
                self.assertEqual(s0.dtype, np.uint16)
                self.assertEqual(tuple(s0.chunks), (8, 8, 8))
                self.assertEqual(s0.attrs["resolution"], [0.08, 0.08, 0.1])
                self.assertEqual(s0.attrs["downsamplingFactors"], [1, 1, 1])
                a0 = s0[...]
                self.assertEqual(int(a0[3, 3, 3]), 950)      # label 1 -> 0.95
                self.assertEqual(int(a0[11, 11, 11]), 500)   # label 2 -> 0.4996
                self.assertEqual(int(a0[0, 0, 0]), 0)        # background
                s1 = f["setup0/timepoint0/s1"]
                self.assertEqual(s1.shape, (8, 8, 8))
                self.assertEqual(s1.attrs["downsamplingFactors"], [2, 2, 2])
                a1 = s1[...]
                self.assertEqual(int(a1[2, 2, 2]), 950)      # relabeled s1
                self.assertEqual(int(a1[6, 6, 6]), 500)

            with z5py.File(str(stage / "nocladesub3_proba.n5"), "r") as f:
                a0 = f["setup0/timepoint0/s0"][...]
                self.assertEqual(int(a0[3, 3, 3]), 0)
                self.assertEqual(int(a0[11, 11, 11]), 1000)  # label 2 -> 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest 2025_scripts/tests/test_generate_nuclei_proba_images.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_outputs'`

- [ ] **Step 3: Implement `write_outputs`** (append after `relabel_block`)

```python
def write_outputs(mask_path, value_tables, stage_dir, levels, group_attrs):
    """Write one uint16 N5 per subtype into stage_dir, mirroring the mask pyramid.

    Mirrors per-level AND group-level (setup0/timepoint0) attributes. Single
    block-wise pass over each mask level. All-zero blocks are skipped; the N5
    fillvalue 0 covers them, keeping the outputs sparse and small.
    """
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for subtype in value_tables:
        out_path = stage_dir / f"{subtype}_proba.n5"
        if out_path.exists():
            shutil.rmtree(out_path)
        with z5py.File(str(out_path), "a") as f:
            g = f.create_group("setup0").create_group("timepoint0")
            for level in levels:
                ds = g.create_dataset(
                    level["name"],
                    shape=level["shape"],
                    chunks=level["chunks"],
                    dtype="uint16",
                    compression="gzip",
                    fillvalue=0,
                )
                for k, v in level["attrs"].items():
                    ds.attrs[k] = v
        written.append(out_path)

    with z5py.File(str(mask_path), "r") as mask_f:
        tp = mask_f["setup0/timepoint0"]
        for level in levels:
            name, shape, chunks = level["name"], level["shape"], level["chunks"]
            mask_ds = tp[name]
            out_dss = {
                subtype: z5py.File(str(stage_dir / f"{subtype}_proba.n5"), "a")[
                    "setup0/timepoint0"][name]
                for subtype in value_tables
            }
            for z0 in range(0, shape[0], chunks[0]):
                for y0 in range(0, shape[1], chunks[1]):
                    for x0 in range(0, shape[2], chunks[2]):
                        sl = (slice(z0, min(z0 + chunks[0], shape[0])),
                              slice(y0, min(y0 + chunks[1], shape[1])),
                              slice(x0, min(x0 + chunks[2], shape[2])))
                        block = mask_ds[sl]
                        if not block.any():
                            continue
                        for subtype, value_table in value_tables.items():
                            out_dss[subtype][sl] = relabel_block(block, value_table)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_nuclei_proba_images.py 2025_scripts/tests/test_generate_nuclei_proba_images.py
git commit -m "Add sparse N5 writer mirroring the mask pyramid"
```

---

### Task 4: Generator — BDV XMLs and size report

**Files:**
- Modify: `2025_scripts/generate_nuclei_proba_images.py`
- Test: `2025_scripts/tests/test_generate_nuclei_proba_images.py`

**Interfaces:**
- Consumes: `write_outputs` (Task 3)
- Produces:
  - `write_local_xmls(subtypes: list[str], local_xml_dir: Path, stage_dir: Path, xml_template: Path) -> list[Path]` — one `{subtype}_proba.xml` per subtype in `local_xml_dir`, copied from the template with `<name>` and `<n5>` updated; `<n5 type="relative">` points from `local_xml_dir` to `stage_dir/{name}.n5`
  - `dir_size(path: Path) -> int` — recursive byte count
  - `report_sizes(stage_dir: Path, mask_path: Path, subtypes: list[str]) -> list[dict]` — per subtype: `file_size_bytes`, `mask_size_bytes`, `ratio`

- [ ] **Step 1: Write the failing tests**

```python
XML_TEMPLATE = """<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.n5">
      <n5 type="relative">mask.n5</n5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>nuclei</name>
        <size>3438 3240 2854</size>
        <voxelSize>
          <unit>micrometer</unit>
          <size>0.08 0.08 0.1</size>
        </voxelSize>
      </ViewSetup>
    </ViewSetups>
    <Timepoints type="range"><first>0</first><last>0</last></Timepoints>
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration setup="0" timepoint="0">
      <ViewTransform type="affine">
        <affine>0.08 0.0 0.0 0.0 0.0 0.08 0.0 0.0 0.0 0.0 0.1 0.0</affine>
      </ViewTransform>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""


class TestWriteLocalXmls(unittest.TestCase):
    def test_writes_xml_with_subtype_name_and_relative_n5_path(self):
        import xml.etree.ElementTree as ET
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            template = d / "nuclei.xml"
            template.write_text(XML_TEMPLATE, encoding="utf-8")
            stage = d / "stage"
            stage.mkdir()
            (stage / "clade6sub19_proba.n5").mkdir()
            xml_dir = d / "local"
            xmls = write_local_xmls(["clade6sub19"], xml_dir, stage, template)
            self.assertEqual(len(xmls), 1)
            root = ET.parse(xml_dir / "clade6sub19_proba.xml").getroot()
            name = root.find(".//ViewSetup/name").text
            self.assertEqual(name, "clade6sub19_proba")
            n5 = root.find(".//ImageLoader/n5")
            self.assertEqual(n5.get("type"), "relative")
            rel = Path(n5.text)
            self.assertTrue((xml_dir / rel).resolve() == (stage / "clade6sub19_proba.n5").resolve())
            self.assertEqual(root.find(".//ImageLoader").get("format"), "bdv.n5")
            self.assertEqual(root.find(".//size").text, "3438 3240 2854")


class TestSizeReport(unittest.TestCase):
    def test_dir_size_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask = d / "mask.n5"
            (mask / "s0").mkdir(parents=True)
            (mask / "s1").mkdir()
            (mask / "s0" / "a").write_bytes(b"x" * 1000)
            (mask / "s1" / "b").write_bytes(b"x" * 500)
            stage = d / "stage"
            stage.mkdir()
            (stage / "clade6sub19_proba.n5").mkdir()
            (stage / "clade6sub19_proba.n5" / "s0").write_bytes(b"x" * 3)
            rows = report_sizes(stage, mask, ["clade6sub19"])
            self.assertEqual(rows[0]["file_size_bytes"], 3)
            self.assertEqual(rows[0]["mask_size_bytes"], 1500)
            self.assertAlmostEqual(rows[0]["ratio"], 3 / 1500)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest 2025_scripts/tests/test_generate_nuclei_proba_images.py -v`
Expected: FAIL with `ImportError: cannot import name 'write_local_xmls'`

- [ ] **Step 3: Implement the three functions** (append after `write_outputs`)

```python
def write_local_xmls(subtypes, local_xml_dir, stage_dir, xml_template):
    """Write {subtype}_proba.xml copies of the template with name + n5 path updated."""
    local_xml_dir = Path(local_xml_dir)
    stage_dir = Path(stage_dir)
    local_xml_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for subtype in subtypes:
        name = f"{subtype}_proba"
        tree = ET.parse(xml_template)
        root = tree.getroot()
        for name_el in root.iter("name"):
            name_el.text = name
        n5_el = root.find(".//ImageLoader/n5")
        n5_el.set("type", "relative")
        n5_el.text = os.path.relpath(stage_dir / f"{name}.n5", local_xml_dir).replace("\\", "/")
        out_path = local_xml_dir / f"{name}.xml"
        ET.indent(root, space="  ")
        tree.write(out_path, encoding="utf-8", xml_declaration=False)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write("\n")
        written.append(out_path)
    return written


def dir_size(path):
    """Recursive byte count of all files under path."""
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def report_sizes(stage_dir, mask_path, subtypes):
    """Per-subtype: on-disk bytes of the generated N5 vs the mask N5, and ratio."""
    mask_total = dir_size(mask_path)
    rows = []
    for subtype in subtypes:
        n5 = Path(stage_dir) / f"{subtype}_proba.n5"
        total = dir_size(n5) if n5.is_dir() else 0
        rows.append({
            "subtype": subtype,
            "file_size_bytes": total,
            "mask_size_bytes": mask_total,
            "ratio": (total / mask_total) if mask_total else 0.0,
        })
    return rows


def print_report(rows):
    print(f"{'subtype':<24}{'file_bytes':>14}{'mask_bytes':>14}{'ratio':>12}")
    for r in rows:
        print(f"{r['subtype']:<24}{r['file_size_bytes']:>14}"
              f"{r['mask_size_bytes']:>14}{r['ratio']:>12.4f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Replace `main()` with the real CLI**

```python
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True,
                   help="Path to the nuclei segmentation N5 (must contain setup0/timepoint0/s*).")
    p.add_argument("--table", required=True,
                   help="Path to detailed_cell_types_cluster_probability.tsv")
    p.add_argument("--stage-dir", required=True,
                   help="Dir for generated {subtype}_proba.n5 files (gitignored, e.g. tmp_celltype_proba)")
    p.add_argument("--local-xml-dir", required=True,
                   help="Dir for local {subtype}_proba.xml files (repo images/local)")
    p.add_argument("--xml-template", default=str(DEFAULT_XML_TEMPLATE))
    p.add_argument("--subtypes", default=None,
                   help="Comma-separated subtypes (default: all from the table)")
    p.add_argument("--report-json", default=None,
                   help="Write the size report to this JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    subtypes = args.subtypes.split(",") if args.subtypes else read_subtype_columns(args.table)
    levels = mirror_level_info(args.mask)
    value_tables = {
        subtype: build_value_table(read_probabilities(args.table, subtype))
        for subtype in subtypes
    }
    write_outputs(args.mask, value_tables, Path(args.stage_dir), levels)
    write_local_xmls(subtypes, Path(args.local_xml_dir), Path(args.stage_dir),
                     Path(args.xml_template))
    rows = report_sizes(Path(args.stage_dir), Path(args.mask), subtypes)
    print_report(rows)
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
    total = sum(r["file_size_bytes"] for r in rows)
    print(f"\nWrote {len(subtypes)} images + local XMLs. "
          f"Total output bytes: {total} "
          f"(mask: {rows[0]['mask_size_bytes'] if rows else 0})",
          file=sys.stderr)
```

- [ ] **Step 6: Run all tests + a dry CLI smoke test**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (8 tests)

Run: `uv run --python 3.11 --with numpy --with z5py python 2025_scripts/generate_nuclei_proba_images.py --help`
Expected: usage text prints, exit 0

- [ ] **Step 7: Commit**

```bash
git add 2025_scripts/generate_nuclei_proba_images.py 2025_scripts/tests/test_generate_nuclei_proba_images.py
git commit -m "Add BDV XML generation and size report to proba image generator"
```

---

### Task 5: dataset.json wiring script

**Files:**
- Create: `2025_scripts/add_proba_sources_and_views.py`
- Test: `2025_scripts/tests/test_add_proba_sources_and_views.py`

**Interfaces:**
- Consumes: `read_subtype_columns` from `generate_nuclei_proba_images`
- Produces:
  - `source_definition(name: str) -> dict`
  - `view_definition(name: str) -> dict`
  - `add_sources_and_views(dataset: dict, subtypes: list[str]) -> dict` (idempotent; skips existing names)

- [ ] **Step 1: Write the failing tests**

```python
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from add_proba_sources_and_views import (
    source_definition,
    view_definition,
    add_sources_and_views,
)

EXISTING = {
    "is2D": False,
    "defaultLocation": {"position": [1.0, 2.0, 3.0]},
    "sources": {
        "raw": {"image": {"imageData": {"bdv.n5": {"relativePath": "x.xml"}}}},
        "nuclei": {"segmentation": {"imageData": {"bdv.n5": {"relativePath": "n.xml"}}}},
    },
    "views": {
        "default": {"uiSelectionGroup": "Figures Vergara2021",
                    "sourceDisplays": [{"imageDisplay": {"sources": ["raw"]}}]},
    },
}


class TestDefinitions(unittest.TestCase):
    def test_source_definition_has_both_paths(self):
        src = source_definition("clade1sub1_proba")
        self.assertEqual(src["image"]["imageData"]["bdv.n5"]["relativePath"],
                         "images/local/clade1sub1_proba.xml")
        self.assertEqual(src["image"]["imageData"]["bdv.n5.s3"]["relativePath"],
                         "images/bdv-n5-s3/celltype_proba/clade1sub1_proba.xml")

    def test_view_definition_is_concise(self):
        view = view_definition("clade1sub1_proba")
        self.assertEqual(view["uiSelectionGroup"], "nuclei_probabilities")
        self.assertNotIn("isExclusive", view)
        self.assertNotIn("viewerTransform", view)
        disp = view["sourceDisplays"][0]["imageDisplay"]
        self.assertEqual(disp["sources"], ["clade1sub1_proba"])
        self.assertEqual(disp["contrastLimits"], [0.0, 1000.0])
        self.assertEqual(disp["name"], "clade1sub1_proba")
        self.assertEqual(set(disp.keys()), {"sources", "contrastLimits", "name"})


class TestAddSourcesAndViews(unittest.TestCase):
    def test_adds_and_preserves_existing(self):
        import copy
        data = add_sources_and_views(copy.deepcopy(EXISTING), ["clade1sub1", "clade6sub19"])
        self.assertIn("clade1sub1_proba", data["sources"])
        self.assertIn("clade6sub19_proba", data["views"])
        self.assertIn("raw", data["sources"])          # untouched
        self.assertIn("default", data["views"])        # untouched
        self.assertEqual(len(data["sources"]), 4)
        self.assertEqual(len(data["views"]), 3)

    def test_idempotent(self):
        import copy
        once = add_sources_and_views(copy.deepcopy(EXISTING), ["clade1sub1"])
        twice = add_sources_and_views(copy.deepcopy(once), ["clade1sub1"])
        self.assertEqual(twice, once)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest 2025_scripts/tests/test_add_proba_sources_and_views.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'add_proba_sources_and_views'`

- [ ] **Step 3: Implement the script**

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Add per-subtype probability sources and views to dataset.json.

Reads the subtype columns from the detailed cluster probability table and adds
one image source ({subtype}_proba, bdv.n5 + bdv.n5.s3) and one additive view
(imageDisplay, contrastLimits [0.0, 1000.0]) per subtype to dataset.json under
the nuclei_probabilities uiSelectionGroup. Idempotent: existing names are left
untouched. Dry-run by default; use --write to apply.

Usage:
    ./add_proba_sources_and_views.py --dataset-json <dataset.json> --table <tsv> [--subtypes a,b,c] [--write]
"""

import argparse
import json
import sys
from pathlib import Path

from generate_nuclei_proba_images import read_subtype_columns

UI_GROUP = "nuclei_probabilities"
S3_PREFIX = "images/bdv-n5-s3/celltype_proba"


def source_definition(name):
    return {
        "image": {
            "imageData": {
                "bdv.n5": {"relativePath": f"images/local/{name}.xml"},
                "bdv.n5.s3": {"relativePath": f"{S3_PREFIX}/{name}.xml"},
            }
        }
    }


def view_definition(name):
    return {
        "uiSelectionGroup": UI_GROUP,
        "sourceDisplays": [{
            "imageDisplay": {
                "sources": [name],
                "contrastLimits": [0.0, 1000.0],
                "name": name,
            }
        }],
    }


def add_sources_and_views(dataset, subtypes):
    sources = dataset.setdefault("sources", {})
    views = dataset.setdefault("views", {})
    added = []
    for subtype in subtypes:
        name = f"{subtype}_proba"
        if name not in sources:
            sources[name] = source_definition(name)
        if name not in views:
            views[name] = view_definition(name)
        added.append(name)
    return dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-json", required=True,
                   help="Path to data/platybrowser_6dpf/dataset.json")
    p.add_argument("--table", required=True,
                   help="Path to detailed_cell_types_cluster_probability.tsv")
    p.add_argument("--subtypes", default=None,
                   help="Comma-separated subtypes (default: all from the table)")
    p.add_argument("--write", action="store_true",
                   help="Apply changes to dataset.json (default: dry-run)")
    return p.parse_args()


def main():
    args = parse_args()
    subtypes = args.subtypes.split(",") if args.subtypes else read_subtype_columns(args.table)
    ds_path = Path(args.dataset_json)
    with open(ds_path, encoding="utf-8") as f:
        dataset = json.load(f)
    before_sources = set(dataset.get("sources", {}))
    before_views = set(dataset.get("views", {}))
    add_sources_and_views(dataset, subtypes)
    new_sources = set(dataset["sources"]) - before_sources
    new_views = set(dataset["views"]) - before_views
    print(f"{len(new_sources)} new sources, {len(new_views)} new views "
          f"(group '{UI_GROUP}')", file=sys.stderr)
    if args.write:
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
            f.write("\n")
        print(f"Wrote {ds_path}", file=sys.stderr)
    else:
        print("Dry run: no changes written. Use --write to apply.", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --python 3.11 --with numpy --with z5py python -m unittest discover -s 2025_scripts/tests -p "test_*.py" -v`
Expected: PASS (all tests, including Task 1–4)

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/add_proba_sources_and_views.py 2025_scripts/tests/test_add_proba_sources_and_views.py
git commit -m "Add script to wire proba sources and views into dataset.json"
```

---

### Task 6: Pilot run (5 subtypes) + size report + Fiji validation

**Files:**
- Modify: `data/platybrowser_6dpf/dataset.json` (5 sources + 5 views)
- Create: `data/platybrowser_6dpf/images/local/{s}_proba.xml` ×5, `data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/{s}_proba.xml` ×5
- Ignored (never committed): `tmp_celltype_proba/*.n5` ×5

**Interfaces:**
- Consumes: generator + wiring scripts (Tasks 1–5), the existing S3 uploader (XML conversion only)

- [ ] **Step 1: Pull before touching dataset.json**

Run: `git pull --rebase`
Expected: up to date (or rebased onto any pushed view changes)

- [ ] **Step 2: Determine the nuclei mask N5 path — ASK THE USER**

The mask N5 is not in this checkout. Ask the user for the local path to
`sbem-6dpf-1-whole-segmented-nuclei.n5` (mounted network drive, or download from
S3 `images/bdv-n5-s3/vergara_2021/`). Record it as `$MASK`. Do not proceed until
provided. Sanity check once provided:

Run: `uv run --python 3.11 --with z5py python -c "import z5py,sys; f=z5py.File(sys.argv[1],'r'); print(sorted(f['setup0/timepoint0'].keys()))" "$MASK"`
Expected: prints `['s0', ...]` (levels of the mask pyramid)

- [ ] **Step 3: Generate the 5 pilot images**

Run:
```bash
uv run --python 3.11 --with numpy --with z5py 2025_scripts/generate_nuclei_proba_images.py \
  --mask "$MASK" \
  --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
  --stage-dir tmp_celltype_proba \
  --local-xml-dir data/platybrowser_6dpf/images/local \
  --subtypes clade6sub19,nocladesub12,clade9sub4,nocladesub20,clade1sub2 \
  --report-json tmp_celltype_proba/size_report.json
```
Expected: prints the size report table (`file_bytes`, `mask_bytes`, `ratio` per
subtype) + a summary line; 5 `.n5` dirs in `tmp_celltype_proba/`; 5 XMLs in
`data/platybrowser_6dpf/images/local/`.

**Deliverable — relay the size report to the user** (file sizes vs the nuclei
segmentation, per subtype and ratio; this is the pilot's requested comparison).

- [ ] **Step 4: Generate S3 XMLs (dry-run of the existing uploader, no upload)**

The uploader expects each `{name}.xml` to sit next to its `{name}.n5` in the
input folder, so copy the local XMLs into the stage dir first (their n5 text is
overwritten by the S3 `Key` anyway).

Run:
```bash
cp data/platybrowser_6dpf/images/local/clade6sub19_proba.xml \
   data/platybrowser_6dpf/images/local/nocladesub12_proba.xml \
   data/platybrowser_6dpf/images/local/clade9sub4_proba.xml \
   data/platybrowser_6dpf/images/local/nocladesub20_proba.xml \
   data/platybrowser_6dpf/images/local/clade1sub2_proba.xml \
   tmp_celltype_proba/
cd 2025_scripts && uv run upload_Alyona_local_n5_to_s3.py \
  -i ../tmp_celltype_proba \
  -o ../data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba \
  -e "https://s3.embl.de" -r "us-west-2" -b "platybrowser-2025" \
  -p "images/bdv-n5-s3/celltype_proba" --dry-run
```
Expected: 5 S3-format XMLs created under
`data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/` (`format="bdv.n5.s3"`,
`Key` = `images/bdv-n5-s3/celltype_proba/{name}.n5`). No upload happens.

- [ ] **Step 5: Wire the 5 sources + views into dataset.json**

Run:
```bash
uv run --python 3.11 --with numpy --with z5py 2025_scripts/add_proba_sources_and_views.py \
  --dataset-json data/platybrowser_6dpf/dataset.json \
  --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
  --subtypes clade6sub19,nocladesub12,clade9sub4,nocladesub20,clade1sub2 --write
```
Expected: prints `5 new sources, 5 new views (group 'nuclei_probabilities')` and
writes `dataset.json`. Verify with `git diff data/platybrowser_6dpf/dataset.json`
— only added entries.

- [ ] **Step 6: Validate**

Run: `python 2025_scripts/validate_dataset_json.py`
Expected: exits 0, no errors printed.

Run: `python 2025_scripts/compress_dataset_json.py --check`
Expected: prints the dataset is already compressed (no default-valued keys).

- [ ] **Step 7: Commit the pilot**

```bash
git add data/platybrowser_6dpf/dataset.json \
  data/platybrowser_6dpf/images/local/clade6sub19_proba.xml \
  data/platybrowser_6dpf/images/local/nocladesub12_proba.xml \
  data/platybrowser_6dpf/images/local/clade9sub4_proba.xml \
  data/platybrowser_6dpf/images/local/nocladesub20_proba.xml \
  data/platybrowser_6dpf/images/local/clade1sub2_proba.xml \
  data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/
git commit -m "Add pilot: 5 per-subtype nuclei probability images and views"
```
Note: the pre-commit hook runs compress + validate; both must pass. `*.n5` files
are gitignored — verify `git status` shows no `.n5` staged.

- [ ] **Step 8: User validates in MoBIE Fiji**

Ask the user to open the project in Fiji and check, for a few of the 5 pilot
entries under the `nuclei_probabilities` dropdown:
1. image loads and overlays the EM/nuclei exactly (black background, intensities at nuclei)
2. toggling a view does not move the camera (additive)
3. multiple pilot views can be open at once
4. recoloring with a LUT (e.g. viridis) works on the image source
5. if the user wants the pyramid to render faster at zoom-out, confirm whether
   the mask's coarse levels (`s1`, `s2`, …) are being used (they are mirrored automatically)

Record any changes requested. Iterate on Tasks 1–5 if a problem surfaces (e.g.
z5py version API differences), then recommit.

---

### Task 7: Full run (274) + S3 upload

**Files:**
- Modify: `data/platybrowser_6dpf/dataset.json` (all 274 sources + views)
- Create: `data/platybrowser_6dpf/images/local/*_proba.xml` ×274, `data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/*_proba.xml` ×274
- S3 (data only, metadata in git): `images/bdv-n5-s3/celltype_proba/{name}.n5` ×274

**Interfaces:**
- Consumes: Tasks 1–6; the existing S3 uploader (upload + XML conversion); `.env` credentials (see `2025_scripts/README.md`)

- [ ] **Step 1: Pull before touching dataset.json**

Run: `git pull --rebase`
Expected: up to date

- [ ] **Step 2: Generate all 274 images**

Run:
```bash
uv run --python 3.11 --with numpy --with z5py 2025_scripts/generate_nuclei_proba_images.py \
  --mask "$MASK" \
  --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
  --stage-dir tmp_celltype_proba \
  --local-xml-dir data/platybrowser_6dpf/images/local \
  --report-json tmp_celltype_proba/size_report.json
```
Expected: size report for all 274 subtypes; 274 `.n5` dirs staged; 274 local
XMLs written. Relay the summary (total bytes, mean ratio vs the nuclei N5) to
the user.

- [ ] **Step 3: Upload to S3 + generate S3 XMLs**

Dry run first:
```bash
cp data/platybrowser_6dpf/images/local/*_proba.xml tmp_celltype_proba/
cd 2025_scripts && uv run upload_Alyona_local_n5_to_s3.py \
  -i ../tmp_celltype_proba \
  -o ../data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba \
  -e "https://s3.embl.de" -r "us-west-2" -b "platybrowser-2025" \
  -p "images/bdv-n5-s3/celltype_proba" --dry-run
```
Then the real run (same command without `--dry-run`). Requires `.env` with S3
credentials (per `2025_scripts/README.md`). Expected: 274 N5 folders uploaded
under `images/bdv-n5-s3/celltype_proba/`, 274 S3 XMLs generated.

- [ ] **Step 4: Wire all 274 sources + views**

Run:
```bash
uv run --python 3.11 --with numpy --with z5py 2025_scripts/add_proba_sources_and_views.py \
  --dataset-json data/platybrowser_6dpf/dataset.json \
  --table data/platybrowser_6dpf/tables/sbem-6dpf-1-whole-segmented-nuclei/detailed_cell_types_cluster_probability.tsv \
  --write
```
Expected: `274 new sources, 274 new views` (the 5 pilot entries already exist
and are skipped — idempotent).

- [ ] **Step 5: Validate + commit**

Run: `python 2025_scripts/validate_dataset_json.py` → exit 0
Run: `python 2025_scripts/compress_dataset_json.py --check` → already compressed

```bash
git add data/platybrowser_6dpf/dataset.json data/platybrowser_6dpf/images/local/
git add data/platybrowser_6dpf/images/bdv-n5-s3/celltype_proba/
git commit -m "Add 274 per-subtype nuclei probability images and views"
```

- [ ] **Step 6: Final verification**

- `git status` clean except nothing; confirm no `.n5` committed
- Confirm counts in `dataset.json`: `nuclei_probabilities` group has 274 views;
  274 `*_proba` sources (use `git grep -c '"clade.*_proba"'` style check or a
  quick python count)
- Ask the user to spot-check a few subtypes in Fiji (S3 path now active)

---

## Self-Review Notes

- **Spec coverage:** images (T1–T4), pyramid mirroring (T2–T3), storage layout
  incl. `celltype_proba` (T4, T6–T7), dataset wiring with both paths + concise
  views (T5), pilot + size report vs nuclei (T6), full run + S3 (T7), verification
  via validate/compress (T6–T7). ✓
- **Type consistency:** `write_outputs(mask_path, value_tables, stage_dir, levels, group_attrs)`
  used identically in T3 test, T4 `main`, T6/T7 commands. `report_sizes(stage_dir,
  mask_path, subtypes)` consistent. `read_subtype_columns(tsv_path)` shared by
  generator and wiring script. ✓
- **No placeholders:** every step has code or an exact command. The only
  environment-dependent values are `$MASK` (user-provided path, with a sanity
  check step) and S3 credentials (`.env`), both explicitly documented.
