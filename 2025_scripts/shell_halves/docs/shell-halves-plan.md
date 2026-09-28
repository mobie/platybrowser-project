# Shell Half-Objects Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate four half-shell image sources (left/right and front/back) from the existing shell mask, upload them to `platybrowser-2025`, and expose them as views in the `sbem` group.

**Architecture:** A new `2025_scripts/generate_shell_halves.py` reads the shell N5, measures its centroid and principal axes by PCA, and zeroes one half-space per output using two planes through the centroid (normals `LR` and `DV`, both containing the main axis), writing four uint8 gzip N5 pyramids mirroring the shell's levels/chunks/attributes. The volumes are mirrored to S3 with `mc`; local and S3 XMLs plus four sources and four views are committed to `dataset.json`.

**Tech Stack:** Python 3.12 via `uv` (inline deps `numpy`, `z5py`), `mc` (MinIO client), MoBIE `dataset.json`.

> **Revision R1 (2026-09-11):** Tasks 1–4 below were originally written with
> `x=y` / `x=-y` planes through the index origin. The `x=-y` plane is degenerate
> for the real shell (a hollow ovoid sitting away from the origin), so those
> plane definitions were replaced by the measured principal-axis planes in
> commit `b5e919a` (see the spec and `2026-09-11-shell-ovoid-orientation.md`).
> The Tasks 1–4 code blocks below are therefore **historical** — their
> `mask_block`/`HALVES` shapes no longer match the current
> `2025_scripts/generate_shell_halves.py`. Tasks 5–7 reflect the current names
> and behavior.

## Global Constraints

- Branch: `shell_halves` (already created off `main`); spec at `docs/superpowers/specs/2026-09-11-shell-halves-design.md`.
- **Never modify or delete an existing S3 object.** Only add new keys under `platybrowser-2025/images/bdv-n5-s3/shell_halves/`.
- Do not touch the original `shell` source or its `sbem` view.
- Source names exactly: `shell_left`, `shell_right`, `shell_front`, `shell_back`.
- Cut planes pass through the measured shell centroid and contain the main axis: normals `LR` (left/right) and `DV` (front/back). See `docs/superpowers/specs/2026-09-11-shell-halves-design.md` and `2026-09-11-shell-ovoid-orientation.md`.
- N5 output: uint8, chunks copied from the source (`96³`), gzip level 1, fillvalue 0, all 5 pyramid levels, group + level attributes mirrored.
- Axis order: z5py array shape is `(z, y, x)`. Axis 1 = y, axis 2 = x, axis 0 = z. A level voxel at index `i` represents full-resolution coordinate `i*ds + (ds−1)/2`.
- Keep/zero semantics: `left`/`front` keep `(p−c)·n ≥ 0`; `right`/`back` keep `≤ 0` (plane voxels kept by both halves of a pair).
- Every `dataset.json` edit must pass `python3 2025_scripts/compress_dataset_json.py --check` and `python3 2025_scripts/validate_dataset_json.py` (also enforced by the pre-commit hook).
- Test command: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest <file> -v`.

---

## File Structure

- Create: `2025_scripts/generate_shell_halves.py` — masking + N5 writing + XML generation + CLI.
- Create: `2025_scripts/tests/test_generate_shell_halves.py` — unit tests.
- Modify: `data/platybrowser_6dpf/dataset.json` — 4 new image sources, 4 new views.
- Generated (committed): `data/platybrowser_6dpf/images/local/shell_left.xml`, `shell_right.xml`, `shell_front.xml`, `shell_back.xml`.
- Generated (committed): `data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves/shell_left.xml`, `shell_right.xml`, `shell_front.xml`, `shell_back.xml`.
- Staging (gitignored): `tmp_shell_halves_src/` (source mirror), `data/rawdata/shell_halves/` (output N5s).

---

### Task 1: Masking primitives

**Files:**
- Create: `2025_scripts/generate_shell_halves.py`
- Test: `2025_scripts/tests/test_generate_shell_halves.py`

**Interfaces:**
- Produces: `HALVES: dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]]`, `HALF_NAMES: list[str]`, `block_slices(shape, chunks)`, `mask_block(block, y0, x0, keep, offset=0) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Create `2025_scripts/tests/test_generate_shell_halves.py`:

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from generate_shell_halves import HALVES, HALF_NAMES, block_slices, mask_block


class TestMasking(unittest.TestCase):
    def test_sag_left_keeps_x_ge_y(self):
        block = np.ones((1, 4, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, HALVES["shell_left"], 0)
        for y in range(4):
            for x in range(5):
                self.assertEqual(out[0, y, x], 1 if x >= y else 0)

    def test_sag_right_keeps_x_le_y(self):
        block = np.ones((1, 4, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, HALVES["shell_right"], 0)
        for y in range(4):
            for x in range(5):
                self.assertEqual(out[0, y, x], 1 if x <= y else 0)

    def test_cor_front_keeps_x_ge_neg_y(self):
        block = np.ones((1, 4, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, HALVES["shell_front"], 0)
        for y in range(4):
            for x in range(5):
                self.assertEqual(out[0, y, x], 1 if x >= -y else 0)

    def test_cor_back_keeps_x_le_neg_y(self):
        block = np.ones((1, 4, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, HALVES["shell_back"], 0)
        for y in range(4):
            for x in range(5):
                self.assertEqual(out[0, y, x], 1 if x <= -y else 0)

    def test_global_offset_used_for_block(self):
        # block at global y0=2, x0=5 with all ones; x-y >= 1 everywhere.
        # Block-local indexing (x,y in 0..2) would give x-y in [-2,2] and fail,
        # so this distinguishes global from block-local masking.
        block = np.ones((1, 3, 3), dtype=np.uint8)
        out = mask_block(block, y0=2, x0=5, keep=HALVES["shell_left"], offset=0)
        self.assertTrue((out == 1).all())

    def test_offset_shifts_plane(self):
        # keep x - y >= 2; at x=2,y=0 -> 2>=2 kept; at x=0,y=0 -> 0>=2 dropped
        block = np.ones((1, 1, 3), dtype=np.uint8)
        out = mask_block(block, y0=0, x0=0, keep=HALVES["shell_left"], offset=2)
        self.assertEqual(list(out[0, 0]), [0, 0, 1])

    def test_half_names_and_order(self):
        self.assertEqual(
            HALF_NAMES,
            ["shell_left", "shell_right", "shell_front", "shell_back"],
        )

    def test_block_slices_cover_shape(self):
        slices = list(block_slices((3, 4, 5), (2, 2, 2)))
        self.assertEqual(len(slices), 2 * 2 * 3)
        covered = sum(
            (sl[0].stop - sl[0].start)
            * (sl[1].stop - sl[1].start)
            * (sl[2].stop - sl[2].start)
            for sl in slices
        )
        self.assertEqual(covered, 3 * 4 * 5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generate_shell_halves'`.

- [ ] **Step 3: Write the minimal implementation**

Create `2025_scripts/generate_shell_halves.py`:

```python
#!/usr/bin/env -S uv run --python 3.12
# /// script
# dependencies = ["numpy", "z5py"]
# ///
"""Generate four half-shell N5 images from the shell mask.

The shell is a binary uint8 mask. Cut it with two vertical 45-degree planes
through the origin, in index space:

    sagittal plane  x = y   -> shell_left  (keep x - y >= offset)
                               shell_right (keep x - y <= offset)
    coronal plane   x = -y  -> shell_front (keep x + y >= offset)
                               shell_back  (keep x + y <= offset)

N5 arrays are stored as (z, y, x): axis 1 is y, axis 2 is x, z is untouched.
Every pyramid level is an exact power-of-two downsample aligned to the origin,
so the condition is scale-invariant and each level is masked with its own
global indices. Outputs mirror the shell pyramid (levels, shapes, chunks,
attributes) and use gzip + fillvalue 0.

Usage:
    ./generate_shell_halves.py --mask <shell.n5> --stage-dir <dir> \
        --local-xml-dir data/platybrowser_6dpf/images/local \
        --s3-xml-dir data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves
"""

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import z5py

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_XML_TEMPLATE = (
    REPO_ROOT / "data/platybrowser_6dpf/images/local/"
    "sbem-6dpf-1-whole-segmented-shell.xml"
)
S3_XML_TEMPLATE = (
    REPO_ROOT / "data/platybrowser_6dpf/images/bdv-n5-s3/vergara_2021/"
    "sbem-6dpf-1-whole-segmented-shell.xml"
)

S3_PREFIX = "images/bdv-n5-s3/shell_halves"
S3_BUCKET = "platybrowser-2025"
S3_ENDPOINT = "https://s3.embl.de"
S3_REGION = "us-west-2"

# name -> predicate(global_x, global_y, offset) -> bool array
HALVES = {
    "shell_left": lambda x, y, o: (x - y) >= o,
    "shell_right": lambda x, y, o: (x - y) <= o,
    "shell_front": lambda x, y, o: (x + y) >= o,
    "shell_back": lambda x, y, o: (x + y) <= o,
}
HALF_NAMES = list(HALVES)


def block_slices(shape, chunks):
    """Yield every (z, y, x) block slice of an array in chunk-aligned order."""
    for z0 in range(0, shape[0], chunks[0]):
        for y0 in range(0, shape[1], chunks[1]):
            for x0 in range(0, shape[2], chunks[2]):
                yield (
                    slice(z0, min(z0 + chunks[0], shape[0])),
                    slice(y0, min(y0 + chunks[1], shape[1])),
                    slice(x0, min(x0 + chunks[2], shape[2])),
                )


def mask_block(block, y0, x0, keep, offset=0):
    """Zero out voxels of a (z, y, x) block failing keep(global_x, global_y)."""
    y = np.arange(y0, y0 + block.shape[1])
    x = np.arange(x0, x0 + block.shape[2])
    keep_xy = keep(x[None, None, :], y[None, :, None], offset)  # (1, by, bx)
    return np.where(keep_xy, block, 0).astype(block.dtype, copy=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_shell_halves.py 2025_scripts/tests/test_generate_shell_halves.py
git commit -m "Add shell half masking primitives"
```

---

### Task 2: N5 introspection and writer

**Files:**
- Modify: `2025_scripts/generate_shell_halves.py`
- Test: `2025_scripts/tests/test_generate_shell_halves.py`

**Interfaces:**
- Consumes: `HALVES`, `HALF_NAMES`, `block_slices`, `mask_block` from Task 1.
- Produces: `mirror_level_info(mask_path) -> list[dict]`, `mirror_group_attrs(mask_path) -> dict`, `write_halves(mask_path, stage_dir, levels, group_attrs, halves=HALVES, offset=0, gzip_level=1) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

Append to `2025_scripts/tests/test_generate_shell_halves.py` (add imports `tempfile`, `z5py`, and the new functions to the existing import block):

```python
import tempfile

import z5py

from generate_shell_halves import (
    mirror_group_attrs,
    mirror_level_info,
    write_halves,
)


def make_mask_n5(path: Path, shape=(3, 4, 5)) -> Path:
    """Tiny single-level uint8 mask N5 with shell-like group attributes."""
    with z5py.File(str(path), "a") as f:
        setup = f.create_group("setup0")
        setup.attrs["dataType"] = "uint8"
        setup.attrs["downsamplingFactors"] = [[1, 1, 1]]
        tp = setup.create_group("timepoint0")
        tp.attrs["multiScale"] = True
        tp.attrs["resolution"] = [0.32, 0.32, 0.4]
        ds = tp.create_dataset(
            "s0", shape=shape, chunks=(2, 2, 2),
            dtype="uint8", compression="gzip", level=1, fillvalue=0,
        )
        ds.attrs["downsamplingFactors"] = [1, 1, 1]
        ds[:] = 1
    return path


class TestWriteHalves(unittest.TestCase):
    def test_writes_four_masked_uint8_n5s(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "mask.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs, offset=0, gzip_level=1)

            self.assertEqual(
                sorted(p.name for p in stage.glob("*.n5")),
                sorted(f"{n}.n5" for n in HALF_NAMES),
            )
            for name, keep in HALVES.items():
                with z5py.File(str(stage / f"{name}.n5"), "r") as f:
                    ds = f["setup0/timepoint0/s0"]
                    self.assertEqual(tuple(ds.shape), (3, 4, 5))
                    self.assertEqual(ds.dtype, np.dtype("uint8"))
                    data = ds[:]
                for y in range(4):
                    for x in range(5):
                        if data[:, y, x].any():
                            self.assertTrue(
                                bool(keep(np.array([x]), np.array([y]), 0)[0]),
                                f"{name} kept removed voxel x={x} y={y}",
                            )

    def test_group_attrs_mirrored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "mask.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs)
            with z5py.File(str(stage / "shell_left.n5"), "r") as f:
                self.assertEqual(dict(f["setup0"].attrs)["dataType"], "uint8")
                self.assertEqual(
                    list(dict(f["setup0/timepoint0"].attrs)["resolution"]),
                    [0.32, 0.32, 0.4],
                )
                self.assertEqual(
                    list(dict(f["setup0/timepoint0/s0"].attrs)["downsamplingFactors"]),
                    [1, 1, 1],
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: FAIL — `ImportError: cannot import name 'write_halves'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `2025_scripts/generate_shell_halves.py` (before the `if __name__` block, which does not exist yet — append at end of file):

```python
def mirror_level_info(mask_path):
    """Level name, shape, chunks and attrs for every s-level of the mask."""
    with z5py.File(str(mask_path), "r") as f:
        tp = f["setup0/timepoint0"]
        return [
            {
                "name": key,
                "shape": tuple(tp[key].shape),
                "chunks": tuple(tp[key].chunks),
                "attrs": dict(tp[key].attrs),
            }
            for key in sorted(tp.keys())
        ]


def mirror_group_attrs(mask_path):
    """The mask's setup0 and timepoint0 group attributes."""
    with z5py.File(str(mask_path), "r") as f:
        return {
            "setup0": dict(f["setup0"].attrs),
            "timepoint0": dict(f["setup0/timepoint0"].attrs),
        }


def write_halves(mask_path, stage_dir, levels, group_attrs, halves=HALVES,
                 offset=0, gzip_level=1):
    """Write one uint8 N5 per half into stage_dir, mirroring the mask pyramid."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in halves:
        out_path = stage_dir / f"{name}.n5"
        if out_path.exists():
            shutil.rmtree(out_path)
        with z5py.File(str(out_path), "a") as f:
            setup = f.create_group("setup0")
            for k, v in group_attrs.get("setup0", {}).items():
                setup.attrs[k] = v
            tp = setup.create_group("timepoint0")
            for k, v in group_attrs.get("timepoint0", {}).items():
                tp.attrs[k] = v
            for lvl in levels:
                ds = tp.create_dataset(
                    lvl["name"], shape=lvl["shape"], chunks=lvl["chunks"],
                    dtype="uint8", compression="gzip", level=gzip_level,
                    fillvalue=0,
                )
                for k, v in lvl["attrs"].items():
                    ds.attrs[k] = v
        written.append(out_path)

    with z5py.File(str(mask_path), "r") as mf:
        mtp = mf["setup0/timepoint0"]
        for name, keep in halves.items():
            with z5py.File(str(stage_dir / f"{name}.n5"), "a") as of:
                otp = of["setup0/timepoint0"]
                for lvl in levels:
                    mds = mtp[lvl["name"]]
                    ods = otp[lvl["name"]]
                    for sl in block_slices(lvl["shape"], lvl["chunks"]):
                        block = mds[sl]
                        if not block.any():
                            continue
                        ods[sl] = mask_block(
                            block, sl[1].start, sl[2].start, keep, offset
                        )
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_shell_halves.py 2025_scripts/tests/test_generate_shell_halves.py
git commit -m "Add shell half N5 writer"
```

---

### Task 3: XML generation

**Files:**
- Modify: `2025_scripts/generate_shell_halves.py`
- Test: `2025_scripts/tests/test_generate_shell_halves.py`

**Interfaces:**
- Consumes: `HALF_NAMES`, `LOCAL_XML_TEMPLATE`, `S3_XML_TEMPLATE`, `S3_PREFIX`, `S3_BUCKET`, `S3_ENDPOINT`, `S3_REGION` from Task 1.
- Produces: `write_local_xmls(names, local_xml_dir, stage_dir, template=LOCAL_XML_TEMPLATE) -> list[Path]`, `write_s3_xmls(names, s3_xml_dir, template=S3_XML_TEMPLATE, prefix=S3_PREFIX) -> list[Path]`.

- [ ] **Step 1: Write the failing test**

Append to `2025_scripts/tests/test_generate_shell_halves.py` (add `ET` import and the two functions to the import block):

```python
import xml.etree.ElementTree as ET

from generate_shell_halves import write_local_xmls, write_s3_xmls


class TestXml(unittest.TestCase):
    def test_local_xml_points_at_staged_n5(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stage = tmp / "rawdata" / "shell_halves"
            local = tmp / "images" / "local"
            out = write_local_xmls(["shell_left"], local, stage)
            root = ET.parse(out[0]).getroot()
            self.assertEqual(
                root.find(".//ViewSetup/name").text, "shell_left")
            self.assertEqual(
                root.find(".//ImageLoader").get("format"), "bdv.n5")
            n5 = root.find(".//ImageLoader/n5")
            self.assertTrue(
                n5.text.replace("\\", "/").endswith(
                    "rawdata/shell_halves/shell_left.n5"),
                n5.text,
            )

    def test_s3_xml_has_bucket_key_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_s3_xmls(["shell_back"], Path(tmp) / "s3")
            root = ET.parse(out[0]).getroot()
            self.assertEqual(
                root.find(".//ViewSetup/name").text, "shell_back")
            self.assertEqual(
                root.find(".//ImageLoader").get("format"), "bdv.n5.s3")
            self.assertEqual(
                root.find(".//Key").text,
                "images/bdv-n5-s3/shell_halves/shell_back.n5",
            )
            self.assertEqual(root.find(".//BucketName").text, "platybrowser-2025")
            self.assertEqual(
                root.find(".//ServiceEndpoint").text, "https://s3.embl.de")
            self.assertEqual(root.find(".//SigningRegion").text, "us-west-2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: FAIL — `ImportError: cannot import name 'write_local_xmls'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `2025_scripts/generate_shell_halves.py`:

```python
def _write_xml(template, out_path, name, loader_text, loader_format, s3):
    """Copy a shell XML template, set the setup name and the loader location."""
    tree = ET.parse(template)
    root = tree.getroot()
    setup_name = root.find(".//ViewSetup/name")
    if setup_name is not None:
        setup_name.text = name
    loader = root.find(".//ImageLoader")
    loader.set("format", loader_format)
    if s3:
        for tag in ("n5", "Key", "SigningRegion", "ServiceEndpoint", "BucketName"):
            for el in loader.findall(tag):
                loader.remove(el)
        ET.SubElement(loader, "Key").text = loader_text
        ET.SubElement(loader, "SigningRegion").text = S3_REGION
        ET.SubElement(loader, "ServiceEndpoint").text = S3_ENDPOINT
        ET.SubElement(loader, "BucketName").text = S3_BUCKET
    else:
        n5_el = loader.find("n5")
        n5_el.set("type", "relative")
        n5_el.text = loader_text
    ET.indent(root, space="  ")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n")
    return out_path


def write_local_xmls(names, local_xml_dir, stage_dir, template=LOCAL_XML_TEMPLATE):
    """Write images/local/<name>.xml pointing at the staged <name>.n5."""
    local_xml_dir = Path(local_xml_dir)
    stage_dir = Path(stage_dir)
    written = []
    for name in names:
        n5_rel = os.path.relpath(stage_dir / f"{name}.n5", local_xml_dir)
        written.append(_write_xml(
            template, local_xml_dir / f"{name}.xml", name,
            n5_rel.replace("\\", "/"), "bdv.n5", s3=False,
        ))
    return written


def write_s3_xmls(names, s3_xml_dir, template=S3_XML_TEMPLATE, prefix=S3_PREFIX):
    """Write S3 XMLs with Key <prefix>/<name>.n5 in bucket platybrowser-2025."""
    s3_xml_dir = Path(s3_xml_dir)
    written = []
    for name in names:
        written.append(_write_xml(
            template, s3_xml_dir / f"{name}.xml", name,
            f"{prefix}/{name}.n5", "bdv.n5.s3", s3=True,
        ))
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_shell_halves.py 2025_scripts/tests/test_generate_shell_halves.py
git commit -m "Add shell half XML generation"
```

---

### Task 4: CLI and end-to-end wiring

**Files:**
- Modify: `2025_scripts/generate_shell_halves.py`
- Test: `2025_scripts/tests/test_generate_shell_halves.py`

**Interfaces:**
- Consumes: all functions from Tasks 1–3.
- Produces: `parse_args()`, `main()`; the script is runnable as
  `uv run --python 3.12 2025_scripts/generate_shell_halves.py --mask ... --stage-dir ... --local-xml-dir ... --s3-xml-dir ...`.

- [ ] **Step 1: Write the failing test**

Append to `2025_scripts/tests/test_generate_shell_halves.py` (add `main` import):

```python
from generate_shell_halves import main


class TestCli(unittest.TestCase):
    def test_main_writes_n5s_and_xmls(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "mask.n5")
            stage = tmp / "stage"
            local = tmp / "local"
            s3 = tmp / "s3"
            import sys as _sys
            argv = _sys.argv
            _sys.argv = [
                "generate_shell_halves.py",
                "--mask", str(mask),
                "--stage-dir", str(stage),
                "--local-xml-dir", str(local),
                "--s3-xml-dir", str(s3),
            ]
            try:
                main()
            finally:
                _sys.argv = argv
            self.assertEqual(len(list(stage.glob("*.n5"))), 4)
            self.assertEqual(len(list(local.glob("*.xml"))), 4)
            self.assertEqual(len(list(s3.glob("*.xml"))), 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `2025_scripts/generate_shell_halves.py`:

```python
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", required=True,
                   help="Path to the shell N5 (setup0/timepoint0/s*).")
    p.add_argument("--stage-dir", required=True,
                   help="Dir for generated <name>.n5 (gitignored).")
    p.add_argument("--local-xml-dir", required=True,
                   help="Dir for local <name>.xml (repo images/local).")
    p.add_argument("--s3-xml-dir", required=True,
                   help="Dir for S3 <name>.xml (repo images/bdv-n5-s3/shell_halves).")
    p.add_argument("--offset", type=int, default=0,
                   help="Plane offset in voxels (default 0 = through origin).")
    p.add_argument("--gzip-level", type=int, default=1,
                   help="Gzip compression level for the output N5s (default 1).")
    return p.parse_args()


def main():
    args = parse_args()
    levels = mirror_level_info(args.mask)
    group_attrs = mirror_group_attrs(args.mask)
    write_halves(args.mask, Path(args.stage_dir), levels, group_attrs,
                 offset=args.offset, gzip_level=args.gzip_level)
    write_local_xmls(HALF_NAMES, Path(args.local_xml_dir), Path(args.stage_dir))
    write_s3_xmls(HALF_NAMES, Path(args.s3_xml_dir))
    print(f"Wrote {len(HALF_NAMES)} half-shell N5s to {args.stage_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --python 3.12 --with pytest --with z5py --with numpy python -m pytest 2025_scripts/tests/test_generate_shell_halves.py -v`
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
git add 2025_scripts/generate_shell_halves.py 2025_scripts/tests/test_generate_shell_halves.py
git commit -m "Add shell halves CLI"
```

---

### Task 5: Verify S3 write access and generate on the real shell

**Files:**
- Generated (gitignored): `tmp_shell_halves_src/`, `data/rawdata/shell_halves/*.n5`
- Generated (committed later in Task 7): the 8 XMLs.

**Interfaces:**
- Consumes: the runnable script from Task 4; `mc` alias `EmblArendtS3`.

- [ ] **Step 1: Verify `EmblArendtS3` can write to `platybrowser-2025`**

```bash
echo "probe" > /tmp/opencode/shell_probe.txt
mc cp /tmp/opencode/shell_probe.txt EmblArendtS3/platybrowser-2025/_write_probe.txt
mc rm EmblArendtS3/platybrowser-2025/_write_probe.txt
```

Expected: both commands succeed. If `mc cp` reports `Access Denied`, STOP and report to the user — do not proceed with the upload tasks.

- [ ] **Step 2: Mirror the shell N5 locally**

```bash
mkdir -p tmp_shell_halves_src
mc mirror --overwrite \
  EmblArendtS3/platybrowser/rawdata/sbem-6dpf-1-whole-segmented-shell.n5/ \
  tmp_shell_halves_src/sbem-6dpf-1-whole-segmented-shell.n5/
```

Expected: `tmp_shell_halves_src/sbem-6dpf-1-whole-segmented-shell.n5/setup0/timepoint0/s0..s4` exist. Verify with `ls tmp_shell_halves_src/sbem-6dpf-1-whole-segmented-shell.n5/setup0/timepoint0/`.

- [ ] **Step 3: Generate the four half-shell N5s and the XMLs**

```bash
uv run --python 3.12 2025_scripts/generate_shell_halves.py \
  --mask tmp_shell_halves_src/sbem-6dpf-1-whole-segmented-shell.n5 \
  --stage-dir data/rawdata/shell_halves \
  --local-xml-dir data/platybrowser_6dpf/images/local \
  --s3-xml-dir data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves
```

Expected: `Wrote 4 half-shell N5s to data/rawdata/shell_halves`.

- [ ] **Step 4: Verify outputs locally**

```bash
uv run --python 3.12 --with z5py --with numpy python -c "
import z5py, numpy as np
src = z5py.File('tmp_shell_halves_src/sbem-6dpf-1-whole-segmented-shell.n5')
stp = src['setup0/timepoint0']
src_total = int(stp['s0'][:].sum())
levels = sorted(stp.keys())
for name in ['shell_left','shell_right','shell_front','shell_back']:
    f = z5py.File(f'data/rawdata/shell_halves/{name}.n5')
    tp = f['setup0/timepoint0']
    assert sorted(tp.keys()) == levels, (name, sorted(tp.keys()))
    for k in levels:
        assert tuple(tp[k].shape) == tuple(stp[k].shape), (name, k)
        assert tp[k].dtype == np.dtype('uint8'), (name, k, tp[k].dtype)
    kept = int(tp['s0'][:].sum())
    assert 0 < kept < src_total, (name, kept, src_total)
    print(name, 'ok', tp['s0'].shape, 'voxels kept', kept)
    f.close()
print('all four outputs valid')
"
```

Expected: four `... ok` lines and `all four outputs valid`. Each kept-voxel count must be strictly between 0 and the source total.

- [ ] **Step 5: Confirm the original shell on S3 is untouched**

```bash
mc stat EmblArendtS3/platybrowser/rawdata/sbem-6dpf-1-whole-segmented-shell.n5/setup0/attributes.json
```

Expected: the object still exists with its original timestamp/size (no rewrite). No commit in this task.

---

### Task 6: Upload to `platybrowser-2025` and verify remotely

**Files:**
- Generated (gitignored): none new.

**Interfaces:**
- Consumes: `data/rawdata/shell_halves/*.n5` from Task 5.

- [ ] **Step 1: Upload the four N5s**

```bash
mc mirror --overwrite \
  data/rawdata/shell_halves/ \
  EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/
```

Expected: 4 `.n5` folders transferred, no errors.

- [ ] **Step 2: Verify the uploaded objects**

```bash
mc ls EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/
mc ls EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/shell_left.n5/setup0/timepoint0/
mc cat EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/shell_left.n5/setup0/attributes.json
mc cat EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/shell_left.n5/setup0/timepoint0/s0/attributes.json
```

Expected: four `*.n5/` entries; `s0`..`s4` present; setup0 attrs contain `"dataType":"uint8"` and 5 `downsamplingFactors`; s0 attrs contain `"dimensions":[860,810,714]`.

- [ ] **Step 3: Confirm the new prefix did not disturb existing keys**

```bash
mc ls EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/
```

Expected: `celltype_proba/`, `coregulon_proba/`, and the new `shell_halves/` only.

---

### Task 7: Wire sources and views into `dataset.json`, validate, commit

**Files:**
- Modify: `data/platybrowser_6dpf/dataset.json`
- Add (generated in Task 5): 4 local XMLs + 4 S3 XMLs.

**Interfaces:**
- Consumes: the XMLs and N5s from Tasks 5–6.

- [ ] **Step 1: Add the four image sources**

In `data/platybrowser_6dpf/dataset.json`, immediately after the closing `},` of the existing `"shell"` source (the block ending at the line before `"brn3a":`), insert:

```json
    "shell_left": {
      "image": {
        "imageData": {
          "bdv.n5": {
            "relativePath": "images/local/shell_left.xml"
          },
          "bdv.n5.s3": {
            "relativePath": "images/bdv-n5-s3/shell_halves/shell_left.xml"
          }
        }
      }
    },
    "shell_right": {
      "image": {
        "imageData": {
          "bdv.n5": {
            "relativePath": "images/local/shell_right.xml"
          },
          "bdv.n5.s3": {
            "relativePath": "images/bdv-n5-s3/shell_halves/shell_right.xml"
          }
        }
      }
    },
    "shell_front": {
      "image": {
        "imageData": {
          "bdv.n5": {
            "relativePath": "images/local/shell_front.xml"
          },
          "bdv.n5.s3": {
            "relativePath": "images/bdv-n5-s3/shell_halves/shell_front.xml"
          }
        }
      }
    },
    "shell_back": {
      "image": {
        "imageData": {
          "bdv.n5": {
            "relativePath": "images/local/shell_back.xml"
          },
          "bdv.n5.s3": {
            "relativePath": "images/bdv-n5-s3/shell_halves/shell_back.xml"
          }
        }
      }
    },
```

- [ ] **Step 2: Add the four views**

Immediately after the closing `},` of the existing `"shell"` view (the block ending at the line before `"virtual-cells":`), insert:

```json
    "shell_left": {
      "uiSelectionGroup": "sbem",
      "sourceDisplays": [
        {
          "imageDisplay": {
            "sources": [
              "shell_left"
            ],
            "contrastLimits": [
              0.0,
              1.0
            ],
            "name": "shell_left"
          }
        }
      ]
    },
    "shell_right": {
      "uiSelectionGroup": "sbem",
      "sourceDisplays": [
        {
          "imageDisplay": {
            "sources": [
              "shell_right"
            ],
            "contrastLimits": [
              0.0,
              1.0
            ],
            "name": "shell_right"
          }
        }
      ]
    },
    "shell_front": {
      "uiSelectionGroup": "sbem",
      "sourceDisplays": [
        {
          "imageDisplay": {
            "sources": [
              "shell_front"
            ],
            "contrastLimits": [
              0.0,
              1.0
            ],
            "name": "shell_front"
          }
        }
      ]
    },
    "shell_back": {
      "uiSelectionGroup": "sbem",
      "sourceDisplays": [
        {
          "imageDisplay": {
            "sources": [
              "shell_back"
            ],
            "contrastLimits": [
              0.0,
              1.0
            ],
            "name": "shell_back"
          }
        }
      ]
    },
```

- [ ] **Step 3: Validate structure and compression**

```bash
python3 2025_scripts/validate_dataset_json.py
python3 2025_scripts/compress_dataset_json.py --check
```

Expected: `data/platybrowser_6dpf/dataset.json: valid` and no compression diff.

- [ ] **Step 4: Commit the script, tests, XMLs and dataset.json**

```bash
git add 2025_scripts/generate_shell_halves.py \
        2025_scripts/tests/test_generate_shell_halves.py \
        data/platybrowser_6dpf/dataset.json \
        data/platybrowser_6dpf/images/local/shell_left.xml \
        data/platybrowser_6dpf/images/local/shell_right.xml \
        data/platybrowser_6dpf/images/local/shell_front.xml \
        data/platybrowser_6dpf/images/local/shell_back.xml \
        data/platybrowser_6dpf/images/bdv-n5-s3/shell_halves/
git commit -m "Add shell sagittal/coronal half sources and views"
```

Expected: pre-commit hook runs compress + validate and passes; commit created.

- [ ] **Step 5: Final verification**

```bash
git status --short
python3 2025_scripts/validate_dataset_json.py
mc ls EmblArendtS3/platybrowser-2025/images/bdv-n5-s3/shell_halves/
```

Expected: clean working tree (except ignored `tmp*` / `data/rawdata/shell_halves/*.n5`); dataset valid; four N5 folders on S3. Then load the four views in MoBIE (user-facing gate).
