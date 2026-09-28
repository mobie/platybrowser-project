import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import z5py

from generate_nuclei_proba_images import (
    read_subtype_columns,
    read_probabilities,
    build_value_table,
    mirror_level_info,
    mirror_group_attrs,
    relabel_block,
    write_outputs,
    write_local_xmls,
    report_sizes,
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


def make_mask_n5(path: Path) -> Path:
    """Minimal 2-level nuclei-like mask: s0 16^3, s1 8^3, labels {1, 2}."""
    import z5py
    with z5py.File(str(path), "a") as f:
        setup = f.create_group("setup0")
        setup.attrs["dataType"] = "uint16"
        setup.attrs["downsamplingFactors"] = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        tp = setup.create_group("timepoint0")
        tp.attrs["multiScale"] = True
        tp.attrs["resolution"] = [0.08, 0.08, 0.1]
        s0 = tp.create_dataset("s0", shape=(16, 16, 16), chunks=(8, 8, 8),
                               dtype="uint32", compression="gzip", fillvalue=0)
        s0.attrs["resolution"] = [0.08, 0.08, 0.1]
        s0.attrs["downsamplingFactors"] = [1, 1, 1]
        s0.attrs["offset"] = [0.0, 0.0, 0.0]
        a0 = np.zeros((16, 16, 16), dtype=np.uint32)
        a0[2:5, 2:5, 2:5] = 1
        a0[10:13, 10:13, 10:13] = 2
        s0[...] = a0
        s1 = tp.create_dataset("s1", shape=(8, 8, 8), chunks=(8, 8, 8),
                               dtype="uint32", compression="gzip", fillvalue=0)
        s1.attrs["resolution"] = [0.08, 0.08, 0.1]
        s1.attrs["downsamplingFactors"] = [2, 2, 2]
        s1.attrs["offset"] = [0.0, 0.0, 0.0]
        a1 = np.zeros((8, 8, 8), dtype=np.uint32)
        a1[1:3, 1:3, 1:3] = 1
        a1[5:7, 5:7, 5:7] = 2
        s1[...] = a1
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


class TestMirrorLevelInfo(unittest.TestCase):
    def test_mirror_level_info_reads_pyramid(self):
        with tempfile.TemporaryDirectory() as d:
            mask = make_mask_n5(Path(d) / "mask.n5")
            levels = mirror_level_info(mask)
            self.assertEqual([lv["name"] for lv in levels], ["s0", "s1"])
            self.assertEqual(levels[0]["shape"], (16, 16, 16))
            self.assertEqual(levels[0]["chunks"], (8, 8, 8))
            self.assertEqual(levels[1]["shape"], (8, 8, 8))
            self.assertEqual(levels[1]["attrs"]["downsamplingFactors"], [2, 2, 2])


class TestMirrorGroupAttrs(unittest.TestCase):
    def test_mirror_group_attrs_reads_setup0_and_timepoint0(self):
        with tempfile.TemporaryDirectory() as d:
            mask = make_mask_n5(Path(d) / "mask.n5")
            attrs = mirror_group_attrs(mask)
            self.assertEqual(attrs["setup0"]["dataType"], "uint16")
            self.assertEqual(attrs["setup0"]["downsamplingFactors"],
                             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
            self.assertEqual(attrs["timepoint0"]["multiScale"], True)
            self.assertEqual(attrs["timepoint0"]["resolution"], [0.08, 0.08, 0.1])


class TestRelabelBlock(unittest.TestCase):
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
            group_attrs = mirror_group_attrs(mask)
            written = write_outputs(mask, value_tables, stage, levels, group_attrs)
            self.assertEqual(len(written), 2)
            self.assertTrue((stage / "clade1sub1_proba.n5").is_dir())

            with z5py.File(str(stage / "clade1sub1_proba.n5"), "r") as f:
                s0 = f["setup0"]
                self.assertEqual(s0.attrs["dataType"], "uint16")
                self.assertEqual(s0.attrs["downsamplingFactors"],
                                 [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
                tp = f["setup0/timepoint0"]
                self.assertEqual(tp.attrs["multiScale"], True)
                self.assertEqual(tp.attrs["resolution"], [0.08, 0.08, 0.1])
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


class TestWriteOutputsParallel(unittest.TestCase):
    def test_parallel_block_fill_matches_sequential(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            mask = make_mask_n5(d / "mask.n5")
            levels = mirror_level_info(mask)
            group_attrs = mirror_group_attrs(mask)
            value_tables = {
                "clade1sub1": build_value_table({1: 0.95, 2: 0.4996}),
                "clade6sub19": build_value_table({1: 0.0003, 2: 0.9996}),
                "nocladesub3": build_value_table({1: 0.0, 2: 1.0}),
            }
            seq_stage = d / "seq"
            par_stage = d / "par"
            write_outputs(mask, value_tables, seq_stage, levels, group_attrs,
                          n_workers=1, gzip_level=1)
            write_outputs(mask, value_tables, par_stage, levels, group_attrs,
                          n_workers=2, gzip_level=1)

            for subtype in value_tables:
                for level in levels:
                    name = level["name"]
                    with z5py.File(str(seq_stage / f"{subtype}_proba.n5"), "r") as f:
                        seq_ds = f["setup0/timepoint0"][name]
                        seq_arr = seq_ds[...]
                        seq_attrs = dict(seq_ds.attrs)
                    with z5py.File(str(par_stage / f"{subtype}_proba.n5"), "r") as f:
                        par_ds = f["setup0/timepoint0"][name]
                        par_arr = par_ds[...]
                        par_attrs = dict(par_ds.attrs)
                    self.assertTrue(np.array_equal(seq_arr, par_arr),
                                    f"{subtype} {name}: arrays differ")
                    self.assertEqual(seq_attrs, par_attrs,
                                     f"{subtype} {name}: attrs differ")
                with z5py.File(str(seq_stage / f"{subtype}_proba.n5"), "r") as f:
                    seq_groups = (dict(f["setup0"].attrs),
                                  dict(f["setup0/timepoint0"].attrs))
                with z5py.File(str(par_stage / f"{subtype}_proba.n5"), "r") as f:
                    par_groups = (dict(f["setup0"].attrs),
                                  dict(f["setup0/timepoint0"].attrs))
                self.assertEqual(seq_groups, par_groups,
                                 f"{subtype}: group attrs differ")


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
