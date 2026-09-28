import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import z5py

from generate_shell_halves import (
    HALVES,
    HALF_NAMES,
    _level_ds_factor,
    block_slices,
    main,
    mask_block,
    mirror_group_attrs,
    mirror_level_info,
    shell_frame,
    write_halves,
    write_local_xmls,
    write_s3_xmls,
)


def make_mask_n5(path: Path, shape=(3, 4, 5), value=255) -> Path:
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
        ds[:] = value
    return path


def make_two_level_mask_n5(path: Path, shape=(4, 4, 4), value=255) -> Path:
    """Two-level uint8 mask: s0 at ds=1, s1 at ds=2."""
    coarse = tuple((s + 1) // 2 for s in shape)
    with z5py.File(str(path), "a") as f:
        setup = f.create_group("setup0")
        setup.attrs["dataType"] = "uint8"
        setup.attrs["downsamplingFactors"] = [[1, 1, 1], [2, 2, 2]]
        tp = setup.create_group("timepoint0")
        tp.attrs["multiScale"] = True
        tp.attrs["resolution"] = [0.32, 0.32, 0.4]
        d0 = tp.create_dataset("s0", shape=shape, chunks=(2, 2, 2),
                               dtype="uint8", compression="gzip", level=1,
                               fillvalue=0)
        d0.attrs["downsamplingFactors"] = [1, 1, 1]
        d0[:] = value
        d1 = tp.create_dataset("s1", shape=coarse, chunks=(2, 2, 2),
                               dtype="uint8", compression="gzip", level=1,
                               fillvalue=0)
        d1.attrs["downsamplingFactors"] = [2, 2, 2]
        d1[:] = value
    return path


class TestMaskBlock(unittest.TestCase):
    def test_keeps_positive_side(self):
        block = np.ones((1, 3, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, 0, np.array([1.0, 0.0, 0.0]),
                         np.array([2.0, 0.0, 0.0]), True, 1.0)
        self.assertEqual(list(out[0, 0]), [0, 0, 1, 1, 1])

    def test_keeps_negative_side(self):
        block = np.ones((1, 3, 5), dtype=np.uint8)
        out = mask_block(block, 0, 0, 0, np.array([1.0, 0.0, 0.0]),
                         np.array([2.0, 0.0, 0.0]), False, 1.0)
        self.assertEqual(list(out[0, 0]), [1, 1, 1, 0, 0])

    def test_downsampling_maps_level_index_to_full_res(self):
        block = np.ones((1, 1, 4), dtype=np.uint8)
        out = mask_block(block, 0, 0, 0, np.array([1.0, 0.0, 0.0]),
                         np.array([4.0, 0.0, 0.0]), True, 2.0)
        # X = 0.5, 2.5, 4.5, 6.5 ; keep X >= 4 -> [0, 0, 1, 1]
        self.assertEqual(list(out[0, 0]), [0, 0, 1, 1])

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

    def test_half_names_and_order(self):
        self.assertEqual(
            HALF_NAMES,
            ["shell_left", "shell_right", "shell_front", "shell_back"],
        )


class TestShellFrame(unittest.TestCase):
    def test_recovers_principal_axes(self):
        # Solid ellipsoid: long along z (AP), wider along (1,-1,0) (LR) than
        # along (1,1,0) (DV).
        n = 41
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(np.float64)
        c = (n - 1) / 2.0
        u_ap = np.array([0.0, 0.0, 1.0])
        u_lr = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        u_dv = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
        pts = np.stack([xx - c, yy - c, zz - c], axis=-1)
        q = ((pts @ u_ap / 12.0) ** 2 + (pts @ u_lr / 8.0) ** 2
             + (pts @ u_dv / 4.0) ** 2)
        mask = (q <= 1.0).astype(np.uint8) * 255
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ellipsoid.n5"
            with z5py.File(str(path), "a") as f:
                setup = f.create_group("setup0")
                setup.attrs["dataType"] = "uint8"
                tp = setup.create_group("timepoint0")
                ds = tp.create_dataset("s0", shape=mask.shape, chunks=(8, 8, 8),
                                       dtype="uint8", compression="gzip",
                                       level=1, fillvalue=0)
                ds[:] = mask
            frame = shell_frame(path)
            self.assertLess(
                float(np.linalg.norm(frame["center"] - np.array([c, c, c]))), 1.0)
            self.assertGreater(abs(float(frame["lr"] @ u_lr)), 0.98)
            self.assertGreater(abs(float(frame["dv"] @ u_dv)), 0.98)
            self.assertGreaterEqual(
                float(frame["lr"] @ np.array([1.0, -1.0, 0.0])), 0.0)
            self.assertGreaterEqual(
                float(frame["dv"] @ np.array([1.0, 1.0, 0.0])), 0.0)

    def test_shell_frame_rejects_empty_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = make_mask_n5(Path(tmp) / "m.n5", value=0)
            with self.assertRaises(ValueError):
                shell_frame(mask)


class TestWriteHalves(unittest.TestCase):
    def test_writes_four_masked_uint8_n5s(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "mask.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            frame = shell_frame(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs, frame, gzip_level=1)

            self.assertEqual(
                sorted(p.name for p in stage.glob("*.n5")),
                sorted(f"{n}.n5" for n in HALF_NAMES),
            )
            with z5py.File(str(mask), "r") as f:
                data0 = f["setup0/timepoint0/s0"][:]
            for name, (axis_key, keep_positive) in HALVES.items():
                with z5py.File(str(stage / f"{name}.n5"), "r") as f:
                    ds = f["setup0/timepoint0/s0"]
                    self.assertEqual(tuple(ds.shape), (3, 4, 5))
                    self.assertEqual(ds.dtype, np.dtype("uint8"))
                    out = ds[:]
                expected = mask_block(
                    data0, 0, 0, 0, frame[axis_key], frame["center"],
                    keep_positive, 1.0,
                )
                np.testing.assert_array_equal(out, expected)
                self.assertTrue(out.any(), name)

    def test_group_attrs_mirrored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "mask.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs, shell_frame(mask))
            with z5py.File(str(stage / "shell_left.n5"), "r") as f:
                self.assertEqual(dict(f["setup0"].attrs)["dataType"], "uint8")
                self.assertEqual(
                    list(dict(f["setup0/timepoint0"].attrs)["resolution"]),
                    [0.32, 0.32, 0.4],
                )
                self.assertIs(dict(f["setup0/timepoint0"].attrs)["multiScale"], True)
                self.assertEqual(
                    list(dict(f["setup0/timepoint0/s0"].attrs)["downsamplingFactors"]),
                    [1, 1, 1],
                )
                ds = f["setup0/timepoint0/s0"]
                self.assertEqual(tuple(ds.chunks), (2, 2, 2))
                self.assertEqual(ds.compression, "gzip")

    def test_level_ds_factor_reads_per_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = make_two_level_mask_n5(Path(tmp) / "m.n5")
            levels = {lvl["name"]: lvl for lvl in mirror_level_info(mask)}
            self.assertEqual(_level_ds_factor(levels["s0"]), 1.0)
            self.assertEqual(_level_ds_factor(levels["s1"]), 2.0)

    def test_write_halves_masks_coarse_level_at_correct_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_two_level_mask_n5(tmp / "m.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            frame = shell_frame(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs, frame)
            with z5py.File(str(mask), "r") as f:
                src1 = f["setup0/timepoint0/s1"][:]
            for name, (axis_key, keep_positive) in HALVES.items():
                with z5py.File(str(stage / f"{name}.n5"), "r") as f:
                    out1 = f["setup0/timepoint0/s1"][:]
                expected = mask_block(src1, 0, 0, 0, frame[axis_key],
                                      frame["center"], keep_positive, 2.0)
                np.testing.assert_array_equal(out1, expected)

    def test_left_right_partition_covers_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mask = make_mask_n5(tmp / "m.n5")
            levels = mirror_level_info(mask)
            attrs = mirror_group_attrs(mask)
            frame = shell_frame(mask)
            stage = tmp / "stage"
            write_halves(mask, stage, levels, attrs, frame)
            with z5py.File(str(mask), "r") as f:
                src = f["setup0/timepoint0/s0"][:]
            out = {}
            for name in ("shell_left", "shell_right"):
                with z5py.File(str(stage / f"{name}.n5"), "r") as f:
                    out[name] = f["setup0/timepoint0/s0"][:]
            union = (out["shell_left"] > 0) | (out["shell_right"] > 0)
            np.testing.assert_array_equal(union, src > 0)


class TestXml(unittest.TestCase):
    def test_local_xml_points_at_staged_n5(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            stage = tmp / "rawdata" / "shell_halves"
            local = tmp / "images" / "local"
            out = write_local_xmls(["shell_left"], local, stage)
            root = ET.parse(out[0]).getroot()
            self.assertEqual(root.find(".//ViewSetup/name").text, "shell_left")
            self.assertEqual(root.find(".//ImageLoader").get("format"), "bdv.n5")
            n5 = root.find(".//ImageLoader/n5")
            self.assertEqual(n5.get("type"), "relative")
            self.assertFalse(n5.text.startswith("/"))
            self.assertTrue(
                n5.text.replace("\\", "/").endswith(
                    "rawdata/shell_halves/shell_left.n5"),
                n5.text,
            )
            self.assertEqual(root.find(".//Attributes/Channel/name").text, "0")

    def test_s3_xml_has_bucket_key_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_s3_xmls(["shell_back"], Path(tmp) / "s3")
            root = ET.parse(out[0]).getroot()
            self.assertEqual(root.find(".//ViewSetup/name").text, "shell_back")
            self.assertEqual(root.find(".//ImageLoader").get("format"), "bdv.n5.s3")
            self.assertEqual(
                root.find(".//Key").text,
                "images/bdv-n5-s3/shell_halves/shell_back.n5",
            )
            self.assertEqual(root.find(".//BucketName").text, "platybrowser-2025")
            self.assertEqual(
                root.find(".//ServiceEndpoint").text, "https://s3.embl.de")
            self.assertEqual(root.find(".//SigningRegion").text, "us-west-2")
            self.assertEqual(root.find(".//Attributes/Channel/name").text, "0")


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
            self.assertEqual(
                sorted(p.name for p in stage.glob("*.n5")),
                sorted(f"{n}.n5" for n in HALF_NAMES),
            )
            self.assertEqual(
                sorted(p.name for p in local.glob("*.xml")),
                sorted(f"{n}.xml" for n in HALF_NAMES),
            )
            self.assertEqual(
                sorted(p.name for p in s3.glob("*.xml")),
                sorted(f"{n}.xml" for n in HALF_NAMES),
            )


if __name__ == "__main__":
    unittest.main()
