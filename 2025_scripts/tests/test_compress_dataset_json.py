import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compress_dataset_json import (
    compress_dataset,
    compress_display,
    compress_view,
    count_keys,
)


class TestImageDisplay(unittest.TestCase):
    def test_defaults_are_stripped(self):
        d = compress_display("imageDisplay", {
            "sources": ["ache"],
            "color": "white",
            "opacity": 1.0,
            "visible": True,
            "invert": False,
            "showImagesIn3d": False,
            "blendingMode": "sum",
            "name": "ache",
            "contrastLimits": [0.0, 1000.0],
        })
        self.assertEqual(d, {"sources": ["ache"], "name": "ache", "contrastLimits": [0.0, 1000.0]})

    def test_name_never_stripped_even_when_equals_source(self):
        # `name` is the viewer UI panel label (no default; getName() has no
        # fallback), so it must survive compression even when == sources[0].
        d = compress_display("imageDisplay", {"sources": ["cells"], "name": "cells"})
        self.assertEqual(d, {"sources": ["cells"], "name": "cells"})
        d = compress_display("segmentationDisplay", {"sources": ["nuclei"], "name": "nuclei"})
        self.assertEqual(d, {"sources": ["nuclei"], "name": "nuclei"})

    def test_white_color_rgb255_is_stripped(self):
        d = compress_display("imageDisplay", {
            "sources": ["ache"], "color": "r=255,g=255,b=255,a=255",
        })
        self.assertEqual(d, {"sources": ["ache"]})

    def test_custom_color_kept(self):
        d = compress_display("imageDisplay", {
            "sources": ["ache"], "color": "magenta",
        })
        self.assertEqual(d, {"sources": ["ache"], "color": "magenta"})

    def test_contrast_limits_never_stripped(self):
        d = compress_display("imageDisplay", {
            "sources": ["raw"], "contrastLimits": [0.0, 255.0],
        })
        self.assertEqual(d, {"sources": ["raw"], "contrastLimits": [0.0, 255.0]})

    def test_custom_name_kept(self):
        d = compress_display("imageDisplay", {
            "sources": ["ache"], "name": "custom label",
        })
        self.assertEqual(d, {"sources": ["ache"], "name": "custom label"})


class TestAnnotationDisplay(unittest.TestCase):
    def test_defaults_are_stripped(self):
        d = compress_display("segmentationDisplay", {
            "sources": ["cells"],
            "visible": True,
            "opacity": 0.5,
            "showTable": True,
            "lut": "glasbey",
            "showAsBoundaries": False,
            "boundaryThickness": 1.0,
            "showScatterPlot": False,
            "scatterPlotAxes": ["anchor_x", "anchor_y"],
            "showSelectedSegmentsIn3d": False,
            "randomColorSeed": 42,
            "opacityNotSelected": 0.15,
            "blendingMode": "alpha",
            "name": "cells",
            "selectedSegmentIds": ["cells;0;1"],
        })
        self.assertEqual(d, {"sources": ["cells"], "name": "cells", "selectedSegmentIds": ["cells;0;1"]})

    def test_non_default_opacity_kept(self):
        d = compress_display("segmentationDisplay", {
            "sources": ["cells"], "opacity": 0.8,
        })
        self.assertEqual(d, {"sources": ["cells"], "opacity": 0.8})


class TestViewLevel(unittest.TestCase):
    def test_is_exclusive_false_stripped(self):
        v = compress_view({"isExclusive": False, "uiSelectionGroup": "prospr"})
        self.assertEqual(v, {"uiSelectionGroup": "prospr"})

    def test_is_exclusive_true_kept(self):
        v = compress_view({"isExclusive": True})
        self.assertEqual(v, {"isExclusive": True})

    def test_empty_source_transforms_stripped(self):
        v = compress_view({"sourceTransforms": []})
        self.assertEqual(v, {})

    def test_timepoint_zero_stripped(self):
        v = compress_view({"viewerTransform": {"normalizedAffine": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], "timepoint": 0}})
        self.assertEqual(v, {"viewerTransform": {"normalizedAffine": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]}})

    def test_timepoint_nonzero_kept(self):
        v = compress_view({"viewerTransform": {"timepoint": 3}})
        self.assertEqual(v, {"viewerTransform": {"timepoint": 3}})

    def test_viewer_transform_not_mutated(self):
        original = {"viewerTransform": {"timepoint": 0, "zoom": 1}}
        compress_view(original)
        self.assertEqual(original, {"viewerTransform": {"timepoint": 0, "zoom": 1}})


class TestDataset(unittest.TestCase):
    def test_compress_applies_to_all_views(self):
        data = {
            "is2D": False,
            "sources": {},
            "views": {
                "a": {"sourceDisplays": [{"imageDisplay": {"sources": ["x"], "opacity": 1.0, "visible": True}}]},
                "b": {"isExclusive": False},
            },
        }
        out = compress_dataset(data)
        self.assertEqual(out["views"]["a"], {"sourceDisplays": [{"imageDisplay": {"sources": ["x"]}}]})
        self.assertEqual(out["views"]["b"], {})
        # original untouched
        self.assertIn("opacity", data["views"]["a"]["sourceDisplays"][0]["imageDisplay"])

    def test_count_keys(self):
        obj = {"a": 1, "b": {"c": 2, "d": [{"e": 3}]}}
        self.assertEqual(count_keys(obj), 5)


class TestCLI(unittest.TestCase):
    def _write(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def test_check_passes_on_compressed(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            self._write(p, {"views": {"a": {"sourceDisplays": [
                {"imageDisplay": {"sources": ["x"]}}]}}})
            rc = main(["--check", "--path", str(p)])
            self.assertEqual(rc, 0)

    def test_check_fails_on_uncompressed(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            self._write(p, {"views": {"a": {"isExclusive": False}}})
            rc = main(["--check", "--path", str(p)])
            self.assertEqual(rc, 1)

    def test_compress_writes_and_is_idempotent(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            verbose = {"views": {"a": {"isExclusive": False, "sourceDisplays": [
                {"segmentationDisplay": {"sources": ["cells"], "opacity": 0.5, "name": "cells"}}]}}}
            self._write(p, verbose)
            rc = main(["--path", str(p)])
            self.assertEqual(rc, 0)
            compressed = json.loads(p.read_text())
            self.assertEqual(compressed["views"]["a"], {"sourceDisplays": [
                {"segmentationDisplay": {"sources": ["cells"], "name": "cells"}}]})
            # idempotent
            rc2 = main(["--check", "--path", str(p)])
            self.assertEqual(rc2, 0)

    def test_missing_file_returns_1(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            rc = main(["--path", str(Path(td) / "nope.json")])
            self.assertEqual(rc, 1)

    def test_invalid_json_returns_1(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            p.write_text("{not json")
            rc = main(["--path", str(p)])
            self.assertEqual(rc, 1)

    def test_directory_path_returns_1(self):
        import tempfile
        from compress_dataset_json import main
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "somedir"
            d.mkdir()
            rc = main(["--path", str(d)])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
