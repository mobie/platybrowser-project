import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_dataset_json import main, validate


def valid_dataset():
    return {
        "is2D": False,
        "defaultLocation": {"position": [1.0, 2.0, 3.0]},
        "sources": {
            "raw": {"image": {"imageData": {"bdv.n5": {"relativePath": "x.xml"}}}},
            "cells": {"segmentation": {"imageData": {"bdv.n5": {"relativePath": "y.xml"}}}},
        },
        "views": {
            "default": {"uiSelectionGroup": "Figures Vergara2021",
                        "sourceDisplays": [{"imageDisplay": {"sources": ["raw"]}}]},
            "cells_view": {"sourceDisplays": [
                {"segmentationDisplay": {"sources": ["cells"],
                                         "selectedSegmentIds": ["cells;0;7"]}}]},
        },
    }


class TestValidate(unittest.TestCase):
    def test_valid_dataset_returns_no_errors(self):
        self.assertEqual(validate(valid_dataset()), [])

    def test_missing_top_level_keys(self):
        errors = validate({"views": {}})
        self.assertTrue(any("is2D" in e for e in errors))
        self.assertTrue(any("defaultLocation" in e for e in errors))
        self.assertTrue(any("sources" in e for e in errors))

    def test_missing_default_view(self):
        data = valid_dataset()
        del data["views"]["default"]
        errors = validate(data)
        self.assertTrue(any("default" in e for e in errors))

    def test_unknown_source_in_display(self):
        data = valid_dataset()
        data["views"]["default"]["sourceDisplays"] = [
            {"imageDisplay": {"sources": ["nonexistent"]}}]
        errors = validate(data)
        self.assertTrue(any("nonexistent" in e for e in errors))

    def test_unknown_source_in_selected_segments(self):
        data = valid_dataset()
        data["views"]["cells_view"]["sourceDisplays"] = [
            {"segmentationDisplay": {"sources": ["cells"],
                                     "selectedSegmentIds": ["nope;0;1"]}}]
        errors = validate(data)
        self.assertTrue(any("nope" in e for e in errors))

    def test_bad_selected_segment_format(self):
        data = valid_dataset()
        data["views"]["cells_view"]["sourceDisplays"] = [
            {"segmentationDisplay": {"sources": ["cells"],
                                     "selectedSegmentIds": ["cells;0"]}}]
        errors = validate(data)
        self.assertTrue(any("expected source;timepoint;id" in e for e in errors))

    def test_display_with_zero_or_two_types(self):
        data = valid_dataset()
        data["views"]["bad"] = {"sourceDisplays": [{"imageDisplay": {}, "segmentationDisplay": {}}]}
        errors = validate(data)
        self.assertTrue(any("exactly one" in e for e in errors))

    def test_bad_contrast_limits(self):
        data = valid_dataset()
        data["views"]["default"]["sourceDisplays"] = [
            {"imageDisplay": {"sources": ["raw"], "contrastLimits": [0.0]}}]
        errors = validate(data)
        self.assertTrue(any("contrastLimits" in e for e in errors))

    def test_null_source_displays_returns_error_not_crash(self):
        data = valid_dataset()
        data["views"]["default"]["sourceDisplays"] = None
        errors = validate(data)
        self.assertTrue(any("sourceDisplays must be a list" in e for e in errors))

    def test_null_sources_returns_error_not_crash(self):
        data = valid_dataset()
        data["views"]["default"]["sourceDisplays"] = [
            {"imageDisplay": {"sources": None}}]
        errors = validate(data)
        self.assertTrue(any("sources must be a list" in e for e in errors))

    def test_string_sources_returns_error_not_crash(self):
        data = valid_dataset()
        data["views"]["default"]["sourceDisplays"] = [
            {"imageDisplay": {"sources": "raw"}}]
        errors = validate(data)
        self.assertTrue(any("raw" in e for e in errors))
        # must not iterate the string character-by-character
        self.assertFalse(any("unknown source: r" in e for e in errors))


class TestValidateCLI(unittest.TestCase):
    def test_valid_file_returns_0(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            with open(p, "w") as f:
                json.dump(valid_dataset(), f)
            self.assertEqual(main(["--path", str(p)]), 0)

    def test_invalid_file_returns_1(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "dataset.json"
            with open(p, "w") as f:
                json.dump({"views": {}}, f)
            self.assertEqual(main(["--path", str(p)]), 1)

    def test_missing_file_returns_1(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(main(["--path", str(Path(td) / "nope.json")]), 1)


if __name__ == "__main__":
    unittest.main()
