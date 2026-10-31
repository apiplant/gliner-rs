"""Regression checks: the optional GPU tie allowance must never hide decoding changes."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_apple_results.py"
SPEC = importlib.util.spec_from_file_location("compare_apple_results", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def mention(text, start, confidence=0.9, **attributes):
    return dict(text=text, start=start, end=start + len(text), confidence=confidence, **attributes)


def entities(*values):
    return {"entities": {"location": list(values)}}


class CompareAppleResultsTests(unittest.TestCase):
    def compare(self, left, right, **kwargs):
        return MODULE.compare_values(left, right, **kwargs)

    def test_float_tolerance_and_typed_primitives(self):
        self.assertFalse(self.compare({"confidence": 0.9}, {"confidence": 0.90001}).errors)
        self.assertTrue(self.compare({"confidence": 0.9}, {"confidence": 0.901}).errors)
        self.assertTrue(self.compare({"value": True}, {"value": 1}).errors)
        self.assertTrue(self.compare({"value": 1}, {"value": 1.0}).errors)
        self.assertTrue(self.compare({"value": "1"}, {"value": 1}).errors)
        self.assertTrue(self.compare({"value": None}, {"value": []}).errors)

    def test_floating_span_coordinates_never_receive_numeric_tolerance(self):
        a = mention("A", 0); a["start"] = 0.0
        b = copy.deepcopy(a); b["start"] = 0.00001
        self.assertTrue(self.compare(entities(a), entities(b)).errors)
        self.assertTrue(self.compare(entities(a), entities(a)).errors)

    def test_overflowing_finite_confidence_delta_is_a_failure(self):
        summary = MODULE.compare_cases([{"confidence": 1e308}], [{"confidence": -1e308}])
        self.assertTrue(summary["errors"])
        self.assertIsNone(summary["max_confidence_delta"])
        json.dumps(summary, allow_nan=False)

    def test_nonfinite_values_always_fail(self):
        for value in (math.nan, math.inf, -math.inf):
            for allow in (False, True):
                result = self.compare(entities(mention("A", 0, value)), entities(mention("A", 0, value)), allow_near_ties=allow)
                self.assertTrue(result.errors)
                self.assertIn("non-finite", result.errors[0])
        left = entities(mention("A", 0, quality={"confidence": math.nan}))
        self.assertTrue(self.compare(left, copy.deepcopy(left), allow_near_ties=True).errors)

    def test_default_order_is_exact_even_for_exact_ties(self):
        a, b = mention("Berlin", 2), mention("Berlin", 30)
        self.assertTrue(self.compare(entities(a, b), entities(b, a)).errors)
        self.assertFalse(self.compare(entities(a, b), entities(b, a), allow_near_ties=True).errors)

    def test_near_tie_is_reported_with_both_original_orders(self):
        a, b = mention("Berlin", 2, 0.99991977), mention("Berlin", 30, 0.99991965)
        candidate_a, candidate_b = copy.deepcopy(a), copy.deepcopy(b)
        candidate_b["confidence"] = candidate_a["confidence"]
        result = self.compare(entities(a, b), entities(candidate_b, candidate_a), allow_near_ties=True)
        self.assertFalse(result.errors)
        self.assertEqual(len(result.reorders), 1)
        reorder = result.reorders[0]
        self.assertEqual(reorder["path"], "$.entities.location")
        self.assertEqual((reorder["start_index"], reorder["end_index_exclusive"]), (0, 2))
        self.assertEqual(reorder["left_order"], [["Berlin", 2, 8], ["Berlin", 30, 36]])
        self.assertEqual(reorder["right_order"], list(reversed(reorder["left_order"])))
        self.assertAlmostEqual(reorder["max_confidence_delta"], 1.2e-7)

    def test_three_item_permutation_is_one_contiguous_block(self):
        a, b, c = mention("A", 0, 0.9), mention("B", 3, 0.9000003), mention("C", 6, 0.9000006)
        result = self.compare(entities(a, b, c), entities(c, a, b), allow_near_ties=True)
        self.assertFalse(result.errors)
        self.assertEqual([(r["start_index"], r["end_index_exclusive"]) for r in result.reorders], [(0, 3)])

    def test_separate_near_tie_blocks_remain_separate(self):
        a, b, c, d = mention("A", 0, 0.99), mention("B", 3, 0.9900001), mention("C", 6, 0.8), mention("D", 9, 0.8000001)
        result = self.compare(entities(a, b, c, d), entities(b, a, d, c), allow_near_ties=True)
        self.assertFalse(result.errors)
        self.assertEqual([(r["start_index"], r["end_index_exclusive"]) for r in result.reorders], [(0, 2), (2, 4)])

    def test_range_must_be_small_on_both_sides(self):
        a, b = mention("A", 0, 0.9), mention("B", 3, 0.9000001)
        wide = copy.deepcopy(b); wide["confidence"] = 0.90001
        for left, right in ((entities(a, b), entities(wide, a)), (entities(a, wide), entities(b, a))):
            result = self.compare(left, right, allow_near_ties=True)
            self.assertTrue(result.errors)
            self.assertEqual(result.reorders, [])
            self.assertIn("confidence ranges", result.errors[0])

    def test_adjacent_chain_cannot_expand_the_range_limit(self):
        a, b, c = mention("A", 0, 0.9), mention("B", 3, 0.9000008), mention("C", 6, 0.9000016)
        self.assertTrue(self.compare(entities(a, b, c), entities(c, a, b), allow_near_ties=True).errors)

    def test_noncontiguous_swap_cannot_cross_a_different_confidence(self):
        a, b, c = mention("A", 0, 0.9), mention("B", 3, 0.7), mention("C", 6, 0.9000001)
        self.assertTrue(self.compare(entities(a, b, c), entities(c, b, a), allow_near_ties=True).errors)

    def test_confidence_outside_reordered_block_stays_strict(self):
        a, b, c = mention("A", 0), mention("B", 3), mention("C", 6, 0.7)
        changed = copy.deepcopy(c); changed["confidence"] = 0.701
        result = self.compare(entities(a, b, c), entities(b, a, changed), allow_near_ties=True)
        self.assertTrue(result.errors)
        self.assertTrue(any("[2].confidence" in error for error in result.errors))

    def test_large_aligned_confidence_delta_cannot_hide_behind_small_ranges(self):
        a, b = mention("A", 0), mention("B", 3)
        x, y = mention("A", 0, 0.8), mention("B", 3, 0.8)
        result = self.compare(entities(a, b), entities(y, x), allow_near_ties=True)
        self.assertTrue(result.errors)
        self.assertEqual(result.reorders, [])

    def test_changed_count_threshold_crossing_is_a_failure(self):
        a, b = mention("A", 0, 0.5000001), mention("B", 3, 0.5)
        result = self.compare(entities(a, b), entities(a), allow_near_ties=True)
        self.assertTrue(result.errors)
        self.assertIn("length", result.errors[0])

    def test_changed_span_text_label_and_coordinates_fail(self):
        a, b = mention("Berlin", 0), mention("Berlin", 20)
        for key, value in (("text", "berlin"), ("start", 21), ("end", 27), ("start", 20.0)):
            changed = copy.deepcopy(b); changed[key] = value
            self.assertTrue(self.compare(entities(a, b), entities(changed, a), allow_near_ties=True).errors)
        self.assertTrue(self.compare(entities(a, b), {"entities": {"city": [b, a]}}, allow_near_ties=True).errors)

    def test_ambiguous_duplicate_identities_do_not_reorder(self):
        a, b = mention("A", 0, 0.9), mention("A", 0, 0.9000002)
        # Add an ordered attribute so reversing duplicates is observably wrong.
        a["tag"] = "a"; b["tag"] = "b"
        self.assertTrue(self.compare(entities(a, b), entities(b, a), allow_near_ties=True).errors)

    def test_attributes_are_compared_recursively(self):
        a = mention("A", 0, category={"label": "city", "confidence": 0.8})
        b = mention("B", 3, category={"label": "city", "confidence": 0.7})
        changed = copy.deepcopy(b); changed["category"]["confidence"] += 0.00001
        self.assertFalse(self.compare(entities(a, b), entities(changed, a), allow_near_ties=True).errors)
        changed["category"]["label"] = "country"
        self.assertTrue(self.compare(entities(a, b), entities(changed, a), allow_near_ties=True).errors)
        changed = copy.deepcopy(b); changed["category"]["confidence"] += 0.001
        self.assertTrue(self.compare(entities(a, b), entities(changed, a), allow_near_ties=True).errors)

    def test_attribute_arrays_keep_exact_order(self):
        a, b = mention("A", 0, tags=["city", "capital"]), mention("B", 3)
        changed = copy.deepcopy(a); changed["tags"].reverse()
        self.assertTrue(self.compare(entities(a, b), entities(b, changed), allow_near_ties=True).errors)

    def test_non_entity_arrays_remain_ordered_even_with_flag(self):
        a, b = mention("A", 0), mention("B", 3)
        examples = [
            ({"records": [a, b]}, {"records": [b, a]}),
            ({"relation_extraction": {"near": [a, b]}}, {"relation_extraction": {"near": [b, a]}}),
            ({"classification": [a, b]}, {"classification": [b, a]}),
            ({"wrapper": entities(a, b)}, {"wrapper": entities(b, a)}),
            ({"entities": {"location": ["A", "B"]}}, {"entities": {"location": ["B", "A"]}}),
        ]
        for left, right in examples:
            self.assertTrue(self.compare(left, right, allow_near_ties=True).errors)

    def test_missing_confidence_is_ineligible_for_reordering(self):
        a, b = mention("A", 0), mention("B", 3)
        del a["confidence"]; del b["confidence"]
        self.assertTrue(self.compare(entities(a, b), entities(b, a), allow_near_ties=True).errors)

    def test_summary_keeps_strict_count_and_reorder_case(self):
        a, b = mention("A", 0), mention("B", 3)
        summary = MODULE.compare_cases([entities(a), entities(a, b)], [entities(a), entities(b, a)], allow_near_ties=True)
        self.assertEqual((summary["strict_cases"], summary["passed"], summary["total"]), (1, 2, 2))
        self.assertEqual(summary["near_tie_reorders"][0]["case"], 1)
        mismatch = MODULE.compare_cases([{}, {}], [{}])
        self.assertEqual((mismatch["strict_cases"], mismatch["passed"], mismatch["total"]), (1, 1, 2))
        self.assertTrue(mismatch["errors"])

    def test_input_rejects_nan_invalid_json_nonobjects_and_empty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            for text in ('{"confidence":NaN}', '{"confidence":Infinity}', '[]', '', '{broken'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.read_jsonl(path)

    def test_cli_defaults_strict_and_emits_valid_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            a, b = mention("A", 0), mention("B", 3)
            left.write_text(json.dumps(entities(a, b)) + "\n", encoding="utf-8")
            right.write_text(json.dumps(entities(b, a)) + "\n", encoding="utf-8")
            strict = subprocess.run([sys.executable, str(SCRIPT), str(left), str(right)], capture_output=True, text=True)
            self.assertEqual(strict.returncode, 1)
            self.assertEqual(json.loads(strict.stdout)["passed"], 0)
            ties = subprocess.run([sys.executable, str(SCRIPT), str(left), str(right), "--allow-near-ties"], capture_output=True, text=True)
            self.assertEqual(ties.returncode, 0)
            self.assertEqual(len(json.loads(ties.stdout)["near_tie_reorders"]), 1)
            right.write_text('{"confidence":NaN}\n', encoding="utf-8")
            invalid = subprocess.run([sys.executable, str(SCRIPT), str(left), str(right)], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertTrue(json.loads(invalid.stdout)["errors"])


if __name__ == "__main__":
    unittest.main()
