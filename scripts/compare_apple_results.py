#!/usr/bin/env python3
"""Compare Apple/Rust JSONL results without hiding discrete output differences.

By default every array must retain its exact order. With --allow-near-ties,
only root $.entities.<label> arrays of objects containing text, integer start/
end, and finite confidence may reorder. Each minimal contiguous permutation
block must have confidence range <= --tie-tolerance independently on BOTH
sides, and identical unique (text,start,end) identities. Values and attributes
are then compared recursively with the ordinary numeric tolerance. Every
accepted reorder is reported with its original index interval and both orders;
records, relations, classifications and nested attribute arrays stay ordered.
No rounding, global sorting, dropped spans or threshold adjustment is applied.

The JSON summary retains strict_cases even when the optional allowance passes
additional cases. Each near_tie_reorders entry identifies a zero-based case and
[start_index,end_index_exclusive) interval, both original orders, both confidence
ranges and the aligned confidence delta. max_confidence_delta covers all aligned
confidence fields (null only if a difference overflows; that case fails). Errors
remain explicit. Exit status is 0 for agreement, 1 for mismatch, 2 for invalid
input. Numeric tolerance is absolute; integer values and types remain exact.
"""
import argparse
from dataclasses import dataclass, field
import json
import math
import sys
from typing import Optional


@dataclass
class Comparison:
    errors: list = field(default_factory=list)
    reorders: list = field(default_factory=list)
    max_confidence_delta: Optional[float] = 0.0


def _display_path(path):
    result = "$"
    for part in path:
        if type(part) is int:
            result += f"[{part}]"
        elif part.isidentifier():
            result += "." + part
        else:
            result += "[" + json.dumps(part, ensure_ascii=False) + "]"
    return result


def _finite_errors(value, path=()):
    if type(value) is float and not math.isfinite(value):
        yield f"{_display_path(path)}: non-finite number"
    elif type(value) is dict:
        for key, child in value.items():
            yield from _finite_errors(child, path + (key,))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _finite_errors(child, path + (index,))


def _span_identity(value):
    if type(value) is not dict:
        return None
    if not {"text", "start", "end", "confidence"} <= value.keys():
        return None
    if (type(value["text"]) is not str or type(value["start"]) is not int
            or type(value["end"]) is not int
            or type(value["confidence"]) not in (int, float)
            or (type(value["confidence"]) is float and not math.isfinite(value["confidence"]))):
        return None
    return value["text"], value["start"], value["end"]


def compare_values(left, right, tolerance=1e-4, allow_near_ties=False, tie_tolerance=1e-6):
    """Return errors and explicit reorder evidence for a single JSON result.

    JSON primitive types are preserved: bool is never numeric, integers remain
    exact, and integer/float changes are type mismatches. Only floats receive
    the absolute tolerance. Non-finite values fail even when both sides agree.
    """
    for name, value in (("tolerance", tolerance), ("tie_tolerance", tie_tolerance)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    result = Comparison()
    for side, value in (("left", left), ("right", right)):
        result.errors.extend(f"{side} {error}" for error in _finite_errors(value))
    if result.errors:
        return result

    def compare(a, b, path):
        label = _display_path(path)
        if type(a) is not type(b):
            result.errors.append(f"{label}: type {type(a).__name__} != {type(b).__name__}")
        elif type(a) is dict:
            for side, value in (("left", a), ("right", b)):
                if ({"text", "start", "end", "confidence"} <= value.keys()
                        and type(value["text"]) is str
                        and type(value["confidence"]) in (int, float)):
                    for coordinate in ("start", "end"):
                        if type(value[coordinate]) is not int:
                            result.errors.append(f"{label}.{coordinate}: {side} span coordinate must be an integer")
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    result.errors.append(f"{_display_path(path + (key,))}: missing on {'left' if key not in a else 'right'}")
                else:
                    compare(a[key], b[key], path + (key,))
        elif type(a) is list:
            if len(a) != len(b):
                result.errors.append(f"{label}: length {len(a)} != {len(b)}")
                return
            entity_array = len(path) == 2 and path[0] == "entities" and type(path[1]) is str
            if allow_near_ties and entity_array and reorder_entities(a, b, path):
                return
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + (index,))
        elif type(a) is float:
            delta = abs(a - b)
            if path and path[-1] == "confidence":
                if not math.isfinite(delta):
                    result.max_confidence_delta = None
                elif result.max_confidence_delta is not None:
                    result.max_confidence_delta = max(result.max_confidence_delta, delta)
            if delta > tolerance:
                result.errors.append(f"{label}: {a!r} != {b!r} (delta {delta:.9g} > {tolerance:.9g})")
        elif a != b:
            result.errors.append(f"{label}: {a!r} != {b!r}")

    def reorder_entities(a, b, path):
        identities_a = [_span_identity(x) for x in a]
        identities_b = [_span_identity(x) for x in b]
        # Ambiguous repeated identities and changed/missing spans stay strict.
        if (None in identities_a or None in identities_b
                or len(set(identities_a)) != len(a)
                or len(set(identities_b)) != len(b)
                or set(identities_a) != set(identities_b)):
            return False
        positions = {identity: index for index, identity in enumerate(identities_a)}
        i = 0
        while i < len(a):
            if identities_a[i] == identities_b[i]:
                compare(a[i], b[i], path + (i,))
                i += 1
                continue
            # Smallest closed permutation interval: no entity can cross its
            # boundaries, so adjacent unrelated confidence values cannot help.
            end = positions[identities_b[i]]
            cursor = i
            while cursor <= end:
                end = max(end, positions[identities_b[cursor]])
                cursor += 1
            av = [x["confidence"] for x in a[i:end + 1]]
            bv = [x["confidence"] for x in b[i:end + 1]]
            ranges = max(av) - min(av), max(bv) - min(bv)
            if any(width > tie_tolerance for width in ranges):
                result.errors.append(
                    f"{_display_path(path)}[{i}:{end + 1}]: reordered confidence ranges "
                    f"{ranges[0]:.9g}/{ranges[1]:.9g} exceed tie tolerance {tie_tolerance:.9g}")
                # Retain the concrete identity/order mismatches as evidence.
                for index in range(i, end + 1):
                    compare(a[index], b[index], path + (index,))
            else:
                right_by_identity = {_span_identity(x): x for x in b[i:end + 1]}
                before = len(result.errors)
                for index in range(i, end + 1):
                    compare(a[index], right_by_identity[identities_a[index]], path + (index,))
                if len(result.errors) == before:
                    result.reorders.append({
                        "path": _display_path(path), "start_index": i,
                        "end_index_exclusive": end + 1,
                        "left_order": [list(x) for x in identities_a[i:end + 1]],
                        "right_order": [list(x) for x in identities_b[i:end + 1]],
                        "left_confidence_range": ranges[0], "right_confidence_range": ranges[1],
                        "max_confidence_delta": max(
                            abs(a[index]["confidence"] - right_by_identity[identities_a[index]]["confidence"])
                            for index in range(i, end + 1)),
                    })
            i = end + 1
        return True

    compare(left, right, ())
    return result


def compare_cases(left, right, tolerance=1e-4, allow_near_ties=False, tie_tolerance=1e-6):
    summary = {"strict_cases": 0, "total": max(len(left), len(right)), "passed": 0,
               "tolerance": tolerance, "allow_near_ties": allow_near_ties,
               "tie_tolerance": tie_tolerance, "near_tie_reorders": [],
               "max_confidence_delta": 0.0, "errors": []}
    for case, (a, b) in enumerate(zip(left, right)):
        strict = compare_values(a, b, tolerance=tolerance)
        summary["strict_cases"] += not strict.errors
        result = (compare_values(a, b, tolerance, True, tie_tolerance)
                  if allow_near_ties and strict.errors else strict)
        summary["passed"] += not result.errors
        if summary["max_confidence_delta"] is None or result.max_confidence_delta is None:
            summary["max_confidence_delta"] = None
        else:
            summary["max_confidence_delta"] = max(summary["max_confidence_delta"], result.max_confidence_delta)
        summary["near_tie_reorders"].extend({"case": case, **item} for item in result.reorders)
        if result.errors:
            summary["errors"].append({"case": case, "differences": result.errors})
    if len(left) != len(right):
        summary["errors"].append({"path": "$", "difference": f"row count {len(left)} != {len(right)}"})
    return summary


def read_jsonl(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")
    values = []
    with open(path, encoding="utf-8") as source:
        for line, text in enumerate(source, start=1):
            if not text.strip():
                continue
            try:
                value = json.loads(text, parse_constant=reject_constant)
            except ValueError as error:
                raise ValueError(f"{path}:{line}: {error}") from error
            if type(value) is not dict:
                raise ValueError(f"{path}:{line}: each result must be a JSON object")
            values.append(value)
    if not values:
        raise ValueError(f"{path}: no JSON result cases")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", help="Rust/reference JSONL")
    parser.add_argument("right", help="Apple/candidate JSONL")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--allow-near-ties", action="store_true")
    parser.add_argument("--tie-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    for name in ("tolerance", "tie_tolerance"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"{name.replace('_', '-')} must be a finite nonnegative number")
    try:
        summary = compare_cases(read_jsonl(args.left), read_jsonl(args.right), args.tolerance,
                                args.allow_near_ties, args.tie_tolerance)
    except (OSError, ValueError) as error:
        print(json.dumps({"strict_cases": 0, "total": 0, "passed": 0,
                          "near_tie_reorders": [], "errors": [str(error)]}, allow_nan=False))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
