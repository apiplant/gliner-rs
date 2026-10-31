#!/usr/bin/env python3
"""Compare every encoder state with a Rust F32 reference; reject non-finite values."""
import argparse
import json
import math
import sys


def compare(left_path, right_path, tolerance):
    maximum = squared = 0.0
    count = cases = 0
    with open(left_path, encoding="utf-8") as left, open(right_path, encoding="utf-8") as right:
        from itertools import zip_longest
        for case, (a_line, b_line) in enumerate(zip_longest(left, right)):
            if a_line is None or b_line is None:
                raise ValueError("Different number of encoder cases")
            a, b = json.loads(a_line), json.loads(b_line)
            for key in ("input_ids", "word_positions", "marker_positions", "encoder_shape"):
                if a[key] != b[key]:
                    raise ValueError(f"case {case}: {key} differs")
            av, bv = a["encoder_values"], b["encoder_values"]
            if len(av) != len(bv) or len(av) != math.prod(a["encoder_shape"]):
                raise ValueError(f"case {case}: invalid encoder value count")
            for index, (x, y) in enumerate(zip(av, bv)):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"case {case}, value {index}: non-finite encoder state")
                delta = abs(x - y)
                maximum = max(maximum, delta)
                squared += delta * delta
                count += 1
            cases += 1
    if not count:
        raise ValueError("No encoder states compared")
    result = {"cases": cases, "values": count, "max_abs_error": maximum,
              "rms_error": math.sqrt(squared / count), "tolerance": tolerance,
              "passed": maximum <= tolerance}
    print(json.dumps(result, sort_keys=True))
    return result["passed"]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left"); p.add_argument("right")
    p.add_argument("--tolerance", type=float, default=1e-3)
    args = p.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        p.error("tolerance must be a finite nonnegative number")
    try:
        sys.exit(0 if compare(args.left, args.right, args.tolerance) else 1)
    except (ValueError, KeyError, OSError) as error:
        print(f"Trace comparison failed: {error}", file=sys.stderr)
        sys.exit(1)
