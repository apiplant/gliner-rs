"""Compare two JSON-lines result files structurally, allowing float tolerance."""
import json
import sys

TOL = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3


def diff(a, b, path="$"):
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or abs(a - b) > TOL:
            yield f"{path}: {a!r} != {b!r}"
        return
    if type(a) is not type(b):
        yield f"{path}: type {type(a).__name__} != {type(b).__name__} ({a!r} vs {b!r})"
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                yield f"{path}.{k}: missing on {'left' if k not in a else 'right'}"
            else:
                yield from diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            yield f"{path}: len {len(a)} != {len(b)}\n  left={a}\n  right={b}"
        for i, (x, y) in enumerate(zip(a, b)):
            yield from diff(x, y, f"{path}[{i}]")
    elif a != b:
        yield f"{path}: {a!r} != {b!r}"


left = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
right = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
bad = 0
max_delta = 0.0
for i, (a, b) in enumerate(zip(left, right)):
    problems = list(diff(a, b))
    if problems:
        bad += 1
        print(f"case {i}:"); [print("  " + p) for p in problems]
print(f"{len(left) - bad}/{len(left)} cases match (tol={TOL})")
sys.exit(1 if bad or len(left) != len(right) else 0)
