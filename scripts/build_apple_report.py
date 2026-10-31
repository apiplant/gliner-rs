#!/usr/bin/env python3
"""Build the self-contained English technical report from recorded benchmark data."""
import argparse
import json
from pathlib import Path


def build(root, destination):
    comparison = json.loads((root / "docs/benchmarks/rust-vs-mlx.json").read_text())
    profiles = json.loads((root / "docs/benchmarks/mac-m3-pro-multi.json").read_text())
    implementation = json.loads((root / "docs/benchmarks/apple-port-configurations.json").read_text())
    data = {
        "comparison": comparison,
        "profiles": profiles,
        "implementation": implementation,
        "validation": {
            "swiftTests": 32, "pythonTests": 36, "inputCasesPerTokenizer": 92,
            "coreCases": 21, "advancedCases": 7,
            "phoneSmall": {"cases": 28, "strictCases": 28, "encoderValues": 1024128,
                           "maxEncoderError": 0.000072479248046875,
                           "loadSeconds": 0.539281083, "extractionSumSeconds": 2.052727505,
                           "peakResidentBytes": 592855040},
            "phoneMulti": {"cases": 28, "strictCases": 27, "encoderValues": 2592768,
                           "maxEncoderError": 0.00011027045547962189,
                           "maxConfidenceError": 0.000013530254364013672,
                           "loadSeconds": 1.331397791, "extractionSumSeconds": 2.715662457,
                           "peakResidentBytes": 1492697088, "peakMLXBytes": 2673877508},
            "confidenceTolerance": 0.0001, "encoderTolerance": 0.001,
            "nearTieDifference": 0.00000011920928955078125,
        },
        "macDemoSmokeTest": {
            "checkpoint": "multi", "precision": "float32", "cacheMiB": "auto",
            "clearCacheAfterRequest": False, "tf32": False,
            "firstCallIncludingLoadMs": 1251.326,
            "subsequentCallsMs": [57.608, 47.763, 54.531, 41.933],
            "note": "Manually spaced worker timings; excludes rendering/actor queue delay; not a paired CLI benchmark.",
        },
        "example": {
            "text": "Michele vive a Milano e lavora per Acme.",
            "entities": [
                {"text": "Michele", "label": "person", "confidence": 0.9974225759506226},
                {"text": "Milano", "label": "city", "confidence": 0.9989032745361328},
                {"text": "Acme", "label": "organization", "confidence": 0.9804876446723938},
            ],
        },
    }
    template = (root / "scripts/templates/apple_report.html").read_text()
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    document = template.replace("__REPORT_DATA__", encoded)
    if "__REPORT_DATA__" in document:
        raise ValueError("Report data was not inserted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    print(f"Report: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    build(repo, args.output or repo / "reports/gliner-apple-report.html")
