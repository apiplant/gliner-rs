"""Run parity cases through the Python reference (gliner2) and print JSON lines.

Usage: python scripts/parity.py MODEL_DIR scripts/parity_cases.json > python.jsonl
Compare with: cargo run --release --example parity -- MODEL_DIR scripts/parity_cases.json
"""
import json
import sys

import torch
from gliner2 import AutoExtractor
from gliner2.processing.word_splitter import CharLevelSplitter, WhitespaceTokenSplitter

LONG_TEXT = " ".join(
    f"In report {i}, analyst Jane Smith of Goldman Sachs said Tesla shipped the Model Y from Berlin while Elon Musk visited Austin."
    for i in range(40)
)


def main():
    model_dir, cases_path = sys.argv[1], sys.argv[2]
    model = AutoExtractor.from_pretrained(model_dir, map_location="cpu")
    model.eval()
    for case in json.load(open(cases_path)):
        text = LONG_TEXT if case.get("long") else case["text"]
        model.processor.word_splitter = CharLevelSplitter() if case.get("char_split") else WhitespaceTokenSplitter()
        if case.get("json") and not case.get("legacy"):
            schema = model._json_schema(case["json"])
        else:
            schema = model.create_schema()
            for name, fields in (case.get("json") or {}).items():
                builder = schema.structure(name)
                for spec in fields:
                    fname, dtype, choices, desc = model._parse_field_spec(spec)
                    builder.field(fname, dtype=dtype, choices=choices, description=desc)
        if case.get("entities"):
            schema.entities(case["entities"])
        if case.get("relations"):
            schema.relations(case["relations"])
        for cls in case.get("classifications", []):
            cfg = dict(cls)
            schema.classification(cfg.pop("task"), cfg.pop("labels"), **cfg)
        with torch.no_grad():
            out = model.extract(text, schema, threshold=case.get("threshold", 0.5),
                                include_confidence=True, include_spans=True)
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
