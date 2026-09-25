#!/usr/bin/env python3
"""Run every question in jev-questions through gliner-classify (decide-1b, CUDA).

v3: same one-call-per-section batching as v1 (`run_jev_questions_decide.py`,
which beat v2's one-call-per-question split 10/16 vs 9/16), but with a plain
human-readable rendering of `state` instead of a single escaped JSON blob —
`resume`'s state is literally `{"resume": "<raw resume text>"}`, so
JSON-dumping it mangles a plaintext resume with escaped newlines/quotes for
no reason.
"""
import json
import os
import subprocess
import sys

QUESTIONS_PATH = "/mnt/extra/projects/jev-questions"
OUT_PATH = "/mnt/extra/projects/decide-1b-answers-v3"
BIN = "/mnt/extra/projects/gliner-rs/target/release/gliner-classify"
MODEL_DIR = "/mnt/extra/ai/gliner/GLiNER2.5-Decide-1B"


def sanitize(text):
    return text.replace(",", ";").replace("\n", " ").strip()


def instructions_text(instr):
    return instr if isinstance(instr, str) else json.dumps(instr, ensure_ascii=False)


def render_state(state):
    """A single string-valued key (the `resume` section) is the raw text
    itself; otherwise render as `key: value` lines, JSON-dumping only the
    nested dict/list values (still far more readable than one top-level blob)."""
    if isinstance(state, dict) and len(state) == 1:
        (only_value,) = state.values()
        if isinstance(only_value, str):
            return only_value
    lines = []
    for k, v in state.items():
        if isinstance(v, (dict, list)):
            lines.append(f"{k}:\n{json.dumps(v, indent=2, ensure_ascii=False)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def build_task(qid, q):
    instr = sanitize(instructions_text(q["instructions"]))
    t = q["type"]
    crit = q.get("criteria")

    if t == "score":
        parts = [f"{i}:{sanitize(instr + ' | ' + level)}" for i, level in enumerate(crit)]
    elif t == "choice":
        parts = []
        for key, desc in crit.items():
            body = f"{instr} | {desc}" if desc else instr
            parts.append(f"{key}:{sanitize(body)}")
    else:  # noul (true/false)
        crit = crit or {}
        false_desc = crit.get("false") or "no, the statement does not hold"
        true_desc = crit.get("true") or "yes, the statement holds"
        parts = [f"false:{sanitize(instr + ' | ' + false_desc)}", f"true:{sanitize(instr + ' | ' + true_desc)}"]

    return f"{qid}={','.join(parts)}"


def to_answer(qid, q, result):
    t = q["type"]
    probs = result["probabilities"]
    if t == "score":
        score = sum(int(k) * p for k, p in probs.items())
        _, top_p = max(probs.items(), key=lambda kv: kv[1])
        return {"type": "score", "score": score, "confidence": top_p, "probabilities": probs}
    if t == "choice":
        return {"type": "choice", "choice": result["label"], "confidence": result["confidence"], "probabilities": probs}
    return {"type": "noul", "noul": probs.get("true", 0.0)}


def main():
    data = json.load(open(QUESTIONS_PATH))
    answers = {}
    env = {**os.environ, "GLINER_MODEL": MODEL_DIR}
    for section, body in data.items():
        state_text = render_state(body["state"])
        questions = body["questions"]
        tasks = [build_task(qid, q) for qid, q in questions.items()]

        cmd = [BIN, "--cuda", "--format", "json", "--all"]
        for t in tasks:
            cmd += ["--task", t]
        cmd.append(state_text)

        print(f"--- {section} ({len(tasks)} questions) ---", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        if proc.returncode != 0:
            print(proc.stdout, file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            answers[section] = {"error": proc.stderr[-2000:]}
            continue

        rows = json.loads(proc.stdout)
        result_row = rows[0]
        answers[section] = {}
        for qid, q in questions.items():
            answers[section][qid] = to_answer(qid, q, result_row[qid])
        print(json.dumps(answers[section], indent=2), file=sys.stderr)

    out = {"model": "GLiNER2.5-Decide-1B", "answers": answers}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
