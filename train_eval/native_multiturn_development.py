#!/usr/bin/env python3
"""Source-bound applied multi-turn development run for Unison's native route.

The checks are explicit auxiliary continuity questions, not Maria's benchmark
or conclusion. Their purpose is to expose favourable and unfavourable applied
behaviour after architecture changes; implementation tests alone cannot do so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer


DEFAULT_OUTPUT = ROOT / "train_eval/native_multiturn_development_20260719.json"
SCENARIOS = (
    ("name", "My name is Maria.", "What is my name?", ("maria",)),
    ("preference", "I prefer tea to coffee.",
     "Which drink did I say I prefer?", ("tea",)),
    ("topic", "I am growing tomatoes in my garden.",
     "What am I growing?", ("tomato", "tomatoes")),
    ("feeling", "I feel nervous about speaking tomorrow.",
     "How did I say I feel?", ("nervous",)),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"development output already exists: {output}")
    if args.store:
        store = args.store.resolve()
        model = CountedCausalTransformer(
            store_path=str(store), packed_path=str(store) + ".packed-absent")
    else:
        model = CountedCausalTransformer()
    rows = []
    for scenario, statement, question, expected_any in SCENARIOS:
        first = model.generate(statement)
        history = [("user", statement), ("assistant", first)]
        second = model.generate(question, history=history)
        lowered = second.lower()
        rows.append({
            "scenario": scenario,
            "turns": [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": first},
                {"role": "user", "content": question},
                {"role": "assistant", "content": second},
            ],
            "auxiliary_expected_any": list(expected_any),
            "auxiliary_continuity_observed": any(
                token in lowered for token in expected_any),
        })
    result = {
        "schema": "unison-native-multiturn-development/v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": (
            "Applied development evidence only; expectations are explicit "
            "Codex-authored probes and Maria assigns any conclusion."
        ),
        "source_commit": git_commit(),
        "identity": model.identity(),
        "development_store": (
            {"path": str(args.store.resolve()), "sha256": sha256(args.store.resolve())}
            if args.store else None
        ),
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/prompt_context.py": sha256(ROOT / "omni/prompt_context.py"),
            "train_eval/native_multiturn_development.py": sha256(Path(__file__)),
            "train_eval/native_transformer_v4_packed_receipt.json": sha256(
                ROOT / "train_eval/native_transformer_v4_packed_receipt.json"),
        },
        "scenarios": len(rows),
        "auxiliary_continuity_observed": sum(
            row["auxiliary_continuity_observed"] for row in rows),
        "rows": rows,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "scenarios": result["scenarios"],
        "auxiliary_continuity_observed": result[
            "auxiliary_continuity_observed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
