#!/usr/bin/env python3
"""Applied transfer panel for Unison's role-conditioned induction head.

The conversations are distinct from the four development examples used to
select the architectural change.  They measure whether the same exact native
transformer operation carries new entities, relations, multiple user turns,
and nearer-turn priority.  This is source-bound development evidence; Maria
Smith assigns conclusions and publication status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer


SCENARIOS = (
    ("destination", ("I am travelling to London by train.",),
     "Where am I travelling?", ("london",)),
    ("schedule", ("My appointment is at seven on Monday.",),
     "When is my appointment?", ("seven", "monday")),
    ("object_location", ("I put the keys beneath the blue lamp.",),
     "Where did I put the keys?", ("beneath", "lamp")),
    ("language", ("I am learning Spanish this winter.",),
     "Which language am I learning?", ("spanish",)),
    ("pet_name", ("My dog is called Pepper.",),
     "What is my dog called?", ("pepper",)),
    ("storage", ("The telescope is beside the red cabinet.",),
     "Where is the telescope?", ("beside", "cabinet")),
    ("nearest_preference",
     ("I prefer apples to pears.", "I prefer tea to coffee."),
     "Which drink did I say I prefer?", ("tea",)),
    ("nearest_destination",
     ("I am travelling to York by train.", "I am travelling to London by train."),
     "Where am I travelling?", ("london",)),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"transfer output already exists: {output}")

    model = CountedCausalTransformer()
    rows = []
    for name, statements, question, expected_any in SCENARIOS:
        history = []
        turns = []
        for statement in statements:
            surface = model.generate(statement, history=history)
            history.extend((("user", statement), ("assistant", surface)))
            turns.extend((
                {"role": "user", "content": statement},
                {"role": "assistant", "content": surface},
            ))
        answer = model.generate(question, history=history)
        turns.extend((
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ))
        lowered = answer.lower()
        rows.append({
            "scenario": name,
            "turns": turns,
            "auxiliary_expected_any": list(expected_any),
            "auxiliary_transfer_observed": any(
                token in lowered for token in expected_any),
        })

    result = {
        "schema": "unison-native-induction-transfer/v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": (
            "Source-bound development evidence; expectations are explicit "
            "Codex-authored probes and Maria Smith assigns conclusions."
        ),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "identity": model.identity(),
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "train_eval/native_induction_transfer.py": sha256(Path(__file__)),
            "train_eval/native_transformer_v4_packed_receipt.json": sha256(
                ROOT / "train_eval/native_transformer_v4_packed_receipt.json"),
        },
        "scenarios": len(rows),
        "auxiliary_transfer_observed": sum(
            row["auxiliary_transfer_observed"] for row in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "scenarios": result["scenarios"],
        "auxiliary_transfer_observed": result["auxiliary_transfer_observed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
