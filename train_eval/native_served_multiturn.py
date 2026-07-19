#!/usr/bin/env python3
"""Source-bound multi-turn execution through Unison's served generation method."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni import discord_bot


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


async def execute() -> list[dict]:
    client = object.__new__(discord_bot.SFTDiscordClient)
    rows = []
    for name, statement, question, expected_any in SCENARIOS:
        session = SimpleNamespace(history_log=[])
        first_trace = []
        first = await client._generate_turn_surface(
            statement, [], session, "served-development", object(),
            trace=first_trace)
        session.history_log.extend((
            ("user", statement), ("assistant", first)))
        second_trace = []
        second = await client._generate_turn_surface(
            question, [], session, "served-development", object(),
            trace=second_trace)
        rows.append({
            "scenario": name,
            "turns": [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": first},
                {"role": "user", "content": question},
                {"role": "assistant", "content": second},
            ],
            "trace": {"first": first_trace, "second": second_trace},
            "auxiliary_expected_any": list(expected_any),
            "auxiliary_continuity_observed": any(
                token in second.lower() for token in expected_any),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"served-path output exists: {output}")
    rows = asyncio.run(execute())
    result = {
        "schema": "unison-native-served-multiturn/v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "Source-bound development evidence; Maria Smith assigns conclusions.",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/discord_bot.py": sha256(ROOT / "omni/discord_bot.py"),
            "train_eval/native_served_multiturn.py": sha256(Path(__file__)),
        },
        "scenarios": len(rows),
        "auxiliary_continuity_observed": sum(
            row["auxiliary_continuity_observed"] for row in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "scenarios": result["scenarios"],
        "auxiliary_continuity_observed": result[
            "auxiliary_continuity_observed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
