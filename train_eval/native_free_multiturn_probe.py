#!/usr/bin/env python3
"""Free multi-turn transcript probe beyond direct induction-copy questions.

No target answer, lexical pass condition, or model judge is used.  The receipt
preserves the complete native served surfaces and stage traces so Maria Smith
can direct the next architectural investigation from applied behaviour.
"""
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


CONVERSATIONS = (
    ("gardening", "I am planning a tomato garden this weekend.",
     "What could help with that project?"),
    ("presentation", "I feel nervous about tomorrow's presentation.",
     "Could we talk through it?"),
    ("chess", "I enjoy playing chess with my sister.",
     "Why might that feel rewarding?"),
    ("travel", "I am taking the train to London.",
     "What should I remember before I leave?"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def execute() -> list[dict]:
    client = object.__new__(discord_bot.SFTDiscordClient)
    rows = []
    for name, statement, followup in CONVERSATIONS:
        session = SimpleNamespace(history_log=[])
        first_trace = []
        first = await client._generate_turn_surface(
            statement, [], session, "free-development", object(),
            trace=first_trace)
        session.history_log.extend((
            ("user", statement), ("assistant", first)))
        second_trace = []
        second = await client._generate_turn_surface(
            followup, [], session, "free-development", object(),
            trace=second_trace)
        rows.append({
            "conversation": name,
            "turns": [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": first},
                {"role": "user", "content": followup},
                {"role": "assistant", "content": second},
            ],
            "trace": {"first": first_trace, "second": second_trace},
            "first_nonempty": bool(first.strip()),
            "second_nonempty": bool(second.strip()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"free transcript output exists: {output}")
    rows = asyncio.run(execute())
    result = {
        "schema": "unison-native-free-multiturn-probe/v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "origin": "Codex-authored auxiliary transcript probe",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "No target answer or pass condition; Maria Smith assigns conclusions."
        },
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/discord_bot.py": sha256(ROOT / "omni/discord_bot.py"),
            "train_eval/native_free_multiturn_probe.py": sha256(Path(__file__)),
        },
        "conversations": len(rows),
        "first_nonempty": sum(row["first_nonempty"] for row in rows),
        "second_nonempty": sum(row["second_nonempty"] for row in rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps({
        "schema": result["schema"], "status": result["status"],
        "conversations": result["conversations"],
        "first_nonempty": result["first_nonempty"],
        "second_nonempty": result["second_nonempty"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
