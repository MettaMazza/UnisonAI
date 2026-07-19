#!/usr/bin/env python3
"""Codex implementation probe for the stacked prompt-context route.

This is auxiliary implementation evidence, not Maria's benchmark or real-run
gate. It measures the exact contextual-key construction separately from greedy
decode and preserves every returned surface.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer
from train_eval.native_transformer_smoke import PROMPTS


OUTPUT = ROOT / "train_eval/native_prompt_context_probe_v1_20260719.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = CountedCausalTransformer()
    identity = model.identity()
    rows = []
    for prompt in PROMPTS:
        began = time.monotonic()
        addresses = model._position_addresses(prompt)
        keys = model._contextual_keys(prompt)
        context_seconds = time.monotonic() - began
        began = time.monotonic()
        tokens = model._generate_tokens_from_keys(keys, 32)
        generation_seconds = time.monotonic() - began
        rows.append({
            "prompt": prompt,
            "prompt_positions": len(addresses),
            "contextual_key_support": len(keys),
            "contextual_key_closure": str(sum(keys.values())),
            "context_seconds": round(context_seconds, 6),
            "generation_seconds": round(generation_seconds, 6),
            "surface": model._surface(tokens),
        })
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_bytes = maximum if sys.platform == "darwin" else maximum * 1024
    result = {
        "schema": "unison-native-prompt-context-probe/v1",
        "status": "completed",
        "provenance": {
            "origin": "Codex-authored auxiliary implementation probe",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": (
                "Not Maria's benchmark, finding, loss, parity definition, "
                "or real-run gate."
            ),
        },
        "identity": identity,
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/prompt_context.py": sha256(ROOT / "omni/prompt_context.py"),
            "train_eval/native_prompt_context_probe.py": sha256(Path(__file__)),
        },
        "maximum_resident_bytes": maximum_bytes,
        "context_seconds": round(
            sum(row["context_seconds"] for row in rows), 6),
        "generation_seconds": round(
            sum(row["generation_seconds"] for row in rows), 6),
        "rows": rows,
    }
    stage = OUTPUT.with_name(OUTPUT.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(OUTPUT)
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "context_seconds": result["context_seconds"],
        "generation_seconds": result["generation_seconds"],
        "maximum_resident_bytes": result["maximum_resident_bytes"],
        "rows": len(rows),
        "all_contextual_keys_close_to_one": all(
            row["contextual_key_closure"] == "1" for row in rows),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
