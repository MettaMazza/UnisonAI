#!/usr/bin/env python3
"""Codex implementation timing for the exact packed native route.

This is not a Maria benchmark or project conclusion. It separates cold packed
store activation from native generation and preserves every returned surface.
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = CountedCausalTransformer()
    started = time.monotonic()
    identity = model.identity()
    load_seconds = time.monotonic() - started
    rows = []
    for prompt in PROMPTS:
        began = time.monotonic()
        surface = model.generate(prompt)
        rows.append({"prompt": prompt, "surface": surface,
                     "seconds": round(time.monotonic() - began, 6)})
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    maximum_bytes = maximum if sys.platform == "darwin" else maximum * 1024
    result = {
        "schema": "unison-native-transformer-latency/v4-packed-integer",
        "status": "completed",
        "provenance": {
            "origin": "Codex-authored implementation timing",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Not Maria's benchmark, finding, parity definition, or run gate.",
        },
        "identity": identity,
        "source_sha256": sha256(ROOT / "omni/native_transformer.py"),
        "load_seconds": round(load_seconds, 6),
        "generation_seconds": round(sum(row["seconds"] for row in rows), 6),
        "maximum_resident_bytes": maximum_bytes,
        "rows": rows,
    }
    if identity.get("prompt_context"):
        suffix = "contextual"
    else:
        suffix = "bounded" if identity.get("value_cache") else "baseline"
    output = ROOT / f"train_eval/native_transformer_latency_v4_packed_{suffix}_20260719.json"
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
