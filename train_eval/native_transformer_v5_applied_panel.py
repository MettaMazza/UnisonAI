#!/usr/bin/env python3
"""Applied prompt panel for the sealed position-owned v5 native runtime.

This records implementation data only. It does not declare Maria's benchmark,
finding, parity, loss, victory, or promotion decision.
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    model = CountedCausalTransformer()
    rows = []
    started = time.monotonic()
    identity = model.identity()
    activation_seconds = time.monotonic() - started
    for prompt in PROMPTS:
        began = time.monotonic()
        surface = model.generate(prompt)
        rows.append({
            "prompt": prompt,
            "surface": surface,
            "seconds": round(time.monotonic() - began, 6),
            "position_runtime": model.last_position_runtime,
        })
    elapsed = time.monotonic() - started
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maximum_bytes = maximum if sys.platform == "darwin" else maximum * 1024
    prior = ROOT / "train_eval/native_transformer_latency_v4_packed_contextual_20260719.json"
    result = {
        "schema": "unison-native-transformer-v5-applied-panel/v1",
        "status": "completed",
        "provenance": {
            "origin": "Codex implementation development run",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": (
                "Not Maria's benchmark, finding, parity definition, loss, "
                "victory, promotion, or run gate."),
        },
        "identity": identity,
        "activation_seconds": round(activation_seconds, 6),
        "generation_seconds": round(sum(row["seconds"] for row in rows), 6),
        "elapsed_seconds": round(elapsed, 6),
        "maximum_resident_bytes": maximum_bytes,
        "rows": rows,
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/position_relation.py": sha256(
                ROOT / "omni/position_relation.py"),
            "train_eval/native_transformer_v5_runtime_receipt.json": sha256(
                ROOT / "train_eval/native_transformer_v5_runtime_receipt.json"),
            "prior_v4_applied_receipt": sha256(prior),
        },
    }
    output = ROOT / "train_eval/native_transformer_v5_applied_panel_20260719.json"
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    model.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
