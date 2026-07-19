#!/usr/bin/env python3
"""Matched panel for exact canonical-prefix response-cache identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer  # noqa: E402
from train_eval.native_transformer_smoke import PROMPTS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def query_totals(rows):
    totals = {name: 0 for name in ("value", "semantic2", "semantic3")}
    for row in rows:
        runtime = row.get("position_runtime") or {}
        for name in totals:
            totals[name] += runtime.get("by_namespace", {}).get(name, {}).get(
                "queries", 0)
    return totals


def main() -> None:
    output = ROOT / "train_eval/native_transformer_v5_prefix_cache_panel_20260719.json"
    if output.exists():
        raise FileExistsError(f"applied receipt exists: {output}")
    baseline_path = ROOT / \
        "train_eval/native_transformer_v5_optimized_applied_panel_20260719.json"
    baseline = json.loads(baseline_path.read_text())
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
    identity_rows = [
        row["prompt"] == prior["prompt"] and row["surface"] == prior["surface"]
        for row, prior in zip(rows, baseline["rows"])
    ]
    result = {
        "schema": "unison-native-transformer-v5-prefix-cache-panel/v1",
        "status": "completed",
        "result_type": "measured implementation result",
        "provenance": {
            "origin": "Codex implementation development run",
            "agent": "Codex", "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Maria Smith assigns conclusions and trial timing.",
        },
        "declared_purpose": (
            "Reuse each response-local exact canonical prefix once without "
            "changing any marginal, surface, sampling rule or cache lifetime."),
        "identity": identity,
        "activation_seconds": round(activation_seconds, 6),
        "generation_seconds": round(sum(row["seconds"] for row in rows), 6),
        "elapsed_seconds": round(elapsed, 6),
        "maximum_resident_bytes": maximum_bytes,
        "query_totals": query_totals(rows),
        "baseline": {
            "path": str(baseline_path.relative_to(ROOT)),
            "sha256": sha256(baseline_path),
            "generation_seconds": baseline["generation_seconds"],
            "maximum_resident_bytes": baseline["maximum_resident_bytes"],
            "query_totals": query_totals(baseline["rows"]),
        },
        "exact_surface_identity": {
            "rows": len(rows), "identical": sum(identity_rows),
            "disagreements": len(identity_rows) - sum(identity_rows),
        },
        "rows": rows,
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/position_relation.py": sha256(
                ROOT / "omni/position_relation.py"),
            "train_eval/native_transformer_v5_runtime_receipt.json": sha256(
                ROOT / "train_eval/native_transformer_v5_runtime_receipt.json"),
            "train_eval/native_transformer_v5_prefix_cache_panel.py":
                sha256(Path(__file__)),
        },
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    model.close()
    print(json.dumps({
        "status": "completed",
        "generation_seconds": result["generation_seconds"],
        "maximum_resident_bytes": maximum_bytes,
        "query_totals": result["query_totals"],
        "baseline_query_totals": result["baseline"]["query_totals"],
        "exact_surface_identity": result["exact_surface_identity"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
