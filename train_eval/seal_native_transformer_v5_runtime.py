#!/usr/bin/env python3
"""Seal the v5 position-owned runtime over the unchanged counted artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "train_eval/native_transformer_v5_runtime_receipt.json"
SOURCES = (
    "omni/native_transformer.py",
    "omni/position_relation.py",
    "omni/prompt_context.py",
    "omni/packed_rows.py",
    "train_eval/seal_native_transformer_v5_runtime.py",
)
BOUND_RECEIPTS = (
    "train_eval/native_transformer_v4_packed_receipt.json",
    "train_eval/native_position_relation_v5_packed_receipt.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    packed = json.loads((ROOT / BOUND_RECEIPTS[0]).read_text())
    position = json.loads((ROOT / BOUND_RECEIPTS[1]).read_text())
    if (packed.get("status") != "sealed"
            or packed.get("schema")
            != "unison-packed-native-transformer-receipt/v1"):
        raise RuntimeError("v4 packed receipt is not sealed")
    if (position.get("status") != "sealed"
            or position.get("schema")
            != "unison-packed-position-relation/v1/receipt"):
        raise RuntimeError("v5 position receipt is not sealed")
    receipt = {
        "schema": "unison-native-transformer-v5-runtime/v1",
        "status": "sealed",
        "provenance": {
            "origin": "Codex implementation seal",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": (
                "Implementation artifact; Maria owns conclusions and run gates."),
        },
        "architecture": {
            "position_value": "exact-relative-position-marginal/v5",
            "position_semantic2": "exact-position-prefix-marginal/v5",
            "position_semantic3": "exact-position-causal-prefix-marginal/v5",
            "missing_position_address": "no-row-contribution",
            "learned_parameters": 0,
        },
        "sources": {path: sha256(ROOT / path) for path in SOURCES},
        "bound_receipts": {
            path: sha256(ROOT / path) for path in BOUND_RECEIPTS},
    }
    stage = OUTPUT.with_name(OUTPUT.name + ".building")
    stage.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    stage.replace(OUTPUT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
