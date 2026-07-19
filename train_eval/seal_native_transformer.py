#!/usr/bin/env python3
"""Seal the counted causal-transformer artifact to its corpus and source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "omni/native_transformer_v4.pkl"
PAIRS = ROOT / "omni/pairs.pkl"
RECEIPT = ROOT / "train_eval/native_transformer_v4_receipt.json"
sys.path.insert(0, str(ROOT))

from omni.native_transformer import ROLE_POLICY, SCHEMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with ARTIFACT.open("rb") as handle:
        record = pickle.load(handle)
    if (record.get("schema") != SCHEMA
            or record.get("role_policy") != ROLE_POLICY):
        raise RuntimeError("native transformer artifact schema or role policy mismatch")
    pair_hash = sha256(PAIRS)
    if record.get("source_pairs_sha256") != pair_hash:
        raise RuntimeError("native transformer is not bound to the current pair corpus")
    receipt = {
        "schema": SCHEMA.replace("/v4", "-receipt/v4"),
        "status": "sealed",
        "artifact": {
            "path": "omni/native_transformer_v4.pkl",
            "bytes": ARTIFACT.stat().st_size,
            "sha256": sha256(ARTIFACT),
        },
        "source_pairs": {
            "path": "omni/pairs.pkl",
            "bytes": PAIRS.stat().st_size,
            "sha256": pair_hash,
        },
        "training": {
            "responses": record["response_count"],
            "causal_targets": record["token_count"],
            "vocabulary": len(record["vocab"]),
            "qk_relations": len(record["qk"]),
            "value_vectors": len(record["values"]),
            "semantic_ffn_keys": len(record["semantic_ffn"]),
            "semantic_ffn3_keys": len(record["semantic_ffn3"]),
            "ffn2_keys": len(record["ffn2"]),
            "ffn3_keys": len(record["ffn3"]),
        },
        "sources": {
            "omni/native_transformer.py": sha256(ROOT / "omni/native_transformer.py"),
            "train_eval/build_native_transformer.py": sha256(
                ROOT / "train_eval/build_native_transformer.py"),
            "train_eval/seal_native_transformer.py": sha256(
                ROOT / "train_eval/seal_native_transformer.py"),
        },
    }
    stage = RECEIPT.with_name(RECEIPT.name + ".building")
    stage.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    stage.replace(RECEIPT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
