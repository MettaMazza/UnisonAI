#!/usr/bin/env python3
"""Seal the exact packed native-transformer serving representation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.packed_rows import PACKED_SCHEMA, verify_packed_files


PACKED = ROOT / "omni/native_transformer_v4_packed"
ARTIFACT_RECEIPT = ROOT / "train_eval/native_transformer_v4_receipt.json"
OUTPUT = ROOT / "train_eval/native_transformer_v4_packed_receipt.json"


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
    artifact_receipt = json.loads(ARTIFACT_RECEIPT.read_text())
    manifest_path = PACKED / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != PACKED_SCHEMA:
        raise RuntimeError("packed manifest schema mismatch")
    if manifest.get("artifact_sha256") != artifact_receipt["artifact"]["sha256"]:
        raise RuntimeError("packed store is not bound to the sealed v4 artifact")
    verification = verify_packed_files(PACKED)
    if verification["status"] != "verified":
        raise RuntimeError(f"packed file verification failed: {verification}")
    packed_bytes = sum(
        path.stat().st_size for path in PACKED.iterdir() if path.is_file())
    receipt = {
        "schema": "unison-packed-native-transformer-receipt/v1",
        "status": "sealed",
        "packed_schema": PACKED_SCHEMA,
        "artifact_sha256": manifest["artifact_sha256"],
        "artifact_receipt_sha256": sha256(ARTIFACT_RECEIPT),
        "manifest_sha256": sha256(manifest_path),
        "packed_bytes": packed_bytes,
        "tables": manifest["tables"],
        "verification": verification,
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "omni/packed_rows.py": sha256(ROOT / "omni/packed_rows.py"),
            "train_eval/pack_native_transformer.py": sha256(
                ROOT / "train_eval/pack_native_transformer.py"),
            "train_eval/seal_packed_native_transformer.py": sha256(
                ROOT / "train_eval/seal_packed_native_transformer.py"),
        },
    }
    stage = OUTPUT.with_name(OUTPUT.name + ".building")
    stage.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    stage.replace(OUTPUT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
