#!/usr/bin/env python3
"""Pack the complete canonical position relation into fixed exact records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.position_relation import RECORD, SCHEMA


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    stage = path.with_name(path.name + ".writing")
    stage.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(stage, path)


def pack_relation(relation_path: Path, relation_receipt_path: Path,
                  output_path: Path, output_receipt_path: Path) -> dict:
    relation_path = relation_path.resolve()
    relation_receipt_path = relation_receipt_path.resolve()
    output_path = output_path.resolve()
    output_receipt_path = output_receipt_path.resolve()
    relation_receipt = json.loads(relation_receipt_path.read_text())
    if (relation_receipt.get("schema")
            != "unison-position-conditioned-canonical-relation/v1"
            or relation_receipt.get("status") != "completed"):
        raise RuntimeError("canonical position relation receipt mismatch")
    if relation_path.stat().st_size != relation_receipt["relation_bytes"]:
        raise RuntimeError("canonical position relation byte length mismatch")

    stage = output_path.with_name(output_path.name + ".building")
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()
    previous_key = None
    rows = observations = 0
    buffer = bytearray()
    with relation_path.open("rb") as source, stage.open("wb") as target:
        for line in source:
            input_digest.update(line)
            fields = line.rstrip(b"\n").split(b"\t")
            if len(fields) != 6:
                raise RuntimeError(f"invalid canonical row at index {rows}")
            values = tuple(int(field) for field in fields)
            if any(value < 0 or value >= (1 << 32) for value in values):
                raise RuntimeError(f"canonical uint32 overflow at index {rows}")
            key, count = values[:5], values[5]
            if count <= 0 or (previous_key is not None and key <= previous_key):
                raise RuntimeError(f"canonical order/count violation at index {rows}")
            previous_key = key
            held = RECORD.pack(*values)
            buffer.extend(held)
            observations += count
            rows += 1
            if len(buffer) >= 8 * 1024 * 1024:
                target.write(buffer)
                output_digest.update(buffer)
                buffer.clear()
            if rows % 10_000_000 == 0:
                print(json.dumps({"phase": "pack", "rows": rows,
                                  "observations": observations}), flush=True)
        if buffer:
            target.write(buffer)
            output_digest.update(buffer)
        target.flush()
        os.fsync(target.fileno())

    if input_digest.hexdigest() != relation_receipt["relation_sha256"]:
        raise RuntimeError("canonical position relation SHA-256 mismatch")
    if rows != relation_receipt["unique_canonical_entries"]:
        raise RuntimeError("canonical position relation row count mismatch")
    if observations != relation_receipt["observations"]:
        raise RuntimeError("canonical position relation observation count mismatch")
    expected_bytes = rows * RECORD.size
    if stage.stat().st_size != expected_bytes:
        raise RuntimeError("packed position relation byte count mismatch")
    os.replace(stage, output_path)

    result = {
        "schema": SCHEMA + "/receipt",
        "status": "sealed",
        "canonical_receipt_sha256": sha256(relation_receipt_path),
        "canonical_relation_sha256": relation_receipt["relation_sha256"],
        "canonical_relation_bytes": relation_receipt["relation_bytes"],
        "unique_canonical_entries": rows,
        "observations": observations,
        "record_format": "little-endian 6 x uint32",
        "record_bytes": RECORD.size,
        "packed_bytes": expected_bytes,
        "packed_sha256": output_digest.hexdigest(),
        "sources": {
            "omni/position_relation.py": sha256(ROOT / "omni/position_relation.py"),
            "train_eval/pack_position_relation.py": sha256(Path(__file__)),
        },
        "provenance": {
            "origin": "Codex exact serving representation",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Implementation artifact; Maria owns conclusions and benchmark timing.",
        },
    }
    atomic_json(output_receipt_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation", type=Path,
                        default=ROOT / "train_eval/native_position_relation_v5/relation.tsv")
    parser.add_argument("--relation-receipt", type=Path,
                        default=ROOT / "train_eval/native_position_relation_v5_receipt_20260719.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "omni/native_position_relation_v5.bin")
    parser.add_argument("--output-receipt", type=Path,
                        default=ROOT / "train_eval/native_position_relation_v5_packed_receipt.json")
    args = parser.parse_args()
    print(json.dumps(pack_relation(
        args.relation, args.relation_receipt, args.output,
        args.output_receipt), sort_keys=True))


if __name__ == "__main__":
    main()
