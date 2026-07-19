#!/usr/bin/env python3
"""Pack the sealed v4 transformer into an exact memory-mapped serving view."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import ROLE_POLICY, SCHEMA
from omni.packed_rows import pack_record, verify_packed_files


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path,
                        default=ROOT / "omni/native_transformer_v4.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "omni/native_transformer_v4_packed")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"packed output already exists: {output}")
    stage = output.with_name(output.name + f".building.{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"packed staging output already exists: {stage}")
    artifact_hash = sha256(artifact)
    with artifact.open("rb") as handle:
        record = pickle.load(handle)
    if (record.get("schema") != SCHEMA
            or record.get("role_policy") != ROLE_POLICY):
        raise RuntimeError("source artifact schema or role policy mismatch")
    try:
        manifest = pack_record(record, stage, artifact_hash, consume=True)
        verification = verify_packed_files(stage)
        if verification["status"] != "verified":
            raise RuntimeError(f"packed verification failed: {verification}")
        os.replace(stage, output)
    except BaseException:
        # Staging is deliberately retained for forensic inspection; no partial
        # directory can replace the serving path.
        raise
    print(json.dumps({"status": "packed", "output": str(output),
                      "artifact_sha256": artifact_hash,
                      "tables": len(manifest["tables"]),
                      "verification": verification}, sort_keys=True))


if __name__ == "__main__":
    main()
