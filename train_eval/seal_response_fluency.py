#!/usr/bin/env python3
"""Validate and seal the role-correct response fluency artifact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_eval.build_response_fluency import BOUNDARY_POLICY, ROLE, SCHEMA


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def seal(artifact: Path, pairs: Path, output: Path) -> dict:
    artifact = artifact.resolve()
    pairs = pairs.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"response-fluency receipt already exists: {output}")
    with artifact.open("rb") as handle:
        store = pickle.load(handle)
    if store.get("schema") != SCHEMA or store.get("role") != ROLE or \
            store.get("boundary_policy") != BOUNDARY_POLICY:
        raise RuntimeError("response-fluency provenance boundary mismatch")
    if store.get("source_pairs_sha256") != sha256(pairs):
        raise RuntimeError("response-fluency pairs hash mismatch")
    if store.get("response_count", 0) <= 0 or store.get("token_count", 0) <= 0 or \
            not store.get("uni") or store.get("maxl") != 4:
        raise RuntimeError("response-fluency content gate failed")
    key_counts = [len(store["stores"][order]) for order in range(1, 5)]
    if any(count <= 0 for count in key_counts):
        raise RuntimeError("response-fluency n-gram level is empty")
    receipt = {
        "schema": "unison-response-fluency-receipt/v1",
        "status": "sealed",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": {
            "path": receipt_path(artifact),
            "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact),
        },
        "source_pairs": {
            "path": receipt_path(pairs),
            "bytes": pairs.stat().st_size,
            "sha256": sha256(pairs),
        },
        "role": ROLE,
        "boundary_policy": BOUNDARY_POLICY,
        "response_count": store["response_count"],
        "token_count": store["token_count"],
        "vocabulary": len(store["uni"]),
        "max_order": store["maxl"],
        "context_key_counts": key_counts,
        "source_sha256": {
            "train_eval/build_response_fluency.py": sha256(
                ROOT / "train_eval/build_response_fluency.py"),
            "train_eval/build_kin_context.py": sha256(
                ROOT / "train_eval/build_kin_context.py"),
            "train_eval/seal_response_fluency.py": sha256(
                ROOT / "train_eval/seal_response_fluency.py"),
        },
        "claim_boundary": (
            "exact response-only fluency counts with hard response boundaries; "
            "artifact is not yet an active generation-quality result"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    receipt = seal(args.artifact, args.pairs, args.output)
    print(json.dumps({
        "status": receipt["status"],
        "response_count": receipt["response_count"],
        "token_count": receipt["token_count"],
        "vocabulary": receipt["vocabulary"],
        "context_key_counts": receipt["context_key_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
