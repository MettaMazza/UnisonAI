#!/usr/bin/env python3
"""Validate and seal the exact generated stores used by Unison quality runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    "train_eval/conv_corpus.txt",
    "omni/pairs.pkl",
    "omni/word_fluency.pkl",
    "omni/word_coupling.pkl",
    "omni/word_kin.pkl",
    "omni/kin_context.pkl",
]
BUILDERS = [
    "train_eval/download_conversational.py",
    "train_eval/build_pairs_from_datasets.py",
    "train_eval/build_fluency.py",
    "train_eval/build_coupling.py",
    "train_eval/build_kin_store.py",
    "train_eval/build_kin_context.py",
    "train_eval/seal_runtime_store_build.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty runtime-store artifact: {relative}")
    return {"sha256": sha256(path), "bytes": path.stat().st_size}


def validate_store_content() -> dict:
    with (ROOT / "omni/pairs.pkl").open("rb") as handle:
        pairs = pickle.load(handle)
    with (ROOT / "omni/word_fluency.pkl").open("rb") as handle:
        fluency = pickle.load(handle)
    with (ROOT / "omni/word_coupling.pkl").open("rb") as handle:
        coupling = pickle.load(handle)
    with (ROOT / "omni/word_kin.pkl").open("rb") as handle:
        kin = pickle.load(handle)
    with (ROOT / "omni/kin_context.pkl").open("rb") as handle:
        contexts = pickle.load(handle)
    counts = {
        "pair_count": int(pairs.get("N", 0)),
        "pair_prompt_word_count": len(pairs.get("inv", {})),
        "fluency_vocabulary": len(fluency.get("uni", {})),
        "fluency_max_order": int(fluency.get("maxl", 0)),
        "coupling_word_count": len(coupling),
        "kin_word_count": len(kin),
        "conditioned_context_count": len(contexts.get("cond", {})),
        "trigram_context_count": len(contexts.get("cond3", {})),
    }
    if any(value <= 0 for value in counts.values()):
        raise RuntimeError(f"runtime store failed non-empty content gate: {counts}")
    return counts


def resolved_revisions(cache: Path | None) -> dict:
    if cache is None or not cache.exists():
        return {}
    revisions = {}
    for ref in sorted(cache.glob("hub/datasets--*/refs/main")):
        name = ref.parents[1].name.removeprefix("datasets--").replace("--", "/")
        revision = ref.read_text().strip()
        if revision:
            revisions[name] = revision
    return revisions


def seal(output: Path, hf_cache: Path | None = None) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"runtime-store build receipt already exists: {output}")
    artifacts = {relative: artifact_record(relative) for relative in ARTIFACTS}
    counts = validate_store_content()
    try:
        import datasets
        datasets_version = datasets.__version__
    except Exception:
        datasets_version = None
    receipt = {
        "schema": "unison-runtime-store-build/v1",
        "status": "sealed",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "content_counts": counts,
        "builder_source_sha256": {
            relative: sha256(ROOT / relative) for relative in BUILDERS
        },
        "resolved_dataset_revisions": resolved_revisions(hf_cache),
        "commands": [
            "python3 train_eval/download_conversational.py",
            "python3 train_eval/build_pairs_from_datasets.py",
            "python3 train_eval/build_fluency.py",
            "python3 train_eval/build_coupling.py",
            "python3 train_eval/build_kin_store.py",
            "python3 train_eval/build_kin_context.py",
        ],
        "environment": {
            "python": platform.python_version(),
            "datasets": datasets_version,
            "platform": platform.platform(),
        },
        "rebuild_boundary": (
            "The exact generated artifacts are hash-bound. Dataset builders currently "
            "request streaming sources without pinned revision arguments; resolved cache "
            "revisions are recorded where available, so future source drift must not be "
            "mistaken for byte-identical rebuildability."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--hf-cache", type=Path)
    args = parser.parse_args()
    receipt = seal(args.output, args.hf_cache)
    print(json.dumps({"status": receipt["status"], **receipt["content_counts"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
