#!/usr/bin/env python3
"""Build the complete position-owned native value/FFN observation relation.

One canonical counted relation stores

    (relative_position, prompt_token, last, previous, next) -> count

Its exact marginals are the established position-owned attention value,
semantic-FFN2, and semantic-FFN3 tables.  Storing the observation once avoids
three copies without sampling, pruning, capping, or changing any count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import _prompt_tokens, _tokens


SCHEMA = "unison-position-conditioned-canonical-relation/v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def observations(prompt: str, response: str, vocab: dict[str, int],
                 bos_id: int, eos_id: int) -> Iterator[tuple[int, int, int, int, int]]:
    """Yield every role-bound position/target observation exactly once."""
    prompt_ids = []
    for token in _prompt_tokens(prompt):
        token_id = vocab.get(token)
        if token_id is None:
            raise RuntimeError(f"prompt token absent from sealed vocabulary: {token!r}")
        prompt_ids.append(token_id)
    response_ids = []
    for token in _tokens(response):
        token_id = vocab.get(token)
        if token_id is None:
            raise RuntimeError(f"response token absent from sealed vocabulary: {token!r}")
        response_ids.append(token_id)

    previous = last = bos_id
    width = len(prompt_ids)
    for next_id in response_ids + [eos_id]:
        for position, key_id in enumerate(prompt_ids):
            relative_position = width - position - 1
            yield relative_position, key_id, last, previous, next_id
        previous, last = last, next_id


def marginal_counts(rows: Iterable[tuple[int, int, int, int, int]]) -> dict:
    """Fixture/reference projection proving the single-relation factorisation."""
    from collections import Counter

    value = Counter()
    semantic2 = Counter()
    semantic3 = Counter()
    canonical = Counter(rows)
    for (position, key_id, last, previous, next_id), count in canonical.items():
        value[(position, key_id, next_id)] += count
        semantic2[(last, position, key_id, next_id)] += count
        semantic3[(previous, last, position, key_id, next_id)] += count
    return {
        "canonical": canonical,
        "value": value,
        "semantic2": semantic2,
        "semantic3": semantic3,
    }


def atomic_json(path: Path, value: dict) -> None:
    stage = path.with_name(path.name + ".writing")
    stage.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(stage, path)


def write_raw(pairs: dict, metadata: dict, raw: Path, state_path: Path,
              source_identity: dict) -> dict:
    state = {
        "schema": SCHEMA + "/build-state",
        "phase": "raw",
        "pair_cursor": 0,
        "observations": 0,
        "raw_bytes": 0,
        "inputs": source_identity,
    }
    if state_path.exists():
        held = json.loads(state_path.read_text())
        if held.get("inputs") != source_identity or held.get("phase") != "raw":
            raise RuntimeError("position-relation build-state provenance mismatch")
        state = held

    raw.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if raw.exists() else "w+b"
    with raw.open(mode) as handle:
        handle.truncate(state["raw_bytes"])
        handle.seek(state["raw_bytes"])
        pending: list[str] = []
        for index in range(state["pair_cursor"], len(pairs["prompts"])):
            for row in observations(
                    pairs["prompts"][index], pairs["responses"][index],
                    metadata["vocab"], metadata["bos_id"], metadata["eos_id"]):
                pending.append("\t".join(map(str, row)) + "\n")
                state["observations"] += 1
            if len(pending) >= 32768:
                handle.write("".join(pending).encode("ascii"))
                pending.clear()
            # Checkpointing changes only recovery granularity, never relation
            # membership. The raw byte boundary is flushed before publication.
            if (index + 1) % 1024 == 0:
                if pending:
                    handle.write("".join(pending).encode("ascii"))
                    pending.clear()
                handle.flush()
                os.fsync(handle.fileno())
                state["pair_cursor"] = index + 1
                state["raw_bytes"] = handle.tell()
                atomic_json(state_path, state)
        if pending:
            handle.write("".join(pending).encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
        state["pair_cursor"] = len(pairs["prompts"])
        state["raw_bytes"] = handle.tell()
        atomic_json(state_path, state)
    return state


def sort_raw(raw: Path, sorted_path: Path, temporary: Path,
             memory: str, workers: int) -> None:
    temporary.mkdir(parents=True, exist_ok=True)
    stage = sorted_path.with_name(sorted_path.name + ".sorting")
    command = [
        "sort", "-S", memory, "--parallel", str(workers), "-T", str(temporary),
        "-n", "-k1,1", "-k2,2", "-k3,3", "-k4,4", "-k5,5",
        "-o", str(stage), str(raw),
    ]
    subprocess.run(command, check=True)
    os.replace(stage, sorted_path)


def aggregate(sorted_path: Path, relation_path: Path) -> tuple[int, int]:
    stage = relation_path.with_name(relation_path.name + ".aggregating")
    unique = total = 0
    held = None
    count = 0
    with sorted_path.open("rt", encoding="ascii") as source, \
            stage.open("wt", encoding="ascii") as target:
        for line in source:
            row = line.rstrip("\n")
            if row == held:
                count += 1
                continue
            if held is not None:
                target.write(f"{held}\t{count}\n")
                unique += 1
                total += count
            held = row
            count = 1
        if held is not None:
            target.write(f"{held}\t{count}\n")
            unique += 1
            total += count
        target.flush()
        os.fsync(target.fileno())
    os.replace(stage, relation_path)
    return unique, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "omni/pairs.pkl")
    parser.add_argument("--metadata", type=Path,
                        default=ROOT / "omni/native_transformer_v4_packed/metadata.pkl")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "train_eval/native_position_relation_v5")
    parser.add_argument("--sort-memory", default="8G")
    parser.add_argument("--sort-workers", type=int, default=4)
    args = parser.parse_args()
    if args.sort_workers <= 0:
        raise ValueError("sort workers must be positive")

    pairs_path = args.pairs.resolve()
    metadata_path = args.metadata.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("rb") as handle:
        pairs = pickle.load(handle)
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    source_identity = {
        "pairs_sha256": sha256(pairs_path),
        "metadata_sha256": sha256(metadata_path),
        "source_sha256": sha256(Path(__file__)),
    }
    raw = output / "observations.tsv"
    sorted_path = output / "observations.sorted.tsv"
    relation = output / "relation.tsv"
    state_path = output / "build_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("inputs") != source_identity:
            raise RuntimeError("position-relation build-state provenance mismatch")
    else:
        state = {"phase": "raw"}

    if state["phase"] == "completed":
        receipt = json.loads((output / "receipt.json").read_text())
        print(json.dumps(receipt, sort_keys=True))
        return
    if state["phase"] == "raw":
        state = write_raw(pairs, metadata, raw, state_path, source_identity)
        state["phase"] = "sort"
        atomic_json(state_path, state)
    if state["phase"] == "sort":
        sort_raw(raw, sorted_path, output / "sort_tmp",
                 args.sort_memory, args.sort_workers)
        state["phase"] = "aggregate"
        atomic_json(state_path, state)
    if state["phase"] != "aggregate":
        raise RuntimeError(f"unknown position-relation build phase: {state['phase']!r}")
    unique, total = aggregate(sorted_path, relation)
    if total != state["observations"]:
        raise RuntimeError("aggregated relation did not preserve every observation")

    result = {
        "schema": SCHEMA,
        "status": "completed",
        "inputs": source_identity,
        "pairs": len(pairs["prompts"]),
        "observations": total,
        "unique_canonical_entries": unique,
        "relation_sha256": sha256(relation),
        "relation_bytes": relation.stat().st_size,
        "resource_configuration": {
            "sort_memory": args.sort_memory,
            "sort_workers": args.sort_workers,
            "effect": "execution resources only; relation membership is unchanged",
        },
        "factorisation": {
            "canonical": "(relative_position,prompt_token,last,previous,next)->count",
            "exact_marginals": ["position_value", "position_semantic2",
                                "position_semantic3"],
            "observations_preserved": total,
        },
        "provenance": {
            "origin": "Codex implementation of documented transformer training port",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Implementation artifact; Maria owns conclusions and benchmark timing.",
        },
    }
    atomic_json(output / "receipt.json", result)
    state["phase"] = "completed"
    atomic_json(state_path, state)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
