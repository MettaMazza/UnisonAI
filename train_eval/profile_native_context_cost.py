#!/usr/bin/env python3
"""Exact corpus-size guard for the next contextual-attention translation.

This instrument counts the work implied by a literal dense self-attention pass.
It does not select, cap, sample, or change the architecture.  Its purpose is to
prevent an implementation from accidentally materializing repeated layer work
that can instead be computed from shared counted relations at serving time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import _prompt_tokens


PAIRS = ROOT / "omni/pairs.pkl"
CONTEXT_LAW = Path(
    "/Users/mettamazza/Desktop/Smithian Fold Theory/"
    "constants/contextual_integration.ep")
OUTPUT = ROOT / "train_eval/native_context_cost_v4_20260719.json"
INTEGRATION_DEPTH = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def order_statistic(sorted_values: list[int], numerator: int,
                    denominator: int) -> int:
    """Nearest-rank order statistic with its integer rule explicit in source."""
    rank = max(
        1,
        (len(sorted_values) * numerator + denominator - 1) // denominator,
    )
    return sorted_values[rank - 1]


def main() -> None:
    with PAIRS.open("rb") as handle:
        pairs = pickle.load(handle)
    lengths = sorted(len(_prompt_tokens(prompt)) for prompt in pairs["prompts"])
    prompt_count = len(lengths)
    position_count = sum(lengths)
    attention_pairs = sum(length * length for length in lengths)
    layered_pairs = INTEGRATION_DEPTH * attention_pairs
    result = {
        "schema": "unison-native-context-cost/v1",
        "status": "completed",
        "provenance": {
            "origin": "Codex-authored exact implementation sizing",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": (
                "Not Maria's benchmark, finding, parity definition, or run gate. "
                "Counts implementation growth only."
            ),
        },
        "inputs": {
            "pairs_sha256": sha256(PAIRS),
            "contextual_integration_law_sha256": sha256(CONTEXT_LAW),
            "source_sha256": sha256(Path(__file__)),
        },
        "integration_depth": INTEGRATION_DEPTH,
        "prompt_count": prompt_count,
        "prompt_positions": position_count,
        "prompt_length": {
            "maximum": lengths[-1],
            "median_nearest_rank": order_statistic(lengths, 1, 2),
            "p95_nearest_rank": order_statistic(lengths, 95, 100),
            "p99_nearest_rank": order_statistic(lengths, 99, 100),
            "mean_exact": {
                "numerator": position_count,
                "denominator": prompt_count,
            },
        },
        "dense_position_pairs_per_layer": attention_pairs,
        "dense_position_pair_touches_at_forced_depth": layered_pairs,
        "uint64_value_only_lower_bound_bytes": layered_pairs * 8,
        "implementation_guard": (
            "Do not materialize per-layer dense position-pair rows. Preserve the "
            "five-layer computation through shared counted relations and "
            "response-local streamed position state."
        ),
    }
    stage = OUTPUT.with_name(OUTPUT.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(OUTPUT)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
