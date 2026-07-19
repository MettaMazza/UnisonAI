#!/usr/bin/env python3
"""Size the complete position-conditioned native training relation exactly.

The next documented transformer port gives every observed prompt occurrence a
relative-position-owned value/semantic address.  This instrument counts the
complete corpus deposition work before that artifact is materialised.  It does
not sample, cap, prune, or alter serving behaviour.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import _prompt_tokens, _tokens


PAIRS = ROOT / "omni/pairs.pkl"
OUTPUT = ROOT / "train_eval/position_conditioned_training_cost_20260719.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with PAIRS.open("rb") as handle:
        pairs = pickle.load(handle)

    prompt_positions = 0
    causal_targets = 0
    deposited_observations = 0
    maximum_pair_observations = 0
    for prompt, response in zip(pairs["prompts"], pairs["responses"]):
        positions = len(_prompt_tokens(prompt))
        targets = len(_tokens(response)) + 1  # learned EOS is a target
        observations = positions * targets
        prompt_positions += positions
        causal_targets += targets
        deposited_observations += observations
        maximum_pair_observations = max(maximum_pair_observations, observations)

    # Each observation can create at most one new row/value entry in each of
    # the three established position-owned tables.  This exact worst case is a
    # safe allocation bound; aggregation can only reduce it.
    relation_count = 3
    result = {
        "schema": "unison-position-conditioned-training-cost/v1",
        "status": "completed",
        "provenance": {
            "origin": "Codex-authored exact implementation sizing",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": (
                "Not Maria's benchmark, finding, parity definition, or run "
                "gate. Counts complete artifact growth only."
            ),
        },
        "inputs": {
            "pairs_sha256": sha256(PAIRS),
            "source_sha256": sha256(Path(__file__)),
        },
        "prompt_response_pairs": len(pairs["prompts"]),
        "prompt_positions": prompt_positions,
        "causal_targets": causal_targets,
        "position_target_observations": deposited_observations,
        "maximum_single_pair_observations": maximum_pair_observations,
        "planned_complete_relations": [
            "position_value(relative_position, token)->next_token",
            "position_semantic2(last, relative_position, token)->next_token",
            "position_semantic3(previous, last, relative_position, token)->next_token",
        ],
        "maximum_rows_per_relation": deposited_observations,
        "maximum_row_value_entries_all_relations": (
            relation_count * deposited_observations
        ),
        "implementation_guard": (
            "Materialise every observed relation or halt. Do not introduce a "
            "position limit, candidate cap, sampled corpus, pruning rule, or "
            "fitted capacity. Aggregate repeated observations before packing."
        ),
    }
    stage = OUTPUT.with_name(OUTPUT.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(OUTPUT)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
