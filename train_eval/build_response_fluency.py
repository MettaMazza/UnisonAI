#!/usr/bin/env python3
"""Build an assistant-response-only fluency store with hard response boundaries."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.word_engine import tokenize
from train_eval.build_kin_context import merge_contractions


SCHEMA = "unison-response-fluency/v1"
ROLE = "assistant-response"
BOUNDARY_POLICY = "reset-before-and-after-every-response"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_tokens(text: str) -> list[str]:
    return [token.lower() for token in merge_contractions(tokenize(text))]


def build_store(pairs_path: Path, output_path: Path, max_order: int = 4) -> dict:
    pairs_path = pairs_path.resolve()
    output_path = output_path.resolve()
    if max_order < 1:
        raise ValueError("max_order must be positive")
    with pairs_path.open("rb") as handle:
        pairs = pickle.load(handle)
    responses = pairs.get("responses")
    if not isinstance(responses, list) or not responses:
        raise RuntimeError("pairs store has no assistant responses")

    stores = [None] + [defaultdict(Counter) for _ in range(max_order)]
    unigram = Counter()
    response_count = 0
    token_count = 0
    for response in responses:
        tokens = response_tokens(response)
        if not tokens:
            continue
        response_count += 1
        token_count += len(tokens)
        unigram.update(tokens)
        # Contexts are constructed only inside this response. No state survives
        # the loop boundary, so cross-response n-grams cannot enter the store.
        for index, word in enumerate(tokens):
            for order in range(1, max_order + 1):
                if index < order:
                    break
                stores[order][tuple(tokens[index - order:index])][word] += 1
    if response_count <= 0 or token_count <= 0 or not unigram:
        raise RuntimeError("assistant responses produced an empty fluency store")

    record = {
        "schema": SCHEMA,
        "role": ROLE,
        "boundary_policy": BOUNDARY_POLICY,
        "source_pairs_sha256": sha256(pairs_path),
        "response_count": response_count,
        "token_count": token_count,
        "maxl": max_order,
        "uni": dict(unigram),
        "stores": [None] + [
            {context: dict(counts) for context, counts in stores[order].items()}
            for order in range(1, max_order + 1)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stage = output_path.with_name(output_path.name + ".building")
    try:
        with stage.open("wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(stage, output_path)
    except BaseException:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass
        raise
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "omni/pairs.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "omni/response_fluency_v1.pkl")
    args = parser.parse_args()
    started = time.monotonic()
    record = build_store(args.pairs, args.output)
    print(f"sealed {record['response_count']:,} assistant responses, "
          f"{record['token_count']:,} tokens, {len(record['uni']):,} words "
          f"in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
