#!/usr/bin/env python3
"""Build the role-bound counted causal-transformer state from the pair corpus."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import build_counted_transformer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=ROOT / "omni/pairs.pkl")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "omni/native_transformer_v4.pkl")
    args = parser.parse_args()
    started = time.monotonic()
    pairs_path = args.pairs.resolve()
    with pairs_path.open("rb") as handle:
        pairs = pickle.load(handle)
    record = build_counted_transformer(
        pairs["prompts"], pairs["responses"], source_sha256=sha256(pairs_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output.with_name(args.output.name + ".building")
    try:
        with stage.open("wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(stage, args.output)
    except BaseException:
        try:
            stage.unlink()
        except FileNotFoundError:
            pass
        raise
    print(f"built {record['response_count']:,} role-bound sequences, "
          f"{record['token_count']:,} causal targets, {len(record['vocab']):,} tokens, "
          f"{len(record['qk']):,} Q/K relations, {len(record['values']):,} value vectors, "
          f"{len(record['semantic_ffn']):,}/{len(record['semantic_ffn3']):,} semantic FFN2/3 keys, "
          f"{len(record['ffn3']):,} FFN keys in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
