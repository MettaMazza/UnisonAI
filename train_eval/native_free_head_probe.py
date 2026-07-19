#!/usr/bin/env python3
"""Inspect exact first-token organs on the free multi-turn transcript panel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer
from train_eval.native_free_multiturn_probe import CONVERSATIONS


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top(distribution: dict[int, Fraction], words: list[str]) -> list[dict]:
    ranked = sorted(distribution, key=lambda token_id: (
        -distribution[token_id], token_id))
    return [
        {"token": words[token_id], "share": str(distribution[token_id])}
        for token_id in ranked[:8]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"head probe output exists: {output}")
    model = CountedCausalTransformer()
    record = model._store()
    bos = record["bos_id"]
    rows = []
    for name, statement, followup in CONVERSATIONS:
        first = model.generate(statement)
        history = [("user", statement), ("assistant", first)]
        context = model._contextual_decoder_state(followup, history)
        keys = context.token_keys
        attention = model._attention(bos, bos, keys)
        semantic = model._semantic_ffn(bos, bos, keys)
        prefix = model._ffn(bos, bos)
        combined = model.next_distribution(bos, bos, keys)
        rows.append({
            "conversation": name,
            "statement": statement,
            "first_surface": first,
            "followup": followup,
            "copy_admitted": model._induction_copy_admitted(context),
            "heads": {
                "attention": top(attention, record["words"]),
                "semantic_ffn": top(semantic, record["words"]),
                "causal_prefix_ffn": top(prefix, record["words"]),
                "combined_lm_head": top(combined, record["words"]),
            },
        })
    result = {
        "schema": "unison-native-free-head-probe/v1",
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "Exact development instrument; no pass condition or conclusion authority.",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "identity": model.identity(),
        "sources": {
            "omni/native_transformer.py": sha256(
                ROOT / "omni/native_transformer.py"),
            "train_eval/native_free_head_probe.py": sha256(Path(__file__)),
            "train_eval/native_free_multiturn_probe.py": sha256(
                ROOT / "train_eval/native_free_multiturn_probe.py"),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps({"schema": result["schema"], "status": result["status"],
                      "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
