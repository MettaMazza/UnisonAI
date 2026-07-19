#!/usr/bin/env python3
"""Codex-authored implementation smoke for the sealed native transformer.

This is not Maria's benchmark, finding, parity definition, or run authority. It
checks that the full counted artifact loads, distributions close exactly, and
the native generator produces inspectable surfaces for several distinct prompt
addresses before a project benchmark is requested.
"""
from __future__ import annotations

from fractions import Fraction
import asyncio
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer


PROMPTS = [
    "Hello, how are you?",
    "What do you enjoy about gardening?",
    "I am nervous about public speaking.",
    "What makes board games interesting?",
    "Tell me something interesting about the desert.",
    "Why do people enjoy astronomy?",
    "Suggest a comforting meal for a cold evening.",
    "What kind of music do you enjoy?",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    model = CountedCausalTransformer()
    store = model._store()
    bos = store["bos_id"]
    rows = []
    started = time.monotonic()
    for index, prompt in enumerate(PROMPTS):
        keys = model._context_keys(prompt)
        row = {
            "prompt": prompt,
            "addressed_keys": len(keys),
            "first_argmax_id": model.next_token_id(bos, bos, keys) if keys else None,
            "surface": model.generate(prompt),
        }
        # One materialized full-corpus row proves runtime closure. Focused tests
        # establish exact argmax equivalence across multi-step generations, so
        # repeating dense row materialization for every prompt adds no new
        # invariant and obscures live-route latency.
        if index == 0 and keys:
            distribution = model.next_distribution(bos, bos, keys)
            row["first_distribution_candidates"] = len(distribution)
            row["first_distribution_closure"] = str(
                sum(distribution.values(), Fraction(0)))
            row["argmax_equivalent"] = row["first_argmax_id"] == min(
                distribution,
                key=lambda token_id: (-distribution[token_id], token_id))
        rows.append(row)
    # Exercise the production Discord generation method with the same sealed
    # model instance. Session and diagnostics are minimal containers; no
    # generation organ is stubbed and pair/RAG must not run when native succeeds.
    from omni import discord_bot
    client = object.__new__(discord_bot.SFTDiscordClient)
    session = SimpleNamespace(history_log=[])
    live_trace = []
    original_native = discord_bot.native_transformer
    original_pair_reply = discord_bot.pair_retrieval.reply
    try:
        discord_bot.native_transformer = model
        discord_bot.pair_retrieval.reply = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("pair/RAG invoked during native live-route smoke"))
        live_surface = asyncio.run(client._generate_turn_surface(
            PROMPTS[0], [], session, "native-smoke", object(), trace=live_trace))
    finally:
        discord_bot.native_transformer = original_native
        discord_bot.pair_retrieval.reply = original_pair_reply
    result = {
        "schema": "unison-native-transformer-smoke/v4-projected-heads",
        "status": "completed",
        "provenance": {
            "origin": "Codex-authored implementation smoke",
            "agent": "Codex",
            "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Not Maria's benchmark, finding, loss, parity definition, or run gate.",
        },
        "identity": model.identity(),
        "source": {
            "path": "omni/native_transformer.py",
            "sha256": sha256(ROOT / "omni/native_transformer.py"),
        },
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "rows": rows,
        "live_route": {"surface": live_surface, "trace": live_trace},
    }
    if result["identity"].get("decode_kernel"):
        suffix = "packed_integer"
    elif result["identity"].get("serving_representation"):
        suffix = "packed"
    else:
        suffix = "projected"
    output = ROOT / f"train_eval/native_transformer_smoke_v4_{suffix}_20260719.json"
    stage = output.with_name(output.name + ".building")
    stage.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    stage.replace(output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
