#!/usr/bin/env python3
"""Run a cumulative native-v5 conversation and reward-learning benchmark."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omni.native_transformer import CountedCausalTransformer  # noqa: E402


OUTPUT = ROOT / "train_eval/native_v5_conversation_learning_trial_20260719.json"
REWARD = ROOT / "train_eval/native_v5_conversation_learning_reward_20260719.pkl"
SCENARIOS = (
    ("name", "My name is Maria.", "What is my name?", ("maria",)),
    ("preference", "I prefer tea to coffee.", "Which drink do I prefer?", ("tea",)),
    ("garden", "I am growing tomatoes in my garden.", "What am I growing?", ("tomato",)),
    ("feeling", "I feel nervous about speaking tomorrow.", "How did I say I feel?", ("nervous",)),
    ("place", "I left the notebook in the kitchen.", "Where is the notebook?", ("kitchen",)),
    ("schedule", "My appointment is on Thursday.", "When is my appointment?", ("thursday",)),
    ("project", "The current project is about protein folding.", "What is the project about?", ("protein", "folding")),
    ("comparison", "The blue box is larger than the red box.", "Which box is larger?", ("blue",)),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(model: CountedCausalTransformer, prompt: str, history=None) -> tuple[str, float]:
    began = time.monotonic()
    surface = model.generate(prompt, history=history)
    return surface, time.monotonic() - began


def main() -> None:
    if OUTPUT.exists() or REWARD.exists():
        raise FileExistsError("conversation/learning development artifacts already exist")
    model = CountedCausalTransformer(reward_path=str(REWARD))
    began = time.monotonic()
    conversation_rows = []
    for name, statement, question, expected in SCENARIOS:
        first, first_seconds = generate(model, statement)
        history = [("user", statement), ("assistant", first)]
        followup, followup_seconds = generate(model, question, history=history)
        lowered = followup.lower()
        conversation_rows.append({
            "scenario": name,
            "turns": [
                {"role": "user", "content": statement},
                {"role": "assistant", "content": first, "seconds": first_seconds},
                {"role": "user", "content": question},
                {"role": "assistant", "content": followup, "seconds": followup_seconds},
            ],
            "first_nonempty": bool(first.strip()),
            "followup_nonempty": bool(followup.strip()),
            "auxiliary_expected_any": list(expected),
            "auxiliary_continuity_observed": any(token in lowered for token in expected),
        })

    reward_prompt = "What kind of music do you enjoy?"
    baseline, baseline_seconds = generate(model, reward_prompt)
    model.mark_feedback(reward_prompt, baseline, good=False)
    after_negative, negative_seconds = generate(model, reward_prompt)
    model.mark_feedback(reward_prompt, after_negative, good=True)
    after_positive, positive_seconds = generate(model, reward_prompt)
    result = {
        "schema": "unison-native-v5-conversation-learning-trial/v1",
        "status": "completed",
        "result_type": "cumulative development benchmark",
        "official_run": False,
        "benchmark_authority": False,
        "governance_authority": False,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "provenance": {
            "origin": "Codex implementation development run",
            "agent": "Codex", "model": "gpt-5.6-sol",
            "reasoning_level": "high",
            "authority": "Maria Smith alone registers official trials and assigns conclusions.",
        },
        "declared_purpose": (
            "Exercise broader free multi-turn serving and the required exact "
            "reward-conditioned learning ledger on the optimized sealed v5 route."
        ),
        "conversation": {
            "scenarios": len(conversation_rows),
            "first_nonempty": sum(row["first_nonempty"] for row in conversation_rows),
            "followup_nonempty": sum(row["followup_nonempty"] for row in conversation_rows),
            "auxiliary_continuity_observed": sum(
                row["auxiliary_continuity_observed"] for row in conversation_rows),
            "rows": conversation_rows,
        },
        "reward_learning": {
            "prompt": reward_prompt,
            "baseline_surface": baseline,
            "after_negative_surface": after_negative,
            "after_positive_surface": after_positive,
            "negative_observation_changed_surface": baseline != after_negative,
            "positive_observation_retained_surface": after_negative == after_positive,
            "seconds": {
                "baseline": baseline_seconds,
                "after_negative": negative_seconds,
                "after_positive": positive_seconds,
            },
            "reward_artifact": {"path": str(REWARD.relative_to(ROOT)), "sha256": sha256(REWARD)},
        },
        "elapsed_seconds": time.monotonic() - began,
        "maximum_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "identity": model.identity(),
        "sources": {
            "omni/native_transformer.py": sha256(ROOT / "omni/native_transformer.py"),
            "train_eval/native_v5_conversation_learning_trial.py": sha256(Path(__file__)),
            "train_eval/native_transformer_v5_runtime_receipt.json": sha256(
                ROOT / "train_eval/native_transformer_v5_runtime_receipt.json"),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "completed",
        "scenarios": len(conversation_rows),
        "first_nonempty": result["conversation"]["first_nonempty"],
        "followup_nonempty": result["conversation"]["followup_nonempty"],
        "continuity": result["conversation"]["auxiliary_continuity_observed"],
        "negative_changed": result["reward_learning"]["negative_observation_changed_surface"],
        "positive_retained": result["reward_learning"]["positive_observation_retained_surface"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
