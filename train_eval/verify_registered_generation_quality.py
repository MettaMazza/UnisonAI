#!/usr/bin/env python3
"""Verify a sealed registered generation-quality result without judging it again."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify(campaign_path: Path, result_dir: Path, verify_runtime: bool) -> dict:
    campaign_path = campaign_path.resolve()
    result_dir = result_dir.resolve()
    campaign = json.loads(campaign_path.read_text())
    registration_path = result_dir / "registration.json"
    calibration_path = result_dir / "calibration.json"
    result_path = result_dir / "result.json"
    seal_path = result_dir / "seal.json"
    for path in (registration_path, calibration_path, result_path, seal_path):
        require(path.is_file(), f"missing sealed evidence file: {path.name}")

    registration = json.loads(registration_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    result = json.loads(result_path.read_text())
    seal = json.loads(seal_path.read_text())
    require(campaign.get("schema") == "unison-generation-quality-campaign/v1",
            "unsupported campaign schema")
    require(registration.get("schema") == "unison-generation-quality-registration/v1",
            "unsupported registration schema")
    require(registration.get("status") == "registered", "registration is not complete")
    require(result.get("schema") == "unison-generation-quality-result/v1" and
            result.get("status") == "completed", "result is not completed")
    require(seal.get("schema") == "unison-generation-quality-seal/v1" and
            seal.get("status") == "completed", "result seal is not completed")

    campaign_sha = sha256_file(campaign_path)
    registration_sha = sha256_file(registration_path)
    calibration_sha = sha256_file(calibration_path)
    result_sha = sha256_file(result_path)
    require(registration.get("campaign_sha256") == campaign_sha,
            "registration does not bind this campaign")
    require(seal.get("registration_sha256") == registration_sha and
            result.get("registration_sha256") == registration_sha,
            "registration hash mismatch")
    require(seal.get("calibration_sha256") == calibration_sha and
            result.get("calibration_sha256") == calibration_sha,
            "calibration hash mismatch")
    require(seal.get("result_sha256") == result_sha, "result hash mismatch")

    require(calibration.get("passed") is True, "judge calibration did not pass")
    per_judge = calibration.get("per_judge", {})
    require(set(per_judge) == {"judge_1_good", "judge_2_good"},
            "calibration judge set mismatch")
    for name, score in per_judge.items():
        require(score.get("passed") is True and score.get("known_good", 0) >= 9 and
                score.get("known_bad", 0) >= 9, f"invalid calibration gate: {name}")

    prompts = campaign["prompts"]
    arms = campaign["arms"]
    rows = result.get("rows", [])
    require(len(rows) == len(prompts), "result prompt count mismatch")
    totals = {arm: 0 for arm in arms}
    for prompt, row in zip(prompts, rows):
        require(row.get("prompt") == prompt, "result prompt order/content mismatch")
        require(set(row.get("replies", {})) == set(arms), "result arm replies mismatch")
        require(set(row.get("verdicts", {})) == set(arms), "result arm verdicts mismatch")
        for arm in arms:
            verdict = row["verdicts"][arm]
            left = verdict.get("judge_1_good")
            right = verdict.get("judge_2_good")
            require(isinstance(left, bool) and isinstance(right, bool),
                    "non-Boolean judge verdict")
            require(verdict.get("pool_good") is (left and right),
                    "pool verdict violates registered unanimity rule")
            require(verdict.get("agreement") is (left == right),
                    "judge agreement field is inconsistent")
            totals[arm] += int(verdict["pool_good"])
    for arm in arms:
        summary = result.get("pool_good", {}).get(arm, {})
        require(summary.get("good") == totals[arm] and
                summary.get("total") == len(prompts) and
                summary.get("rate") == totals[arm] / len(prompts),
                f"summary mismatch for arm: {arm}")

    for relative, expected in registration.get("source_sha256", {}).items():
        source = ROOT / relative
        require(source.is_file() and sha256_file(source) == expected,
                f"registered source drift: {relative}")
    if verify_runtime:
        for relative, binding in registration.get("runtime_artifacts", {}).items():
            if binding.get("status") != "present":
                continue
            artifact = ROOT / relative
            require(artifact.is_file() and artifact.stat().st_size == binding["bytes"] and
                    sha256_file(artifact) == binding["sha256"],
                    f"registered runtime artifact drift: {relative}")

    return {
        "schema": "unison-generation-quality-verification/v1",
        "status": "verified",
        "calibration_passed": True,
        "prompt_count": len(prompts),
        "pool_good": result["pool_good"],
        "runtime_artifacts_checked": verify_runtime,
        "seal_sha256": sha256_file(seal_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--verify-runtime", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.campaign, args.result_dir, args.verify_runtime),
                     sort_keys=True))


if __name__ == "__main__":
    main()
