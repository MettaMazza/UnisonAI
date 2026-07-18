#!/usr/bin/env python3
"""Two-stage immutable registration and execution for free-generation quality.

Stage 1 binds the campaign, source, runtime stores, and local judge digests.
Stage 2 revalidates every hash, calibrates both judges, generates every declared
arm, and seals results in a directory that must not already exist.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_bytes(record: dict) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()


def load_campaign(path: Path) -> tuple[dict, bytes]:
    raw = path.resolve().read_bytes()
    campaign = json.loads(raw)
    if campaign.get("schema") not in {
            "unison-generation-quality-campaign/v1",
            "unison-generation-quality-campaign/v2"}:
        raise ValueError("unsupported generation-quality campaign")
    for relative, expected in campaign["source_sha256"].items():
        source = ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise RuntimeError(f"campaign source drift: {relative}")
    return campaign, raw


def response_arm_state(campaign: dict) -> dict | None:
    relative = campaign.get("response_fluency_arm")
    if not relative:
        return None
    arm_path = ROOT / relative
    arm = json.loads(arm_path.read_text())
    if arm.get("schema") != "unison-response-fluency-runtime-arm/v1" or \
            arm.get("status") != "registered":
        raise RuntimeError("campaign response-fluency arm is not registered")
    receipt_path = ROOT / arm["receipt"]["path"]
    artifact_path = ROOT / arm["artifact"]["path"]
    if sha256_file(receipt_path) != arm["receipt"]["sha256"] or \
            sha256_file(artifact_path) != arm["artifact"]["sha256"] or \
            artifact_path.stat().st_size != arm["artifact"]["bytes"]:
        raise RuntimeError("campaign response-fluency arm binding drift")
    return {
        "path": relative,
        "sha256": sha256_file(arm_path),
        "receipt_sha256": arm["receipt"]["sha256"],
        "artifact_sha256": arm["artifact"]["sha256"],
        "artifact_bytes": arm["artifact"]["bytes"],
    }


def runtime_artifact_state(campaign: dict) -> dict:
    state = {}
    missing = []
    for relative in campaign["required_runtime_artifacts"]:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
        else:
            state[relative] = {"status": "present", "sha256": sha256_file(path),
                               "bytes": path.stat().st_size}
    for relative in campaign.get("optional_runtime_artifacts", []):
        path = ROOT / relative
        state[relative] = ({"status": "present", "sha256": sha256_file(path),
                            "bytes": path.stat().st_size} if path.is_file()
                           else {"status": "absent"})
    if missing:
        raise FileNotFoundError("missing required runtime artifacts: " + ", ".join(missing))
    receipt_path = ROOT / campaign["store_build_receipt"]
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != "unison-runtime-store-build/v1" or \
            receipt.get("status") != "sealed":
        raise RuntimeError("runtime-store build receipt is not sealed")
    for relative, binding in receipt["artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"runtime-store artifact drift from build receipt: {relative}")
    return state


def ollama_digests(campaign: dict) -> dict:
    endpoint = campaign["judge_endpoint"].rstrip("/") + "/api/tags"
    with urllib.request.urlopen(endpoint, timeout=10) as response:
        models = json.loads(response.read().decode()).get("models", [])
    available = {row.get("name"): row.get("digest") for row in models}
    bound = {}
    for model in campaign["judges"]:
        digest = available.get(model)
        if not digest:
            raise RuntimeError(f"registered judge model is unavailable: {model}")
        bound[model] = digest
    return bound


def build_registration(campaign_path: Path) -> dict:
    campaign, raw = load_campaign(campaign_path)
    return {
        "schema": ("unison-generation-quality-registration/v2"
                   if campaign["schema"].endswith("/v2") else
                   "unison-generation-quality-registration/v1"),
        "campaign_sha256": sha256_bytes(raw),
        "source_sha256": campaign["source_sha256"],
        "runtime_artifacts": runtime_artifact_state(campaign),
        "judge_digests": ollama_digests(campaign),
        "seed": campaign["seed"],
        "prompts_sha256": sha256_bytes(json_bytes({"prompts": campaign["prompts"]})),
        "generation_arm": response_arm_state(campaign),
        "status": "registered",
    }


def register(campaign_path: Path, registration_path: Path) -> dict:
    registration_path = registration_path.resolve()
    if registration_path.exists():
        raise FileExistsError(f"registration already exists: {registration_path}")
    record = build_registration(campaign_path)
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_bytes(json_bytes(record))
    return record


def _judge_pair(prompt, reply):
    from train_eval.judge import judge, judge2
    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(judge, prompt, reply)
        right = executor.submit(judge2, prompt, reply)
        a = left.result()
        b = right.result()
    return {"judge_1_good": bool(a[0]), "judge_1_tail": a[1],
            "judge_2_good": bool(b[0]), "judge_2_tail": b[1],
            "pool_good": bool(a[0] and b[0]), "agreement": bool(a[0] == b[0])}


def _calibrate():
    from train_eval.judge_calibration import GOOD, BAD
    rows = []
    for expected, examples in ((True, GOOD), (False, BAD)):
        for prompt, reply in examples:
            verdict = _judge_pair(prompt, reply)
            rows.append({"expected_good": expected, "prompt": prompt,
                         "reply": reply, "verdict": verdict})
    per_judge = {}
    for key in ("judge_1_good", "judge_2_good"):
        good_hits = sum(row["verdict"][key] for row in rows if row["expected_good"])
        bad_hits = sum(not row["verdict"][key] for row in rows if not row["expected_good"])
        per_judge[key] = {"known_good": good_hits, "known_bad": bad_hits,
                          "passed": good_hits >= 9 and bad_hits >= 9}
    return {"rows": rows, "per_judge": per_judge,
            "passed": all(item["passed"] for item in per_judge.values())}


def run(campaign_path: Path, registration_path: Path, output_dir: Path) -> dict:
    campaign, campaign_raw = load_campaign(campaign_path)
    registration_raw = registration_path.resolve().read_bytes()
    registration = json.loads(registration_raw)
    if registration.get("schema") not in {
            "unison-generation-quality-registration/v1",
            "unison-generation-quality-registration/v2"}:
        raise ValueError("unsupported generation-quality registration")
    rebuilt = build_registration(campaign_path)
    for field in ("campaign_sha256", "source_sha256", "runtime_artifacts",
                  "judge_digests", "seed", "prompts_sha256", "generation_arm"):
        if rebuilt[field] != registration.get(field):
            raise RuntimeError(f"registered generation environment drift: {field}")
    if registration["campaign_sha256"] != sha256_bytes(campaign_raw):
        raise RuntimeError("campaign hash does not match registration")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"sealed result already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="unison-generation-quality-", dir=output_dir.parent))
    try:
        (stage / "registration.json").write_bytes(registration_raw)
        calibration = _calibrate()
        (stage / "calibration.json").write_bytes(json_bytes(calibration))
        if not calibration["passed"]:
            raise RuntimeError("judge calibration failed; generation scores are invalid")

        from omni.free_gen import free_gen
        from omni.pair_retrieval import pair_retrieval
        from omni.word_engine import WordEngine, word_engine, tokenize, _content_words

        word_engine._load_coupling()
        response_engine = None
        if "response_surface" in campaign["arms"]:
            if not campaign.get("response_fluency_arm"):
                raise RuntimeError("response-surface arm lacks a registered runtime arm")
            response_engine = WordEngine()
            identity = response_engine.configure_registered_fluency(
                ROOT / campaign["response_fluency_arm"])
            if identity["runtime_arm"]["sha256"] != \
                    registration["generation_arm"]["sha256"]:
                raise RuntimeError("loaded response-surface arm differs from registration")
        rngs = {arm: random.Random(campaign["seed"]) for arm in campaign["arms"]}
        rows = []
        totals = {arm: 0 for arm in campaign["arms"]}
        for prompt in campaign["prompts"]:
            schema = _content_words(tokenize(prompt)) * 2
            replies = {}
            for arm in campaign["arms"]:
                rng = rngs[arm]
                if arm == "baseline":
                    reply = (word_engine.structured_unfold(schema, rng)
                             or word_engine.unfold_response(schema, rng) or "")
                elif arm == "f1":
                    reply = free_gen.generate(prompt, rng=rng) or ""
                elif arm == "f3":
                    reply = free_gen.generate_planned(prompt, rng=rng) or ""
                elif arm == "response_surface":
                    reply = (response_engine.structured_unfold(schema, rng)
                             or response_engine.unfold_response(schema, rng) or "")
                elif arm == "pair_surface":
                    reply = pair_retrieval.reply(prompt, history=[]) or ""
                else:
                    raise RuntimeError(f"unsupported registered generation arm: {arm}")
                replies[arm] = reply.strip()
            verdicts = {}
            for arm in campaign["arms"]:
                verdicts[arm] = _judge_pair(prompt, replies[arm])
                totals[arm] += verdicts[arm]["pool_good"]
            rows.append({"prompt": prompt, "replies": replies, "verdicts": verdicts})

        n = len(campaign["prompts"])
        result = {
            "schema": ("unison-generation-quality-result/v2"
                       if campaign["schema"].endswith("/v2") else
                       "unison-generation-quality-result/v1"),
            "status": "completed",
            "registration_sha256": sha256_bytes(registration_raw),
            "calibration_sha256": sha256_file(stage / "calibration.json"),
            "rows": rows,
            "pool_good": {arm: {"good": totals[arm], "total": n,
                                "rate": totals[arm] / n}
                          for arm in campaign["arms"]},
        }
        result_bytes = json_bytes(result)
        (stage / "result.json").write_bytes(result_bytes)
        seal = {"schema": ("unison-generation-quality-seal/v2"
                           if campaign["schema"].endswith("/v2") else
                           "unison-generation-quality-seal/v1"),
                "status": "completed",
                "registration_sha256": sha256_bytes(registration_raw),
                "calibration_sha256": sha256_file(stage / "calibration.json"),
                "result_sha256": sha256_bytes(result_bytes)}
        (stage / "seal.json").write_bytes(json_bytes(seal))
        os.replace(stage, output_dir)
        return seal
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def inspect(campaign_path: Path) -> dict:
    campaign, raw = load_campaign(campaign_path)
    missing = [relative for relative in campaign["required_runtime_artifacts"]
               if not (ROOT / relative).is_file()]
    return {"schema": campaign["schema"], "campaign_sha256": sha256_bytes(raw),
            "source_status": "bound", "missing_runtime_artifacts": missing,
            "ready_to_register": not missing}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("inspect")
    check.add_argument("campaign", type=Path)
    reg = sub.add_parser("register")
    reg.add_argument("campaign", type=Path)
    reg.add_argument("registration", type=Path)
    execute = sub.add_parser("run")
    execute.add_argument("campaign", type=Path)
    execute.add_argument("registration", type=Path)
    execute.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if args.command == "inspect":
        record = inspect(args.campaign)
    elif args.command == "register":
        record = register(args.campaign, args.registration)
    else:
        record = run(args.campaign, args.registration, args.output_dir)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
