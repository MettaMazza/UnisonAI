import hashlib
from pathlib import Path
import pickle
import random
import json
import tempfile
import unittest

from omni.word_engine import WordEngine


class ResponseFluencyActivationTests(unittest.TestCase):
    def _store(self, path, *, role="assistant-response",
               boundary="reset-before-and-after-every-response"):
        record = {
            "schema": "unison-response-fluency/v1",
            "role": role,
            "boundary_policy": boundary,
            "maxl": 2,
            "uni": {"hello": 2, "there": 1, "friend": 1},
            "stores": [None, {("hello",): {"there": 1, "friend": 1}},
                       {("say", "hello"): {"there": 1}}],
        }
        path.write_bytes(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))

    def test_explicit_response_surface_is_hash_bound_and_generates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.pkl"
            self._store(path)
            engine = WordEngine(path)
            identity = engine.fluency_identity()
            self.assertEqual(identity["schema"], "unison-response-fluency/v1")
            self.assertEqual(identity["role"], "assistant-response")
            self.assertEqual(identity["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            word, depth, candidates = engine.sample_next_unfold(
                ["say", "hello"], random.Random(0))
            self.assertEqual((word, depth, candidates), ("there", 2, 2))

    def test_role_or_boundary_mismatch_never_activates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.pkl"
            self._store(path, role="mixed-role")
            engine = WordEngine(path)
            self.assertEqual(engine._load_fluency()["maxl"], 0)
            self.assertEqual(engine.fluency_identity(), {})

    def test_explicit_legacy_store_never_activates_as_response_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pkl"
            self._store(path)
            record = pickle.loads(path.read_bytes())
            record.pop("schema")
            path.write_bytes(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            engine = WordEngine(path)
            self.assertEqual(engine._load_fluency()["maxl"], 0)
            self.assertEqual(engine.fluency_identity(), {})

    def test_store_can_be_switched_without_renaming_live_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.pkl"
            second = Path(temporary) / "second.pkl"
            self._store(first)
            self._store(second)
            engine = WordEngine(first)
            first_hash = engine.fluency_identity()["sha256"]
            engine.configure_fluency_store(second)
            self.assertEqual(engine.fluency_identity()["sha256"],
                             hashlib.sha256(second.read_bytes()).hexdigest())
            self.assertEqual(first_hash, hashlib.sha256(first.read_bytes()).hexdigest())

    def test_registered_runtime_arm_binds_receipt_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "response.pkl"
            receipt = root / "receipt.json"
            arm = root / "arm.json"
            self._store(artifact)
            artifact_binding = {
                "path": str(artifact),
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            receipt.write_text(json.dumps({
                "schema": "unison-response-fluency-receipt/v1",
                "status": "sealed",
                "artifact": artifact_binding,
                "role": "assistant-response",
                "boundary_policy": "reset-before-and-after-every-response",
            }))
            arm.write_text(json.dumps({
                "schema": "unison-response-fluency-runtime-arm/v1",
                "status": "registered",
                "receipt": {
                    "path": str(receipt),
                    "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                },
                "artifact": artifact_binding,
            }))
            identity = WordEngine().configure_registered_fluency(arm)
            self.assertEqual(identity["sha256"], artifact_binding["sha256"])
            self.assertEqual(identity["runtime_arm"]["path"], str(arm.resolve()))

    def test_registered_runtime_arm_halts_on_receipt_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arm = root / "arm.json"
            receipt = root / "receipt.json"
            receipt.write_text("{}")
            arm.write_text(json.dumps({
                "schema": "unison-response-fluency-runtime-arm/v1",
                "status": "registered",
                "receipt": {"path": str(receipt), "sha256": "0" * 64},
                "artifact": {},
            }))
            with self.assertRaisesRegex(RuntimeError, "receipt hash"):
                WordEngine().configure_registered_fluency(arm)


if __name__ == "__main__":
    unittest.main()
