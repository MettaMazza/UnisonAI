from pathlib import Path
import pickle
import tempfile
import unittest

from train_eval.build_response_fluency import (
    BOUNDARY_POLICY,
    ROLE,
    SCHEMA,
    build_store,
)
from train_eval.seal_response_fluency import seal


class ResponseFluencyTests(unittest.TestCase):
    def build(self, prompts, responses):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        pairs = root / "pairs.pkl"
        output = root / "response_fluency.pkl"
        with pairs.open("wb") as handle:
            pickle.dump({"prompts": prompts, "responses": responses}, handle)
        record = build_store(pairs, output)
        with output.open("rb") as handle:
            loaded = pickle.load(handle)
        return temporary, record, loaded

    def test_only_assistant_responses_enter_the_store(self):
        temporary, record, loaded = self.build(
            ["forbidden_prompt_token"], ["Hello there."])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(record["schema"], SCHEMA)
        self.assertEqual(record["role"], ROLE)
        self.assertEqual(record["boundary_policy"], BOUNDARY_POLICY)
        self.assertNotIn("forbidden_prompt_token", loaded["uni"])
        self.assertIn("hello", loaded["uni"])

    def test_response_boundary_prevents_cross_response_ngrams(self):
        temporary, _, loaded = self.build(
            ["p1", "p2"], ["Hello there.", "Cook dinner."])
        self.addCleanup(temporary.cleanup)
        for order in range(1, loaded["maxl"] + 1):
            for context, continuations in loaded["stores"][order].items():
                self.assertFalse(context[-1] == "." and "cook" in continuations)

    def test_counts_are_exact_and_contractions_are_merged(self):
        temporary, record, loaded = self.build(
            ["p", "p"], ["I don't know.", "I don't know."])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(record["response_count"], 2)
        self.assertEqual(loaded["uni"]["don't"], 2)
        self.assertEqual(loaded["stores"][1][("don't",)]["know"], 2)

    def test_empty_response_store_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.pkl"
            with pairs.open("wb") as handle:
                pickle.dump({"prompts": ["only prompt"], "responses": []}, handle)
            with self.assertRaisesRegex(RuntimeError, "no assistant responses"):
                build_store(pairs, root / "out.pkl")

    def test_receipt_binds_artifact_and_source_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pairs = root / "pairs.pkl"
            artifact = root / "response_fluency.pkl"
            receipt = root / "receipt.json"
            with pairs.open("wb") as handle:
                pickle.dump({"prompts": ["prompt"],
                             "responses": ["A complete assistant response."]}, handle)
            build_store(pairs, artifact)
            record = seal(artifact, pairs, receipt)
            self.assertEqual(record["status"], "sealed")
            self.assertTrue(receipt.is_file())
            with self.assertRaises(FileExistsError):
                seal(artifact, pairs, receipt)


if __name__ == "__main__":
    unittest.main()
