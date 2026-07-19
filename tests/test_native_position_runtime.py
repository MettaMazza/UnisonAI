from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from omni.native_transformer import CountedCausalTransformer, build_counted_transformer
from omni.position_relation import RECORD, SCHEMA
from train_eval.build_position_conditioned_relation import observations


class NativePositionRuntimeTests(unittest.TestCase):
    def test_sealed_position_relation_changes_the_exact_runtime_scores(self):
        prompts = ["alpha beta", "alpha beta"]
        responses = ["one.", "two."]
        record = build_counted_transformer(prompts, responses)
        canonical = Counter(
            row
            for prompt, response in zip(prompts, responses)
            for row in observations(
                prompt, response, record["vocab"],
                record["bos_id"], record["eos_id"])
        )
        with tempfile.TemporaryDirectory() as held:
            root = Path(held)
            relation_path = root / "position.bin"
            relation_path.write_bytes(b"".join(
                RECORD.pack(*row, count)
                for row, count in sorted(canonical.items())
            ))
            receipt_path = root / "position.json"
            receipt_path.write_text(json.dumps({
                "schema": SCHEMA + "/receipt",
                "status": "sealed",
                "packed_bytes": relation_path.stat().st_size,
                "unique_canonical_entries": len(canonical),
                "packed_sha256": "fixture",
            }))
            reward = root / "reward.pkl"
            baseline = CountedCausalTransformer(
                record=record, reward_path=str(reward))
            positioned = CountedCausalTransformer(
                record=record, reward_path=str(reward),
                position_path=str(relation_path),
                position_receipt_path=str(receipt_path))
            context = positioned._contextual_decoder_state("alpha beta")
            bos = record["bos_id"]
            baseline_scores = baseline._integer_residual_scores(
                bos, bos, context.token_keys, decoder_context=context)
            cache = {}
            positioned_scores = positioned._integer_residual_scores(
                bos, bos, context.token_keys, decoder_context=context,
                position_cache=cache)
            common = set(baseline_scores) | set(positioned_scores)
            baseline_gcd = math.gcd(*baseline_scores.values())
            positioned_gcd = math.gcd(*positioned_scores.values())
            self.assertNotEqual(
                {key: baseline_scores.get(key, 0) // baseline_gcd
                 for key in common},
                {key: positioned_scores.get(key, 0) // positioned_gcd
                 for key in common},
            )
            self.assertTrue(cache)
            self.assertEqual(
                {address[0] for address in cache},
                {"value", "semantic2", "semantic3"},
            )
            self.assertTrue(all(count >= 0 for rows in cache.values()
                                for count in rows.values()))
            positioned.close()


if __name__ == "__main__":
    unittest.main()
