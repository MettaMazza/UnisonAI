from fractions import Fraction
import asyncio
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omni.native_transformer import (
    CountedCausalTransformer,
    SCHEMA,
    ROLE_POLICY,
    build_counted_transformer,
)
from omni.packed_rows import load_packed_record, pack_record, verify_packed_files


class CountedCausalTransformerTests(unittest.TestCase):
    def setUp(self):
        prompts = [
            "Tell me about gardening",
            "What do you enjoy about gardening?",
            "Tell me about astronomy",
        ]
        responses = [
            "Gardening grows patience.",
            "Gardening grows food.",
            "Astronomy studies stars.",
        ]
        self.record = build_counted_transformer(prompts, responses)
        self.temp = tempfile.TemporaryDirectory()
        self.model = CountedCausalTransformer(
            record=self.record,
            reward_path=os.path.join(self.temp.name, "reward.pkl"),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_schema_and_role_bound_training(self):
        self.assertEqual(self.record["schema"], SCHEMA)
        self.assertEqual(self.record["role_policy"], ROLE_POLICY)
        self.assertEqual(self.record["response_count"], 3)
        self.assertGreater(self.record["token_count"], 3)

    def test_attention_training_retains_complete_prompt_tokens(self):
        record = build_counted_transformer(
            ["Why, why gardening?"], ["Because it grows."])
        vocab = record["vocab"]
        bos = record["bos_id"]
        # Repetition is an observed attention occurrence, and function words
        # plus punctuation are not stripped before Q/K/V training.
        self.assertEqual(record["qk"][(bos, vocab["why"])], 2)
        self.assertIn((bos, vocab[","]), record["qk"])
        self.assertIn((bos, vocab["?"]), record["qk"])

    def test_counted_query_key_head_changes_active_key_weights(self):
        record = build_counted_transformer(
            ["alpha beta", "alpha beta"], ["one.", "two."])
        vocab = record["vocab"]
        bos = record["bos_id"]
        # Supply equal structural keys and an alternate query whose counted
        # compatibility is deliberately asymmetric.
        record["qk"][(bos, vocab["alpha"])] = 1
        record["qk"][(bos, vocab["beta"])] = 3
        model = CountedCausalTransformer(record=record)
        weights = model._attention_key_weights(
            bos, {vocab["alpha"]: Fraction(1), vocab["beta"]: Fraction(1)})
        self.assertGreater(weights[vocab["beta"]], weights[vocab["alpha"]])

    def test_lazy_exact_argmax_matches_full_lm_head(self):
        for prompt in ("gardening", "astronomy"):
            keys = self.model._context_keys(prompt)
            prev_id = last_id = self.record["bos_id"]
            for _ in range(8):
                distribution = self.model.next_distribution(prev_id, last_id, keys)
                expected = min(
                    distribution,
                    key=lambda token_id: (-distribution[token_id], token_id))
                actual = self.model.next_token_id(prev_id, last_id, keys)
                self.assertEqual(actual, expected)
                if actual == self.record["eos_id"]:
                    break
                prev_id, last_id = last_id, actual

    def test_bounded_prepared_value_rows_preserve_exact_argmax(self):
        keys = self.model._context_keys("gardening")
        record = self.record
        prepared = {}
        for key_id in keys:
            counts = record["values"].get(key_id)
            if counts:
                prepared[key_id] = (counts, sum(counts.values()))
        bos = record["bos_id"]
        self.assertEqual(
            self.model.next_token_id(bos, bos, keys),
            self.model.next_token_id(
                bos, bos, keys, prepared_values=prepared),
        )

    def test_packed_serving_rows_and_generation_are_exactly_equivalent(self):
        packed_path = Path(self.temp.name) / "packed"
        manifest = pack_record(
            self.record, packed_path, artifact_sha256="a" * 64)
        self.assertEqual(manifest["schema"], "unison-packed-exact-rows/v1")
        self.assertEqual(verify_packed_files(packed_path)["status"], "verified")
        packed = load_packed_record(packed_path)
        for namespace in (
                "profiles", "qk", "values", "semantic_ffn",
                "semantic_ffn3", "ffn2", "ffn3"):
            original = self.record[namespace]
            packed_table = packed[namespace]
            self.assertEqual(len(packed_table), len(original))
            for key, value in original.items():
                self.assertEqual(packed_table.get(key), value)
        packed_model = CountedCausalTransformer(
            record=packed,
            reward_path=os.path.join(self.temp.name, "packed_reward.pkl"),
        )
        for prompt in ("gardening", "astronomy"):
            source_keys = self.model._context_keys(prompt)
            packed_keys = packed_model._context_keys(prompt)
            self.assertEqual(source_keys, packed_keys)
            bos = self.record["bos_id"]
            self.assertEqual(
                self.model.next_distribution(bos, bos, source_keys),
                packed_model.next_distribution(bos, bos, packed_keys),
            )
            self.assertEqual(self.model.generate(prompt), packed_model.generate(prompt))

    def test_role_provenance_mismatch_halts(self):
        broken = dict(self.record)
        broken["role_policy"] = "mixed-role-output"
        with self.assertRaises(SystemExit):
            CountedCausalTransformer(record=broken)._store()

    def test_default_packed_receipt_drift_halts(self):
        from omni import native_transformer as native_module
        packed = Path(self.temp.name) / "default_packed"
        pack_record(self.record, packed, artifact_sha256="b" * 64)
        receipt = Path(self.temp.name) / "receipt.json"
        receipt.write_text(json.dumps({
            "schema": "unison-packed-native-transformer-receipt/v1",
            "status": "sealed",
            "manifest_sha256": "0" * 64,
        }))
        model = CountedCausalTransformer(packed_path=str(packed))
        with patch.object(native_module, "DEFAULT_PACKED", str(packed)), \
                patch.object(native_module, "DEFAULT_PACKED_RECEIPT", str(receipt)):
            with self.assertRaises(SystemExit):
                model._store()

    def test_default_packed_missing_receipt_halts(self):
        from omni import native_transformer as native_module
        packed = Path(self.temp.name) / "default_packed"
        pack_record(self.record, packed, artifact_sha256="b" * 64)
        missing = Path(self.temp.name) / "missing-receipt.json"
        model = CountedCausalTransformer(packed_path=str(packed))
        with patch.object(native_module, "DEFAULT_PACKED", str(packed)), \
                patch.object(native_module, "DEFAULT_PACKED_RECEIPT", str(missing)):
            with self.assertRaises(SystemExit):
                model._store()

    def test_default_packed_runtime_source_drift_halts(self):
        from omni import native_transformer as native_module
        packed = Path(self.temp.name) / "default_packed"
        pack_record(self.record, packed, artifact_sha256="b" * 64)
        manifest = packed / "manifest.json"
        receipt = Path(self.temp.name) / "receipt.json"
        receipt.write_text(json.dumps({
            "schema": "unison-packed-native-transformer-receipt/v1",
            "status": "sealed",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sources": {
                "omni/native_transformer.py": "0" * 64,
                "omni/packed_rows.py": "0" * 64,
            },
        }))
        model = CountedCausalTransformer(packed_path=str(packed))
        with patch.object(native_module, "DEFAULT_PACKED", str(packed)), \
                patch.object(native_module, "DEFAULT_PACKED_RECEIPT", str(receipt)):
            with self.assertRaises(SystemExit):
                model._store()

    def test_transformer_organs_close_exactly(self):
        keys = self.model._context_keys("gardening")
        bos = self.record["bos_id"]
        attention = self.model._attention(bos, bos, keys)
        ffn = self.model._ffn(bos, bos)
        head = self.model.next_distribution(bos, bos, keys)
        self.assertEqual(sum(attention.values(), Fraction(0)), 1)
        self.assertEqual(sum(ffn.values(), Fraction(0)), 1)
        self.assertEqual(sum(head.values(), Fraction(0)), 1)

    def test_prompt_attention_changes_the_causal_distribution(self):
        bos = self.record["bos_id"]
        garden = self.model.next_distribution(
            bos, bos, self.model._context_keys("gardening"))
        stars = self.model.next_distribution(
            bos, bos, self.model._context_keys("astronomy"))
        astronomy = self.record["vocab"]["astronomy"]
        self.assertGreater(stars.get(astronomy, 0), garden.get(astronomy, 0))
        self.assertNotEqual(garden, stars)
        self.assertTrue(self.model.generate("gardening").startswith("Gardening"))
        self.assertTrue(self.model.generate("astronomy").startswith("Astronomy"))

    def test_history_has_exact_dyadic_position_share(self):
        keys = self.model._positional_keys(
            "gardening", history=[("user", "astronomy")])
        vocab = self.record["vocab"]
        self.assertEqual(keys[vocab["gardening"]], Fraction(1))
        self.assertEqual(keys[vocab["astronomy"]], Fraction(1, 2))

    def test_contextual_position_route_closes_and_preserves_order(self):
        forward = self.model._contextual_keys("red gardening astronomy")
        reordered = self.model._contextual_keys("red astronomy gardening")
        self.assertEqual(sum(forward.values(), Fraction(0)), 1)
        self.assertEqual(sum(reordered.values(), Fraction(0)), 1)
        self.assertNotEqual(forward, reordered)

    def test_contextual_positions_reach_value_and_semantic_boundary_exactly(self):
        context = self.model._contextual_decoder_state(
            "gardening astronomy gardening")
        bos = self.record["bos_id"]
        addressed = self.model._attention_key_weights(bos, context.token_keys)
        position_sources = self.model._decoder_position_sources(
            addressed, context)
        sources = self.model._decoder_value_sources(addressed, context)
        self.assertEqual(
            {position_index for position_index, _, _ in position_sources},
            {0, 1, 2},
        )
        self.assertEqual(
            [(weight, key_id) for _, weight, key_id in position_sources],
            sources,
        )
        self.assertEqual(len(sources), 3)
        self.assertEqual(
            sum((weight for weight, _ in sources), Fraction(0)),
            sum(addressed.values(), Fraction(0)),
        )
        token_scores = self.model._integer_residual_scores(
            bos, bos, context.token_keys)
        position_scores = self.model._integer_residual_scores(
            bos, bos, context.token_keys, decoder_context=context)
        common = set(token_scores) | set(position_scores)
        left = {key: token_scores.get(key, 0) for key in common}
        right = {key: position_scores.get(key, 0) for key in common}
        left_gcd = math.gcd(*left.values())
        right_gcd = math.gcd(*right.values())
        self.assertEqual(
            {key: value // left_gcd for key, value in left.items()},
            {key: value // right_gcd for key, value in right.items()},
        )

    def test_reward_conditioning_is_counted_and_persistent(self):
        before = self.model.generate("gardening")
        self.model.mark_feedback("gardening", before, good=True)
        reloaded = CountedCausalTransformer(
            record=self.record,
            reward_path=self.model.reward_path,
        )
        self.assertTrue(reloaded._rewards())
        self.assertEqual(len(reloaded._reward_events), 1)
        self.assertTrue(reloaded._reward_events[0]["good"])
        self.assertEqual(len(reloaded._reward_events[0]["prompt_sha256"]), 64)
        self.assertEqual(reloaded.generate("gardening"), before)

    def test_live_discord_surface_uses_native_route_before_rag(self):
        from omni.discord_bot import SFTDiscordClient
        client = object.__new__(SFTDiscordClient)
        session = SimpleNamespace(history_log=[])
        trace = []
        with patch("omni.discord_bot.native_transformer", self.model), \
                patch("omni.discord_bot.pair_retrieval.reply",
                      side_effect=AssertionError("RAG should not run")):
            reply = asyncio.run(client._generate_turn_surface(
                "gardening", [], session, "fixture", object(), trace=trace))
        self.assertTrue(reply.startswith("Gardening"), reply)
        stages = trace[0]["segments"][0]["stages"]
        self.assertEqual(stages[0]["stage"], "native_causal_transformer")
        self.assertEqual(client._last_native_feedback, [("gardening", reply)])
        self.assertFalse(client._last_rag_feedback)

    def test_rag_surface_cannot_enter_native_reward_provenance(self):
        from omni.discord_bot import SFTDiscordClient
        client = object.__new__(SFTDiscordClient)
        session = SimpleNamespace(history_log=[])
        unavailable = SimpleNamespace(available=lambda: False)
        with patch("omni.discord_bot.native_transformer", unavailable), \
                patch("omni.discord_bot.pair_retrieval.reply",
                      return_value="A RAG response."):
            reply = asyncio.run(client._generate_turn_surface(
                "gardening", [], session, "fixture", object()))
        self.assertEqual(reply, "A RAG response.")
        self.assertEqual(client._last_native_feedback, [])
        self.assertTrue(client._last_rag_feedback)


if __name__ == "__main__":
    unittest.main()
