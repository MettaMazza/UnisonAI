from fractions import Fraction
import asyncio
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omni.native_transformer import (
    CountedCausalTransformer,
    SCHEMA,
    ROLE_POLICY,
    build_counted_transformer,
)


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

    def test_role_provenance_mismatch_halts(self):
        broken = dict(self.record)
        broken["role_policy"] = "mixed-role-output"
        with self.assertRaises(SystemExit):
            CountedCausalTransformer(record=broken)._store()

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
