from fractions import Fraction
import unittest

from omni.core import INTEGRATION_DEPTH
from omni.native_transformer import build_counted_transformer
from omni.prompt_context import (PositionAddress, aggregate_keys,
                                 contextualize, decoder_context, positional_head,
                                 project_tokens,
                                 _distribution, _mix_distributions,
                                 _weighted_mix)


class PromptContextTests(unittest.TestCase):
    def setUp(self):
        self.record = build_counted_transformer(
            [
                "red garden grows",
                "blue stars shine",
                "red stars glow",
                "garden plants grow",
            ],
            [
                "plants need light",
                "stars emit light",
                "stars emit color",
                "plants need water",
            ],
        )

    def addresses(self, words):
        return [PositionAddress(
            token_id=self.record["vocab"][word],
            turn_age=0,
            within_turn=index,
            sequence_index=index,
        ) for index, word in enumerate(words)]

    def test_repeated_tokens_keep_distinct_addresses(self):
        addresses = self.addresses(["red", "stars", "red"])
        self.assertEqual(addresses[0].token_id, addresses[2].token_id)
        self.assertNotEqual(addresses[0], addresses[2])
        positions = contextualize(addresses, self.record["profiles"])
        self.assertEqual(len(positions), 3)
        self.assertEqual(
            [position.address.within_turn for position in positions],
            [0, 1, 2],
        )
        context = decoder_context(positions)
        self.assertIsNotNone(context)
        self.assertEqual(len(context.positions), 3)
        self.assertIn(0, context.position_shares)
        self.assertIn(2, context.position_shares)
        self.assertEqual(sum(context.position_shares.values(), Fraction(0)), 1)
        self.assertEqual(sum(context.token_keys.values(), Fraction(0)), 1)

    def test_relative_position_head_closes_exactly(self):
        addresses = self.addresses(["red", "stars", "glow"])
        for query_index in range(len(addresses)):
            head = positional_head(addresses, query_index)
            self.assertEqual(sum(head.values(), Fraction(0)), 1)
            self.assertGreater(head[query_index], head[(query_index + 1) % 3])

    def test_five_layer_states_and_projection_close(self):
        addresses = self.addresses(["red", "stars", "glow"])
        positions = contextualize(addresses, self.record["profiles"])
        self.assertEqual(INTEGRATION_DEPTH, 5)
        for position in positions:
            self.assertEqual(sum(position.state.values(), Fraction(0)), 1)
        keys = aggregate_keys(positions)
        self.assertEqual(sum(keys.values(), Fraction(0)), 1)
        self.assertTrue(all(
            len(position.state) <= len(addresses) for position in positions
        ))

    def test_position_order_changes_contextual_projection(self):
        forward = contextualize(
            self.addresses(["red", "stars", "garden"]),
            self.record["profiles"],
        )
        reordered = contextualize(
            self.addresses(["red", "garden", "stars"]),
            self.record["profiles"],
        )
        garden = self.record["vocab"]["garden"]
        forward_tokens = project_tokens(forward[0].state,
                                        [position.address for position in forward])
        reordered_tokens = project_tokens(
            reordered[0].state,
            [position.address for position in reordered],
        )
        self.assertNotEqual(forward_tokens.get(garden),
                            reordered_tokens.get(garden))

    def test_older_turn_mass_is_dyadically_aged(self):
        red = self.record["vocab"]["red"]
        stars = self.record["vocab"]["stars"]
        addresses = [
            PositionAddress(red, 1, 0, 0),
            PositionAddress(stars, 0, 0, 1),
        ]
        head = positional_head(addresses, 1)
        self.assertEqual(head[0], Fraction(1, 4))
        self.assertEqual(head[1], Fraction(3, 4))

    def test_shared_denominator_mix_equals_fraction_reference(self):
        left = {0: Fraction(2, 3), 1: Fraction(1, 3)}
        right = {0: Fraction(1, 5), 2: Fraction(4, 5)}
        weighted = [(Fraction(3, 7), left), (Fraction(4, 7), right)]
        reference = _weighted_mix(weighted, "test reference")
        integer = _mix_distributions([
            (weight, _distribution(source, "test source"))
            for weight, source in weighted
        ], "test integer").fractions()
        self.assertEqual(integer, reference)


if __name__ == "__main__":
    unittest.main()
