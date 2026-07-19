from pathlib import Path
import tempfile
import unittest

from omni.native_transformer import build_counted_transformer
from train_eval.build_position_conditioned_relation import (
    aggregate, atomic_json, marginal_counts, observations, sort_raw, write_raw,
)


class PositionConditionedRelationTests(unittest.TestCase):
    def test_one_canonical_relation_has_all_exact_marginals(self):
        prompts = ["red lamp red", "blue lamp"]
        responses = ["it glows", "it rests"]
        record = build_counted_transformer(prompts, responses)
        rows = []
        for prompt, response in zip(prompts, responses):
            rows.extend(observations(
                prompt, response, record["vocab"],
                record["bos_id"], record["eos_id"],
            ))
        held = marginal_counts(rows)

        self.assertEqual(sum(held["canonical"].values()), len(rows))
        self.assertEqual(sum(held["value"].values()), len(rows))
        self.assertEqual(sum(held["semantic2"].values()), len(rows))
        self.assertEqual(sum(held["semantic3"].values()), len(rows))
        # The repeated token occupies two distinct observed relative positions.
        red = record["vocab"]["red"]
        self.assertTrue(any(key[0] == 2 and key[1] == red
                            for key in held["canonical"]))
        self.assertTrue(any(key[0] == 0 and key[1] == red
                            for key in held["canonical"]))

    def test_external_build_preserves_every_fixture_observation(self):
        prompts = ["red lamp red", "blue lamp"]
        responses = ["it glows", "it rests"]
        record = build_counted_transformer(prompts, responses)
        pairs = {"prompts": prompts, "responses": responses}
        identity = {"fixture": "position-relation"}
        with tempfile.TemporaryDirectory() as held:
            root = Path(held)
            raw = root / "raw.tsv"
            state_path = root / "state.json"
            state = write_raw(pairs, record, raw, state_path, identity)
            sorted_path = root / "sorted.tsv"
            relation = root / "relation.tsv"
            sort_raw(raw, sorted_path, root / "tmp", "1M", 1)
            unique, total = aggregate(sorted_path, relation)

            expected = marginal_counts(
                row
                for prompt, response in zip(prompts, responses)
                for row in observations(
                    prompt, response, record["vocab"],
                    record["bos_id"], record["eos_id"],
                )
            )["canonical"]
            observed = {}
            for line in relation.read_text().splitlines():
                fields = tuple(map(int, line.split("\t")))
                observed[fields[:5]] = fields[5]
            self.assertEqual(total, state["observations"])
            self.assertEqual(unique, len(expected))
            self.assertEqual(observed, dict(expected))


if __name__ == "__main__":
    unittest.main()
