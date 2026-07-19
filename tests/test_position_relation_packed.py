import json
from pathlib import Path
import tempfile
import unittest

from omni.position_relation import PackedPositionRelation
from train_eval.pack_position_relation import pack_relation


class PackedPositionRelationTests(unittest.TestCase):
    def test_exact_prefix_marginals_and_pack_receipt(self):
        rows = [
            (0, 7, 2, 2, 9, 3),
            (0, 7, 2, 2, 10, 1),
            (0, 7, 2, 8, 9, 2),
            (0, 7, 4, 2, 11, 5),
            (1, 7, 2, 2, 12, 7),
        ]
        with tempfile.TemporaryDirectory() as held:
            root = Path(held)
            relation = root / "relation.tsv"
            relation.write_text("".join(
                "\t".join(map(str, row)) + "\n" for row in rows))
            import hashlib
            relation_hash = hashlib.sha256(relation.read_bytes()).hexdigest()
            relation_receipt = root / "relation_receipt.json"
            relation_receipt.write_text(json.dumps({
                "schema": "unison-position-conditioned-canonical-relation/v1",
                "status": "completed",
                "relation_bytes": relation.stat().st_size,
                "relation_sha256": relation_hash,
                "unique_canonical_entries": len(rows),
                "observations": sum(row[-1] for row in rows),
            }))
            packed = root / "relation.bin"
            receipt = root / "packed_receipt.json"
            result = pack_relation(relation, relation_receipt, packed, receipt)
            self.assertEqual(result["observations"], 18)
            with PackedPositionRelation(packed, receipt) as store:
                self.assertEqual(store.value_counts(0, 7), {9: 5, 10: 1, 11: 5})
                self.assertEqual(store.semantic2_counts(2, 0, 7), {9: 5, 10: 1})
                self.assertEqual(store.semantic3_counts(2, 2, 0, 7),
                                 {9: 3, 10: 1})
                self.assertEqual(store.semantic3_counts(8, 2, 0, 7), {9: 2})
                self.assertEqual(store.value_counts(1, 7), {12: 7})
                self.assertEqual(store.value_counts(2, 7), {})


if __name__ == "__main__":
    unittest.main()
