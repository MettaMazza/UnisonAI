from fractions import Fraction
import unittest

from omni import core
from omni.generation_boundaries import (
    ADMISSION_LOCK,
    BINDING_LOCK,
    CASCADE_FACTOR,
    COHERENCE_LOCK,
    CONTEXT_DEPTH,
    REEXPRESSION_MINIMUM,
    has_reexpression_support,
    minimal_share_prefix,
    reaches_binding_lock,
)


class GenerationBoundaryTests(unittest.TestCase):
    def test_canonical_identities(self):
        self.assertIs(COHERENCE_LOCK, core.FOCUS_LOCK)
        self.assertIs(ADMISSION_LOCK, core.SPREAD_LOCK)
        self.assertIs(BINDING_LOCK, core.SPREAD_LOCK)
        self.assertEqual(CASCADE_FACTOR, Fraction(1, 2))
        self.assertEqual(CONTEXT_DEPTH, 5)
        self.assertEqual(REEXPRESSION_MINIMUM, 2)

    def test_minimal_prefix(self):
        ranked = [("a", Fraction(2, 5)), ("b", Fraction(3, 10)),
                  ("c", Fraction(1, 5)), ("d", Fraction(1, 10))]
        self.assertEqual(minimal_share_prefix(ranked), ranked[:2])
        self.assertEqual(minimal_share_prefix([("a", Fraction(3, 5)),
                                               ("b", Fraction(2, 5))]),
                         [("a", Fraction(3, 5))])
        retained = [("a", Fraction(1, 4)), ("b", Fraction(1, 8))]
        self.assertEqual(minimal_share_prefix(retained), retained)

    def test_invalid_share_inputs_halt(self):
        with self.assertRaises(SystemExit):
            minimal_share_prefix([("a", Fraction(-1, 2))])
        with self.assertRaises(SystemExit):
            minimal_share_prefix([("a", Fraction(1, 4)), ("b", Fraction(1, 2))])

    def test_binding_and_reexpression_boundaries(self):
        self.assertTrue(reaches_binding_lock(1, 2))
        self.assertFalse(reaches_binding_lock(2, 5))
        self.assertFalse(reaches_binding_lock(0, 0))
        with self.assertRaises(SystemExit):
            reaches_binding_lock(3, 2)
        self.assertFalse(has_reexpression_support(1))
        self.assertTrue(has_reexpression_support(2))


if __name__ == "__main__":
    unittest.main()
