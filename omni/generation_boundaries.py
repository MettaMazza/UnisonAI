"""Executable Step 322–324 generation boundaries.

The canonical quantities live in :mod:`omni.core`.  This module exposes the
boundary operations so active generation paths reuse the same forced identities
instead of repeating numeric literals.
"""
from fractions import Fraction

from omni.core import (
    FOCUS_LOCK,
    INTEGRATION_DEPTH,
    REEXPRESS_MIN,
    SPREAD_LOCK,
    cascade_share,
    halt_violation,
)


COHERENCE_LOCK = FOCUS_LOCK
CASCADE_FACTOR = cascade_share(1)
CONTEXT_DEPTH = INTEGRATION_DEPTH
ADMISSION_LOCK = SPREAD_LOCK
BINDING_LOCK = SPREAD_LOCK
REEXPRESSION_MINIMUM = REEXPRESS_MIN


def _exact(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def minimal_share_prefix(ranked_shares, lock=ADMISSION_LOCK):
    """Return the smallest strongest-first prefix whose exact mass reaches lock.

    ``ranked_shares`` contains ``(object, share)`` pairs.  Shares must already be
    expressed against the One and ordered from strongest to weakest.  When the
    retained/addressable mass is smaller than the lock, all retained entries are
    returned; the boundary never invents missing mass.
    """
    entries = list(ranked_shares)
    lock = _exact(lock)
    if lock <= 0 or lock > 1:
        halt_violation(f"generation admission lock outside (0,1]: {lock}")
    exact = []
    previous = None
    for key, share in entries:
        share = _exact(share)
        if share < 0:
            halt_violation(f"negative generation share for {key!r}: {share}")
        if previous is not None and share > previous:
            halt_violation("generation shares are not strongest-first")
        previous = share
        exact.append((key, share))
    if sum((share for _, share in exact), Fraction(0)) < lock:
        return entries
    admitted = []
    mass = Fraction(0)
    for original, (_, share) in zip(entries, exact):
        admitted.append(original)
        mass += share
        if mass >= lock:
            break
    return admitted


def reaches_binding_lock(part, total):
    """Whether a counted part reaches the half-One binding lock."""
    if not isinstance(part, int) or not isinstance(total, int):
        halt_violation("binding counts must be exact integers")
    if part < 0 or total < 0 or part > total:
        halt_violation(f"invalid binding counts: part={part}, total={total}")
    if total == 0:
        return False
    return part * BINDING_LOCK.denominator >= total * BINDING_LOCK.numerator


def has_reexpression_support(count):
    """Fresh cross-variant composition requires the binary minimum support."""
    if not isinstance(count, int) or count < 0:
        halt_violation(f"invalid re-expression support count: {count!r}")
    return count >= REEXPRESSION_MINIMUM
