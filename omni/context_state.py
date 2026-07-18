"""THE INTEGRATED CONTEXT STATE — contextual_integration.ep made engineering.

What stacked attention layers buy with gradients, as the forced cascade over kin depth
(ernos-verified, 11/11): the context distribution is diffused through the counted coupling
graph for b+c = 5 rounds — round k carrying order-k relations at the fold weight 2^-k —
and the integrated state S = Σ_k 2^-k D_k is a partition of the One. Depth forced (the
covering depth), weights forced (successive halvings), normalization exact. The graph's
degree cap is an engineering dimension (marked), never a derivation input.

Ranking use only (floats order candidates; no served probability).
"""
from collections import Counter
from omni.word_engine import word_engine, tokenize, _content_words
from omni.generation_boundaries import (
    CASCADE_FACTOR,
    CONTEXT_DEPTH,
    minimal_share_prefix,
)

DEPTH = CONTEXT_DEPTH       # b + c — the covering depth (core lock, halt-verified at wake)
                            # the spread capacity — generation_selection_law.ep (FORCED, reused):
                       # a step spreads to the minimal strongest neighbour set whose shares
                       # complete the lock 1/2; the tail is suppressed. The former TOP_NEIGH
                       # engineering cap is RETIRED — the lock is the capacity.


_SPREAD_CACHE = {}


def _spread_of(w, coup):
    """Memoized unit-capacity spread of one word (the coupling graph is static within
    a process; re-sorting neighbours per call was measured at 0.6s/state)."""
    c = _SPREAD_CACHE.get(w)
    if c is None:
        nb = coup.get(w)
        if not nb:
            c = ()
        else:
            ranked = sorted(nb.items(), key=lambda kv: (-kv[1], kv[0]))
            z_all = float(sum(v for _, v in ranked)) or 1.0
            shares = [(n, v / z_all) for n, v in ranked]
            admitted = minimal_share_prefix(shares)
            admitted_names = {n for n, _ in admitted}
            spread = [(n, v) for n, v in ranked if n in admitted_names]
            z = float(sum(v for _, v in spread)) or 1.0
            c = tuple((n, v / z) for n, v in spread)
        _SPREAD_CACHE[w] = c
    return c


def integrated_state(text, history=None):
    """S = Σ_{k=1..5} 2^-k D_k. D_1 = the live context distribution (recency 2^-age);
    D_{k+1} = one diffusion round of D_k through the coupling graph (each word's mass
    split over its top neighbours proportionally to counted coupling)."""
    coup = word_engine._load_coupling() or {}

    d = Counter()
    for w in _content_words(tokenize(str(text).lower())):
        d[w] += 1.0
    for age, (_, t) in enumerate(reversed(list(history or [])[-4:]), start=1):
        for w in _content_words(tokenize(str(t).lower())):
            d[w] += 2.0 ** -age
    tot = sum(d.values())
    if not tot:
        return Counter()
    for w in d:
        d[w] /= tot

    S = Counter()
    weight = float(CASCADE_FACTOR)     # step-1 weight = the fold factor 1/2
    for k in range(DEPTH):
        for w, m in d.items():
            S[w] += weight * m
        if k == DEPTH - 1:
            # THE FLOOR (forced: the cascade closes to the One with the floor exactly —
            # the deepest level carries its halving twice). Implementation previously
            # stopped at 31/32: a measured law deviation, caught by the mass check.
            for w, m in d.items():
                S[w] += weight * m
            break
        # one diffusion round: order-(k+1) relations
        nxt = Counter()
        for w, m in d.items():
            spread = _spread_of(w, coup)
            if not spread:
                nxt[w] += m            # no neighbours: mass stays (self-loop)
                continue
            # UNIT-CAPACITY SPREAD (forced), memoized: the minimal strongest prefix
            # whose cumulative share reaches the lock 1/2; the tail is suppressed
            for n, v in spread:        # v is already the normalized share (memoized)
                nxt[n] += m * v
        d = nxt
        weight *= float(CASCADE_FACTOR)  # one fold deeper per round
    return S
