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
from omni.core import INTEGRATION_DEPTH, SPREAD_LOCK

DEPTH = INTEGRATION_DEPTH   # b + c — the covering depth (core lock, halt-verified at wake)
LOCK = float(SPREAD_LOCK)   # the spread capacity — generation_selection_law.ep (FORCED, reused):
                       # a step spreads to the minimal strongest neighbour set whose shares
                       # complete the lock 1/2; the tail is suppressed. The former TOP_NEIGH
                       # engineering cap is RETIRED — the lock is the capacity.


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
    weight = 0.5                       # step-1 weight = the fold factor 1/2
    for k in range(DEPTH):
        for w, m in d.items():
            S[w] += weight * m
        # one diffusion round: order-(k+1) relations
        nxt = Counter()
        for w, m in d.items():
            nb = coup.get(w)
            if not nb:
                nxt[w] += m            # no neighbours: mass stays (self-loop)
                continue
            # UNIT-CAPACITY SPREAD (forced): the minimal strongest prefix whose
            # cumulative share reaches the lock 1/2; the tail is suppressed
            ranked = sorted(nb.items(), key=lambda kv: -kv[1])
            z_all = float(sum(v for _, v in ranked)) or 1.0
            spread, acc = [], 0.0
            for n, v in ranked:
                spread.append((n, v))
                acc += v / z_all
                if acc >= LOCK:
                    break
            z = float(sum(v for _, v in spread)) or 1.0
            for n, v in spread:
                nxt[n] += m * (v / z)
        d = nxt
        weight *= 0.5                  # one fold deeper per round
    return S
