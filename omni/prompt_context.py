"""Exact factorised kernel for Unison's stacked prompt self-attention organ.

This is the production form of the one-to-one transformer port. It
keeps positions distinct, performs established query/key/value attention,
identity residuals, exact normalization, and a counted feed-forward transform
for every layer. The layer depth and integration shares come directly from
``constants/contextual_integration.ep`` through ``omni.core``.

The kernel factorises sparse rational states over shared count rows while tests
compare its mixture against the direct Fraction form. It does not alter the
complete computation with a cap or truncation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Mapping, Sequence

from omni.core import (GEN_B, INTEGRATION_DEPTH, cascade_share,
                       halt_violation)


@dataclass(frozen=True)
class PositionAddress:
    """One token occurrence, not an aggregated token identity."""

    token_id: int
    turn_age: int
    within_turn: int
    sequence_index: int


@dataclass(frozen=True)
class ContextualPosition:
    address: PositionAddress
    # Exact hidden-state shares over the response-local position basis. Keeping
    # the latent basis factorised is representation-equivalent to carrying the
    # same position value vectors through dense vocabulary coordinates.
    state: Mapping[int, Fraction]


@dataclass(frozen=True)
class DecoderContext:
    """Final causal prompt state before decoder value/FFN projection."""

    positions: tuple[PositionAddress, ...]
    position_shares: Mapping[int, Fraction]
    token_keys: Mapping[int, Fraction]


@dataclass(frozen=True)
class _Distribution:
    """One exact distribution as integer mass over a shared denominator."""

    counts: Mapping[int, int]
    total: int

    def fractions(self) -> dict[int, Fraction]:
        return {key: Fraction(value, self.total)
                for key, value in self.counts.items() if value > 0}


def _integer_distribution(counts: Mapping[int, int], organ: str) -> _Distribution:
    positive = {key: int(value) for key, value in counts.items() if value > 0}
    if not positive:
        halt_violation(f"prompt contextual {organ} has no positive integer mass")
    divisor = 0
    for value in positive.values():
        divisor = math.gcd(divisor, value)
    if divisor > 1:
        positive = {key: value // divisor for key, value in positive.items()}
    return _Distribution(positive, sum(positive.values()))


def _distribution(values: Mapping[int, Fraction], organ: str) -> _Distribution:
    normalized = _normalize(values, organ)
    common = math.lcm(*(value.denominator for value in normalized.values()))
    return _integer_distribution({
        key: value.numerator * (common // value.denominator)
        for key, value in normalized.items()
    }, organ)


def _cascade_distribution(scores: Mapping[int, Fraction], organ: str,
                          floor_key: int) -> _Distribution:
    """Rank an exact selection product onto the complete dyadic cascade.

    ``partition_localization.ep`` supplies ranks ``1/2^r`` and the closing
    remainder equal to the final rank share. That remainder stays on the
    identity address, matching the contextual-integration floor and residual.
    Equal products are indistinguishable, so their occupied rank mass is
    divided exactly rather than broken by an authored left/right preference.
    """
    positive = {key: value for key, value in scores.items() if value > 0}
    if not positive:
        halt_violation(f"prompt contextual {organ} has no positive product")
    by_score: defaultdict[Fraction, list[int]] = defaultdict(list)
    for key, value in positive.items():
        by_score[value].append(key)
    width = len(positive)
    rank = 1
    shares = {}
    for score in sorted(by_score, reverse=True):
        members = by_score[score]
        final_rank = rank + len(members) - 1
        group_mass = sum(
            (cascade_share(held_rank)
             for held_rank in range(rank, final_rank + 1)),
            Fraction(0),
        )
        member_share = group_mass / len(members)
        for key in members:
            shares[key] = member_share
        rank = final_rank + 1
    # The closing remainder is the no-zero identity floor. Keeping it on the
    # query/self address is the position-space form of the residual path.
    shares[floor_key] = shares.get(floor_key, Fraction(0)) + cascade_share(width)
    return _distribution(shares, organ)


def _mix_distributions(weighted_sources, organ: str) -> _Distribution:
    """Exact normalized mixture while every source keeps one denominator."""
    groups: dict[int, defaultdict[int, int]] = {}
    for weight, source in weighted_sources:
        if weight <= 0:
            continue
        denominator = weight.denominator * source.total
        target = groups.setdefault(denominator, defaultdict(int))
        for key, count in source.counts.items():
            target[key] += weight.numerator * count
    if not groups:
        halt_violation(f"prompt contextual {organ} has no positive mixture")
    common = math.lcm(*groups.keys())
    mass: defaultdict[int, int] = defaultdict(int)
    for denominator, counts in groups.items():
        scale = common // denominator
        for key, count in counts.items():
            mass[key] += count * scale
    return _integer_distribution(mass, organ)


def _normalize(values: Mapping[int, Fraction], organ: str) -> dict[int, Fraction]:
    positive = {key: value for key, value in values.items() if value > 0}
    if not positive:
        halt_violation(f"prompt contextual {organ} has no positive mass")
    common = math.lcm(*(value.denominator for value in positive.values()))
    integer_mass = {
        key: value.numerator * (common // value.denominator)
        for key, value in positive.items()
    }
    total = sum(integer_mass.values())
    if total <= 0:
        halt_violation(f"prompt contextual {organ} integer mass did not close")
    result = {key: Fraction(value, total)
              for key, value in integer_mass.items() if value > 0}
    return result


def _weighted_mix(weighted_sources, organ: str) -> dict[int, Fraction]:
    """Normalize an exact linear mixture with grouped integer accumulation."""
    groups: dict[int, defaultdict[int, int]] = {}
    for weight, source in weighted_sources:
        if weight <= 0:
            continue
        for key, value in source.items():
            if value <= 0:
                continue
            numerator = weight.numerator * value.numerator
            denominator = weight.denominator * value.denominator
            groups.setdefault(denominator, defaultdict(int))[key] += numerator
    if not groups:
        halt_violation(f"prompt contextual {organ} has no positive mixture")
    common = math.lcm(*groups.keys())
    integer_mass: defaultdict[int, int] = defaultdict(int)
    for denominator, numerators in groups.items():
        scale = common // denominator
        for key, numerator in numerators.items():
            integer_mass[key] += numerator * scale
    total = sum(integer_mass.values())
    if total <= 0:
        halt_violation(f"prompt contextual {organ} integer mixture did not close")
    result = {key: Fraction(value, total)
              for key, value in integer_mass.items() if value > 0}
    return result


def _counted_dot(left: Mapping[int, int], left_fallback: int,
                 right: Mapping[int, int], right_fallback: int) -> Fraction:
    """Dot two unit-normalized counted embedding rows using integers first."""
    left = {key: value for key, value in left.items() if value > 0}
    right = {key: value for key, value in right.items() if value > 0}
    if not left:
        left = {left_fallback: 1}
    if not right:
        right = {right_fallback: 1}
    left_total = sum(left.values())
    right_total = sum(right.values())
    if len(left) > len(right):
        left, right = right, left
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    return Fraction(numerator, left_total * right_total)


def positional_head(addresses: Sequence[PositionAddress],
                    query_index: int) -> dict[int, Fraction]:
    """Exact relative-position head with no learned positional parameter.

    Distance ``d`` receives the forced cascade share ``1/2^(d+1)``. When two
    positions have the same distance, that distance share is divided exactly
    by its observed multiplicity, so no left/right tie preference is invented.
    Older turns carry the already-forced dyadic turn-age share.
    """
    query = addresses[query_index]
    multiplicity: defaultdict[int, int] = defaultdict(int)
    for source in addresses:
        multiplicity[abs(source.sequence_index - query.sequence_index)] += 1
    raw = {}
    for source_index, source in enumerate(addresses):
        distance = abs(source.sequence_index - query.sequence_index)
        distance_share = cascade_share(distance + 1)
        tied_share = distance_share / multiplicity[distance]
        turn_share = Fraction(1, GEN_B ** source.turn_age)
        raw[source_index] = tied_share * turn_share
    return _cascade_distribution(
        raw, "relative-position head", query_index).fractions()


def _bilinear(left: _Distribution, right: _Distribution,
              gram: Sequence[Sequence[Fraction]]) -> Fraction:
    """Exact Q/K product in the counted-embedding Gram relation."""
    groups: defaultdict[int, int] = defaultdict(int)
    for left_position, left_count in left.counts.items():
        for right_position, right_count in right.counts.items():
            relation = gram[left_position][right_position]
            if relation <= 0:
                continue
            numerator = left_count * right_count * relation.numerator
            denominator = relation.denominator
            groups[denominator] += numerator
    if not groups:
        return Fraction(0)
    common = math.lcm(*groups.keys())
    numerator = sum(value * (common // denominator)
                    for denominator, value in groups.items())
    return Fraction(numerator, left.total * right.total * common)


def _attention_head(states: Sequence[_Distribution],
                    positions: Mapping[int, Fraction], query_index: int,
                    gram: Sequence[Sequence[Fraction]]) -> _Distribution | None:
    raw = {
        source_index: positional_share
        * _bilinear(states[query_index], states[source_index], gram)
        for source_index, positional_share in positions.items()
    }
    raw = {key: value for key, value in raw.items() if value > 0}
    return _cascade_distribution(
        raw, "query/key head", query_index) if raw else None


def _attend(states: Sequence[_Distribution],
            addresses: Sequence[PositionAddress],
            query_index: int,
            gram: Sequence[Sequence[Fraction]]) -> _Distribution:
    structural_shares = positional_head(addresses, query_index)
    structural = _distribution(structural_shares, "relative-position head")
    association = _attention_head(
        states, structural_shares, query_index, gram)
    heads = (structural, association) if association is not None else (structural,)
    source_weights = _mix_distributions(
        ((Fraction(1), head) for head in heads), "attention heads")
    return _mix_distributions(
        ((Fraction(count, source_weights.total), states[source_index])
         for source_index, count in source_weights.counts.items()),
        "attention value",
    )


def _feed_forward(state: _Distribution,
                  relation_rows: Sequence[_Distribution]) -> _Distribution:
    """Counted prompt FFN over the factorised position value basis."""
    return _mix_distributions((
        (Fraction(count, state.total), relation_rows[position_index])
        for position_index, count in state.counts.items()
    ), "feed-forward")


def contextualize(addresses: Sequence[PositionAddress],
                  profiles: Mapping[int, Mapping[int, int]]) \
        -> list[ContextualPosition]:
    """Execute the five exact prompt-attention blocks and forced integration."""
    if not addresses:
        return []
    profile_rows = [profiles.get(address.token_id, {}) for address in addresses]
    gram = [
        [
            _counted_dot(
                profile_rows[left_index], left.token_id,
                profile_rows[right_index], right.token_id,
            )
            for right_index, right in enumerate(addresses)
        ]
        for left_index, left in enumerate(addresses)
    ]
    relation_rows = []
    for position_index, row in enumerate(gram):
        positive = {index: value for index, value in enumerate(row) if value > 0}
        if not positive:
            positive = {position_index: Fraction(1)}
        relation_rows.append(_cascade_distribution(
            positive, "counted FFN relation", position_index))

    # Values live on a distinct response-local position basis. The counted
    # embedding Gram relation above supplies Q/K and FFN operations without
    # expanding the hidden state into the complete vocabulary at every layer.
    initial = [_Distribution({position_index: 1}, 1)
               for position_index in range(len(addresses))]
    states = initial
    layers: list[list[_Distribution]] = []
    for _ in range(INTEGRATION_DEPTH):
        next_states = []
        for query_index, state in enumerate(states):
            attended = _attend(states, addresses, query_index, gram)
            residual_state = _mix_distributions(
                ((Fraction(1), state), (Fraction(1), attended)),
                "attention residual",
            )
            transformed = _feed_forward(residual_state, relation_rows)
            residual_state = _mix_distributions(
                ((Fraction(1), residual_state),
                 (Fraction(1), transformed)),
                "FFN residual",
            )
            next_states.append(residual_state)
        layers.append(next_states)
        states = next_states

    floor = Fraction(1, GEN_B ** INTEGRATION_DEPTH)
    weights = [cascade_share(layer) for layer in range(1, INTEGRATION_DEPTH + 1)]
    if sum(weights, floor) != 1:
        halt_violation("prompt contextual integration weights do not close")
    contextual = []
    for position_index, address in enumerate(addresses):
        integrated = _mix_distributions(
            [(floor, initial[position_index])] + [
                (weight, layer[position_index])
                for weight, layer in zip(weights, layers)
            ],
            "five-layer integration",
        )
        contextual.append(ContextualPosition(
            address=address,
            state=integrated.fractions(),
        ))
    return contextual


def decoder_context(positions: Iterable[ContextualPosition]) -> DecoderContext | None:
    """Preserve final-position hidden mass in position and token coordinates.

    A causal language model predicts from the final causally available hidden
    position. Summing every position would undo positional context and return a
    bag of tokens. The decoder therefore retains that final state over distinct
    prompt positions while also exposing its exact token projection for Q/K.
    """
    positions = list(positions)
    if not positions:
        return None
    final = max(positions, key=lambda position: position.address.sequence_index)
    position_shares = _normalize(
        final.state, "final-position decoder state")
    output: defaultdict[int, Fraction] = defaultdict(Fraction)
    for source_index, share in position_shares.items():
        token_id = positions[source_index].address.token_id
        output[token_id] += share
    return DecoderContext(
        positions=tuple(position.address for position in positions),
        position_shares=position_shares,
        token_keys=_normalize(output, "final-position decoder projection"),
    )


def aggregate_keys(positions: Iterable[ContextualPosition]) -> dict[int, Fraction]:
    """Compatibility readout of the final contextual state in token coordinates."""
    context = decoder_context(positions)
    return {} if context is None else dict(context.token_keys)


def project_tokens(state: Mapping[int, Fraction],
                   addresses: Sequence[PositionAddress]) -> dict[int, Fraction]:
    """Read one factorised hidden state in token coordinates for verification."""
    output: defaultdict[int, Fraction] = defaultdict(Fraction)
    for position_index, share in state.items():
        output[addresses[position_index].token_id] += share
    return _normalize(output, "token-coordinate readout")
