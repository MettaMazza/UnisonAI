"""Counted causal attention transformer for Unison's native generalisation route.

This module is a one-to-one constitutional translation of established decoder
transformer computation.  It does not repair or replace the retired historical
fallback.

Standard organ                         Counted SFT execution
-------------------------------        ----------------------------------------
role-bound causal training sequence    full prompt tokens + assistant BOS/response/EOS
token embedding                        exact prompt co-occurrence profile
causal Q/K/V attention                 prefix query + prompt-token keys -> next-token counts
positional/context representation      current turn plus dyadically aged history
feed-forward key/value memory          assistant-prefix -> next-token count table
residual connection                    unit attention distribution + unit FFN distribution
normalisation                          exact ``Fraction`` shares closing to the One
language-model head                    categorical next-token count shares
autoregressive decoding                causal greedy decode until learned EOS
reward-conditioned training            Laplace good/bad transition observations

There are no learned floating-point weights.  Corpus learning is one counted
pass (the closed-form categorical MLE); feedback is persisted observation state,
not a fixed parameter or an agent-selected reward coefficient.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import os
import pickle
import re
from typing import Iterable, Mapping, Optional, Sequence

from omni.core import BAND, GEN_B, halt_violation
from omni.word_engine import tokenize


SCHEMA = "unison-counted-causal-transformer/v4"
ROLE_POLICY = "prompt-keys/assistant-causal-values"
BOS = "\x02assistant"
EOS = "\x03assistant"
DEFAULT_STORE = os.path.join(os.path.dirname(__file__), "native_transformer_v4.pkl")
DEFAULT_REWARD = os.path.join(os.path.dirname(__file__), "native_transformer_reward_v4.pkl")
_TAILS = frozenset({"s", "t", "re", "ve", "ll", "d", "m"})
_CLOSE = frozenset(".,!?;:)]}%…’\"'")
_OPEN = frozenset("([{“‘")


def _merge_contractions(tokens: Sequence[str]) -> list[str]:
    """Apply the same deterministic token merge used by the sealed fluency build."""
    out: list[str] = []
    index = 0
    while index < len(tokens):
        if (index + 2 < len(tokens)
                and tokens[index + 1] in {"'", "’"}
                and tokens[index + 2].lower() in _TAILS
                and tokens[index].isalpha()):
            out.append(tokens[index] + "'" + tokens[index + 2])
            index += 3
        else:
            out.append(tokens[index])
            index += 1
    return out


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _merge_contractions(tokenize(text))]


def _prompt_tokens(text: str) -> list[str]:
    """The complete deterministic prompt-token sequence.

    A causal transformer does not discard function words, punctuation, or
    repetition before attention. Earlier development versions incorrectly
    reduced the prompt to unique content words; v4 retains the complete token
    sequence and aggregates repeated observations only when exact attention
    shares are accumulated. Distinct within-turn positional addresses remain a
    separately traced production organ rather than being falsely claimed here.
    """
    return _tokens(text)


def _plain(mapping):
    if isinstance(mapping, defaultdict):
        mapping = dict(mapping)
    return {key: dict(value) if isinstance(value, (Counter, defaultdict)) else value
            for key, value in mapping.items()}


def build_counted_transformer(prompts: Sequence[str], responses: Sequence[str],
                              source_sha256: str = "fixture") -> dict:
    """Execute the standard causal-language-model training pass as exact counts.

    Every prompt/assistant pair is one role-bound training sequence.  Assistant
    transitions are deposited once into the causal attention and FFN memories;
    no epochs, gradient estimates, fitted coefficients, pruning thresholds, or
    candidate caps are introduced.
    """
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("prompt and response sequences must be non-empty and aligned")

    vocab: dict[str, int] = {}
    words: list[str] = []

    def word_id(word: str) -> int:
        found = vocab.get(word)
        if found is None:
            found = len(words)
            vocab[word] = found
            words.append(word)
        return found

    bos_id, eos_id = word_id(BOS), word_id(EOS)
    unigram: Counter[int] = Counter()
    profiles: defaultdict[int, Counter[int]] = defaultdict(Counter)
    qk: Counter[tuple[int, int]] = Counter()
    values: defaultdict[int, Counter[int]] = defaultdict(Counter)
    semantic_ffn: defaultdict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    semantic_ffn3: defaultdict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    ffn2: defaultdict[int, Counter[int]] = defaultdict(Counter)
    ffn3: defaultdict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    response_count = 0
    token_count = 0

    for prompt, response in zip(prompts, responses):
        prompt_words = _prompt_tokens(prompt)
        response_words = _tokens(response)
        if not prompt_words or not response_words:
            continue
        prompt_ids = [word_id(word) for word in prompt_words]
        response_ids = [word_id(word) for word in response_words]

        # Counted embedding training: a token's vector is its prompt-context
        # co-occurrence row, the explicit sparse counterpart of an embedding row.
        for left in prompt_ids:
            for right in prompt_ids:
                if left != right:
                    profiles[left][right] += 1

        prev_id = bos_id
        last_id = bos_id
        for next_id in response_ids + [eos_id]:
            unigram[next_id] += 1
            ffn2[last_id][next_id] += 1
            ffn3[(prev_id, last_id)][next_id] += 1
            for key_id in prompt_ids:
                # Standard attention factorisation: the generated-prefix query
                # and prompt key accumulate a scalar compatibility count; the
                # key owns one value vector. Sequence prediction remains the
                # separate FFN organ rather than being duplicated inside V.
                qk[(last_id, key_id)] += 1
                values[key_id][next_id] += 1
                semantic_ffn[(last_id, key_id)][next_id] += 1
                semantic_ffn3[(prev_id, last_id, key_id)][next_id] += 1
            prev_id, last_id = last_id, next_id
            token_count += 1
        response_count += 1

    if not response_count or not unigram:
        raise RuntimeError("training pass produced no assistant transitions")

    profile_index: defaultdict[int, set[int]] = defaultdict(set)
    for token_id, row in profiles.items():
        for context_id in row:
            profile_index[context_id].add(token_id)

    return {
        "schema": SCHEMA,
        "role_policy": ROLE_POLICY,
        "source_pairs_sha256": source_sha256,
        "response_count": response_count,
        "token_count": token_count,
        "vocab": vocab,
        "words": words,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "unigram": dict(unigram),
        "profiles": _plain(profiles),
        "profile_index": {key: tuple(sorted(value))
                          for key, value in profile_index.items()},
        "qk": dict(qk),
        "values": _plain(values),
        "semantic_ffn": _plain(semantic_ffn),
        "semantic_ffn3": _plain(semantic_ffn3),
        "ffn2": _plain(ffn2),
        "ffn3": _plain(ffn3),
    }


def _unit(counts: Mapping[int, int]) -> dict[int, Fraction]:
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {token_id: Fraction(count, total) for token_id, count in counts.items()
            if count > 0}


def _add(target: defaultdict[int, Fraction], source: Mapping[int, Fraction],
         weight: Fraction = Fraction(1)) -> None:
    for token_id, share in source.items():
        target[token_id] += weight * share


def _normalize(values: Mapping[int, Fraction]) -> dict[int, Fraction]:
    total = sum(values.values(), Fraction(0))
    if total <= 0:
        return {}
    result = {token_id: value / total for token_id, value in values.items()
              if value > 0}
    if sum(result.values(), Fraction(0)) != 1:
        halt_violation("native transformer LM head did not close to the One")
    return result


class CountedCausalTransformer:
    """Executable one-block decoder transformer over exact counted state."""

    def __init__(self, store_path: str = DEFAULT_STORE,
                 reward_path: str = DEFAULT_REWARD, record: dict | None = None):
        self.store_path = store_path
        self.reward_path = reward_path
        self._record = record
        self._reward = None
        self._reward_events = None
        self._unit_cache = {}

    def available(self) -> bool:
        return self._record is not None or os.path.exists(self.store_path)

    def _store(self) -> dict:
        if self._record is None:
            with open(self.store_path, "rb") as handle:
                self._record = pickle.load(handle)
        record = self._record
        if record.get("schema") != SCHEMA or record.get("role_policy") != ROLE_POLICY:
            halt_violation("native transformer store provenance mismatch")
        return record

    def _rewards(self) -> dict:
        if self._reward is None:
            try:
                with open(self.reward_path, "rb") as handle:
                    loaded = pickle.load(handle)
                if loaded.get("schema") != SCHEMA + "/reward":
                    halt_violation("native transformer reward provenance mismatch")
                self._reward = loaded.get("counts", {})
                self._reward_events = loaded.get("events", [])
            except FileNotFoundError:
                self._reward = {}
                self._reward_events = []
        return self._reward

    def identity(self) -> dict:
        record = self._store()
        identity = {
            "schema": record["schema"],
            "role_policy": record["role_policy"],
            "source_pairs_sha256": record["source_pairs_sha256"],
            "responses": record["response_count"],
            "tokens": record["token_count"],
        }
        if os.path.exists(self.store_path):
            with open(self.store_path, "rb") as handle:
                identity["sha256"] = hashlib.sha256(handle.read()).hexdigest()
        return identity

    def _positional_keys(self, text: str,
                         history: Iterable | None = None) -> dict[int, Fraction]:
        """Token keys with exact dyadic turn age, before Q/K addressing."""
        record = self._store()
        vocab = record["vocab"]
        weighted: defaultdict[int, Fraction] = defaultdict(Fraction)

        history_texts: list[str] = []
        for event in history or ():
            if isinstance(event, (tuple, list)) and len(event) >= 2:
                role, value = str(event[0]).lower(), str(event[1])
                if role in {"user", "human", "prompt"}:
                    history_texts.append(value)
            elif isinstance(event, dict):
                role = str(event.get("role", "")).lower()
                if role in {"user", "human", "prompt"}:
                    history_texts.append(str(event.get("content", "")))

        turns = history_texts + [text]
        if len(turns) >= 2 and turns[-2].strip().lower() == text.strip().lower():
            del turns[-2]
        for age, turn in enumerate(reversed(turns)):
            positional_share = Fraction(1, GEN_B ** age)
            for word in _prompt_tokens(turn):
                token_id = vocab.get(word)
                if token_id is not None:
                    weighted[token_id] += positional_share
        return dict(weighted)

    def _context_keys(self, text: str, history: Iterable | None = None) -> dict[int, Fraction]:
        """Causally available full-token keys with exact turn-position shares."""
        return self._positional_keys(text, history)

    def _attention_key_weights(self, last_id: int,
                               keys: Mapping[int, Fraction]) -> dict[int, Fraction]:
        """Combine exact structural, information, and Q/K association heads.

        Standard transformer heads learn different projections of the same
        token state.  Here the projections are established counted forms: the
        identity/structural head, reciprocal key exposure (the exact
        inverse-frequency information projection), and conditional Q/K
        association.  Every available head closes separately to the One before
        exact addition; no fitted blend or lexical routing rule enters.
        """
        if not keys:
            return {}
        record = self._store()
        structural_total = sum(keys.values(), Fraction(0))
        if structural_total <= 0:
            return {}
        structural = {
            key_id: share / structural_total for key_id, share in keys.items()
            if share > 0
        }
        exposure = {
            key_id: sum(record["values"].get(key_id, {}).values())
            for key_id in structural
        }
        information = {
            key_id: keys[key_id] / held
            for key_id, held in exposure.items() if held > 0
        }
        association = {
            key_id: keys[key_id]
            * Fraction(record["qk"].get((last_id, key_id), 0), held)
            for key_id, held in exposure.items()
            if held > 0 and record["qk"].get((last_id, key_id), 0) > 0
        }
        heads = [structural]
        for head in (information, association):
            total = sum(head.values(), Fraction(0))
            if total > 0:
                heads.append({key_id: value / total
                              for key_id, value in head.items()})
        addressed: defaultdict[int, Fraction] = defaultdict(Fraction)
        for head in heads:
            _add(addressed, head)
        return dict(addressed)

    @staticmethod
    def _add_count_row(target: defaultdict[int, Fraction],
                       counts: Mapping[int, int], weight: Fraction) -> None:
        """Add an exact normalized count row without materializing its unit row."""
        total = sum(counts.values())
        if total <= 0 or weight <= 0:
            return
        factor = weight / total
        for token_id, count in counts.items():
            if count > 0:
                target[token_id] += factor * count

    def _unnormalized_residual(self, prev_id: int, last_id: int,
                               keys: Mapping[int, Fraction]) -> dict[int, Fraction]:
        """Exact residual mass for greedy decode, without dense unit-row caches.

        Each organ is accumulated with the same unit normalization and identity
        residual coefficient as ``next_distribution``. Omitting only the final
        common normalization cannot change the argmax.
        """
        record = self._store()
        addressed = self._attention_key_weights(last_id, keys)
        residual: defaultdict[int, Fraction] = defaultdict(Fraction)

        value_rows = [(weight, record["values"].get(key_id))
                      for key_id, weight in addressed.items()]
        value_rows = [(weight, counts) for weight, counts in value_rows if counts]
        value_weight = sum((weight for weight, _ in value_rows), Fraction(0))
        if value_weight > 0:
            attention: defaultdict[int, Fraction] = defaultdict(Fraction)
            for weight, counts in value_rows:
                self._add_count_row(attention, counts, weight / value_weight)
            _add(residual, attention)

        semantic_rows = []
        for key_id, weight in addressed.items():
            counts = record["semantic_ffn3"].get((prev_id, last_id, key_id))
            if not counts:
                counts = record["semantic_ffn"].get((last_id, key_id))
            if counts:
                semantic_rows.append((weight, counts))
        semantic_weight = sum((weight for weight, _ in semantic_rows), Fraction(0))
        if semantic_weight > 0:
            semantic: defaultdict[int, Fraction] = defaultdict(Fraction)
            for weight, counts in semantic_rows:
                self._add_count_row(semantic, counts, weight / semantic_weight)
            _add(residual, semantic)

        counts = record["ffn3"].get((prev_id, last_id))
        if not counts:
            counts = record["ffn2"].get(last_id)
        if not counts:
            counts = record["unigram"]
        self._add_count_row(residual, counts, Fraction(1))

        rewards = self._rewards()
        for token_id in list(residual):
            good, bad = rewards.get((prev_id, last_id, token_id), (0, 0))
            residual[token_id] *= Fraction(good + 1, good + bad + GEN_B)
        return dict(residual)

    def next_token_id(self, prev_id: int, last_id: int,
                      keys: Mapping[int, Fraction]) -> Optional[int]:
        """Exact greedy argmax, equivalent to the normalized LM-head argmax."""
        residual = self._unnormalized_residual(prev_id, last_id, keys)
        if not residual:
            return None
        return min(residual, key=lambda token_id: (-residual[token_id], token_id))

    def _attention(self, prev_id: int, last_id: int,
                   keys: Mapping[int, Fraction]) -> dict[int, Fraction]:
        """Causal Q/K/V attention: prefix query, prompt keys, count-vector values."""
        record = self._store()
        addressed = self._attention_key_weights(last_id, keys)
        branches: list[tuple[Fraction, dict[int, Fraction]]] = []
        for key_id, key_weight in addressed.items():
            counts = record["values"].get(key_id)
            if counts:
                branches.append((key_weight,
                                 self._cached_unit("value", key_id, counts)))
        if not branches:
            return {}
        key_total = sum((weight for weight, _ in branches), Fraction(0))
        out: defaultdict[int, Fraction] = defaultdict(Fraction)
        for weight, values in branches:
            _add(out, values, weight / key_total)
        return _normalize(out)

    def _cached_unit(self, namespace: str, key, counts: Mapping[int, int]) -> dict[int, Fraction]:
        """Memoize an immutable store row's exact unit distribution.

        This changes no value and imposes no candidate or cache-size cap; it
        avoids reconstructing the same exact Fractions at every generated token.
        """
        address = (namespace, key)
        held = self._unit_cache.get(address)
        if held is None:
            held = _unit(counts)
            self._unit_cache[address] = held
        return held

    def _ffn(self, prev_id: int, last_id: int) -> dict[int, Fraction]:
        """Explicit FFN KV memory addressed by the generated causal prefix."""
        record = self._store()
        counts = record["ffn3"].get((prev_id, last_id))
        if not counts:
            counts = record["ffn2"].get(last_id)
        if not counts:
            counts = record["unigram"]
        return self._cached_unit("prefix_ffn", (prev_id, last_id), counts)

    def _semantic_ffn(self, prev_id: int, last_id: int,
                      keys: Mapping[int, Fraction]) -> dict[int, Fraction]:
        """FFN transformation of the attended prompt state.

        Standard transformer order is attention first, then FFN over that
        hidden state. The explicit counted KV address is therefore
        ``(generated-prefix query, attended prompt key)``; its value is the
        exact next-token count vector learned in the same causal pass.
        """
        record = self._store()
        addressed = self._attention_key_weights(last_id, keys)
        branches: list[tuple[Fraction, dict[int, Fraction]]] = []
        for key_id, key_weight in addressed.items():
            address = (prev_id, last_id, key_id)
            counts = record["semantic_ffn3"].get(address)
            namespace = "semantic_ffn3"
            if not counts:
                address = (last_id, key_id)
                counts = record["semantic_ffn"].get(address)
                namespace = "semantic_ffn2"
            if counts:
                branches.append((key_weight,
                                 self._cached_unit(
                                     namespace, address, counts)))
        if not branches:
            return {}
        total = sum((weight for weight, _ in branches), Fraction(0))
        out: defaultdict[int, Fraction] = defaultdict(Fraction)
        for weight, values in branches:
            _add(out, values, weight / total)
        return _normalize(out)

    def next_distribution(self, prev_id: int, last_id: int,
                          keys: Mapping[int, Fraction]) -> dict[int, Fraction]:
        """Attention -> residual -> FFN -> residual -> exact LM-head normalization."""
        attention = self._attention(prev_id, last_id, keys)
        semantic_ffn = self._semantic_ffn(prev_id, last_id, keys)
        ffn = self._ffn(prev_id, last_id)
        residual: defaultdict[int, Fraction] = defaultdict(Fraction)
        if attention:
            _add(residual, attention)
        if semantic_ffn:
            _add(residual, semantic_ffn)
        if ffn:
            _add(residual, ffn)

        rewards = self._rewards()
        for token_id in list(residual):
            good, bad = rewards.get((prev_id, last_id, token_id), (0, 0))
            # Classical Laplace preference share. It is counted learned state;
            # no reward scale or clipping parameter is introduced.
            residual[token_id] *= Fraction(good + 1, good + bad + GEN_B)
        return _normalize(residual)

    def generate_tokens(self, text: str, history: Iterable | None = None,
                        token_budget: int = BAND) -> list[str]:
        """Standard greedy autoregressive decode under an explicit resource budget."""
        if not isinstance(token_budget, int) or token_budget <= 0:
            raise ValueError("token_budget must be a positive integer")
        record = self._store()
        keys = self._context_keys(text, history)
        if not keys:
            return []
        prev_id = last_id = record["bos_id"]
        out: list[str] = []
        for _ in range(token_budget):
            next_id = self.next_token_id(prev_id, last_id, keys)
            if next_id is None:
                break
            if next_id == record["eos_id"]:
                break
            out.append(record["words"][next_id])
            prev_id, last_id = last_id, next_id
        return out

    @staticmethod
    def _surface(tokens: Sequence[str]) -> str:
        surface = ""
        for token in tokens:
            if not surface:
                surface = token[:1].upper() + token[1:]
            elif token[:1] in _CLOSE or surface[-1:] in _OPEN:
                surface += token
            else:
                surface += " " + token
        return re.sub(r"\s+([.,!?;:])", r"\1", surface).strip()

    def generate(self, text: str, history: Iterable | None = None,
                 token_budget: int = BAND) -> str:
        return self._surface(self.generate_tokens(text, history, token_budget))

    def mark_feedback(self, text: str, response: str, good: bool) -> None:
        """Deposit one reward observation for each served causal transition."""
        record = self._store()
        response_ids = [record["vocab"].get(token) for token in _tokens(response)]
        response_ids = [token_id for token_id in response_ids if token_id is not None]
        if not response_ids:
            return
        rewards = self._rewards()
        prev_id = last_id = record["bos_id"]
        for next_id in response_ids + [record["eos_id"]]:
            wins, losses = rewards.get((prev_id, last_id, next_id), (0, 0))
            rewards[(prev_id, last_id, next_id)] = (
                wins + (1 if good else 0), losses + (0 if good else 1))
            prev_id, last_id = last_id, next_id
        self._reward_events.append({
            "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "surface_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
            "good": bool(good),
            "transition_count": len(response_ids) + 1,
        })
        payload = {"schema": SCHEMA + "/reward", "counts": rewards,
                   "events": self._reward_events}
        stage = self.reward_path + ".building"
        with open(stage, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(stage, self.reward_path)


native_transformer = CountedCausalTransformer()
