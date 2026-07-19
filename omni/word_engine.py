"""
The WORD tier — generation over word-tokens.

The generation hierarchy is WORD -> PHRASE(SEG) -> CHAR, in that order, the way
humans produce language:
  WORD    the primary unit — produce whole words (never spell a word you know);
  PHRASE  words BUILD into phrases/segments — whole constructions come whole as
          the words assemble ("nice to meet you");
  CHAR    the fallback below word — only a word it does NOT know is spelled out,
          then it resumes at the word level.

The char engine (`memory.py`) generates one character at a time with longest-
suffix selection. This module adds the scale between segment and character: it
generates one WORD at a time, so every atom is a real word.

GENERATION uses the fold's generation law — unit-capacity selection over the
LONGEST held word-suffix, continuations valued as exact rational shares, SAMPLED
(`sample_next_unit`). Committing to the single deepest matching context keeps a
walk inside one orbit's flow and branches only where that deep context genuinely
has more than one observed continuation, so generation stays coherent instead of
stitching across orbits. The word context reaches up to WORD_CTX_MAX words —
"longest held suffix", not GEN_B*GEN_C (=6), which is the fold's structural depth
constant, not the generation-context cap.

`_scored`/`sample_next` also expose the fold-factor level-MIX distribution
(2^level across all depths) — the arm the word-scale campaign
(omni/benchmarks/rung5e_*) used to beat a trained transformer on cross-entropy —
but that is for probability calibration; generation itself uses unit-capacity,
because mixing shallow levels stitches across orbits.

GENERALISATION comes from counted KINSHIP (`NEIGH`/`kinship`/`kin_expand`, ported
from the deleted unison_chat.py): words that share contexts are kin, so a novel
prompt can route to a taught orbit through its content-words' kin.

The word stores are built from the SAME held orbits the char engine learns from
(`graph.orbits`), so the word tier learns whatever Unison is taught.
"""
import re
from fractions import Fraction
from collections import defaultdict, Counter

from omni.core import CTX_MAX, GEN_B, GEN_C
from omni.generation_boundaries import COHERENCE_LOCK
from omni.logging_config import get_logger

logger = get_logger("OmniWordEngine", "word_engine.log")

# The fold's generation law is "unit-capacity selection over the LONGEST held
# suffix" — the char engine matches suffixes up to thousands of characters, not
# to GEN_B*GEN_C=6. CTX_MAX=6 is the fold's structural depth constant, NOT the
# generation-context cap; capping the WORD context at 6 lost the thread across
# any answer longer than 6 words (shared mid-phrases collide and it jumps orbit).
# So the word tier matches the longest held word-suffix up to a depth that spans
# a full conversational answer + its reasoning.
WORD_CTX_MAX = 48

# UNFOLD depth cap (memory_abstraction.ep): generation must not re-excite a long
# held surface (that is verbatim replay). Capping the context to a few words forces
# re-composition from general statistics -- the closure regime -- so a whole
# memorized answer can never be walked out word-for-word.
UNFOLD_CTX_MAX = 4

# ── Counted kinship (generalisation), ported from the deleted unison_chat.py ──
# Two words are KIN when they share contexts (neighbours). This is the exact-count
# stand-in for a trained embedding: "what do they CALL you?" can route to the
# "what's your NAME?" orbit because CALL and NAME keep similar company.
KIN_FLOOR = 1.0 / (GEN_B * GEN_C)   # 1/6 — a candidate must clear this kin score
KIN_K = GEN_C                        # kin breadth per word = the colour count (3)
KIN_VOTE = 0.5                       # kin votes at half weight — the fold factor 1/GEN_B
_STOPWORDS = {
    "the", "and", "you", "are", "for", "that", "this", "with", "have", "was",
    "your", "what", "how", "not", "but", "can", "all", "get", "one", "out",
    "its", "who", "why", "did", "does", "about", "just", "like", "some", "any",
    "been", "were", "they", "them", "there", "here", "then", "than", "his", "her",
}

# Words, and every non-word non-space char (punctuation AND the engine's control
# tokens \x02..\x05) as its own token.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

STOP = "\x02"                 # start of next user turn -> stop generating
END_USER = "\x03"            # end of a user turn -> keep as context, don't emit
THINK_OPEN, THINK_CLOSE = "\x04", "\x05"

# Spacing rules for reconstructing surface text from tokens.
_CLOSE = set(".,!?;:)]}%…’”\"'")
_OPEN = set("([{“‘")


def tokenize(s):
    return _TOKEN_RE.findall(s)


def _content_words(words):
    """The informative words of a query — length >= 3 and not a common function
    word — the ones whose kin can route a novel prompt to a taught orbit."""
    return [w.lower() for w in words if len(w) >= 3 and w.lower() not in _STOPWORDS]


# register the generic fallback must avoid serving (code / letter / essay / dialogue / list)
_REGISTER_LEAK = ("main function", "prompt the user", "variable", "palindrome", "best regards",
                  "sincerely", "sign off", "conclude by", "as follows", "click here",
                  "dear friend", "dear colleague", "electronic source", "furthermore",
                  "moreover", "sofia:", "page ", "chapter ", "http")
# first/second-person conversational markers — a span that reads like talk, not prose
_CONV_MARKERS = {"i", "you", "your", "we", "my", "me", "im", "youre", "were", "lets", "thats",
                 "ive", "youve", "weve", "ill", "youll", "dont", "its", "yeah", "hey", "how",
                 "what", "nice", "hi", "hello", "thanks", "good"}


def _degenerate_repeat(content_words):
    """True when a spliced surface echoes the same content word (the classic pivot-splice
    artefact: '...vastness of the ocean is a wonder... of the ocean...'). Any content word
    appearing 2+ times in a short reply marks an incoherent seam, not natural emphasis."""
    if len(content_words) < 3:
        return False
    seen = {}
    for w in content_words:
        seen[w] = seen.get(w, 0) + 1
        if seen[w] >= 2:
            return True
    return False


def _cycled(words, reps=3, max_period=6):
    """True when the tail of `words` is a short block repeated `reps` times —
    a backstop against degenerate loops (e.g. 'you are Maria you are Maria …').

    Repeated *punctuation* is NOT a degenerate loop: an ellipsis ('.','.','.'),
    '!!!' or '…' are legitimate and extremely common in the teacher's reasoning
    ('Wait...', 'That's...'). Only flag a cycle whose repeated block carries real
    (alphanumeric) content — otherwise generation halts at the first ellipsis."""
    n = len(words)
    for p in range(1, max_period + 1):
        if n >= p * reps and all(words[-p:] == words[-(k + 1) * p:-k * p] for k in range(1, reps)):
            block = words[-p:]
            if any(any(ch.isalnum() for ch in tok) for tok in block):
                return True
    return False


def _needs_space(surface, tok):
    if not surface:
        return False
    if len(tok) == 1 and tok in _CLOSE:
        return False
    if surface[-1] in _OPEN:
        return False
    return True


class WordEngine:
    """Multi-level word-orbit stores + fold-mix next-word prediction."""

    def __init__(self, fluency_path=None):
        self.stores = [defaultdict(lambda: defaultdict(int)) for _ in range(WORD_CTX_MAX + 1)]
        self.vocab = 1
        self._sig = None
        # ── kinship substrate ──
        self.neigh = {}                       # word -> {neighbour: count}
        self.tok_freq = Counter()
        self.orbit_prompts = []               # per orbit: its \x02..\x03 prompt char-context
        self.orbit_content = []               # per orbit: its prompt's content-word set
        self.word_index = defaultdict(set)    # content word -> orbit indices
        self.neigh_index = defaultdict(set)   # context word -> words holding it as neighbour
        self._fluency_path = fluency_path
        self._fluency_identity = None
        self._fluency_registration = None

    def configure_fluency_store(self, path):
        """Select one explicit generation-surface store and drop cached state.

        The global live engine keeps the historical default when ``path`` is
        omitted. Development arms can select the sealed response-only artifact
        without renaming files or silently changing the live path.
        """
        self._fluency_path = path
        self._fluency = None
        self._fluency_identity = None
        self._fluency_registration = None
        self._openers = None

    def configure_registered_fluency(self, registration_path):
        """Activate one explicitly registered response-only runtime arm.

        The registration binds the already sealed artifact receipt and artifact
        bytes. Nothing is selected when this method is not called.
        """
        import hashlib, json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]

        def resolve(value):
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (root / path).resolve()

        arm_path = resolve(registration_path)
        arm_raw = arm_path.read_bytes()
        arm = json.loads(arm_raw)
        if arm.get("schema") != "unison-response-fluency-runtime-arm/v1" or \
                arm.get("status") != "registered":
            raise RuntimeError("unsupported response-fluency runtime arm")
        receipt_binding = arm.get("receipt", {})
        receipt_path = resolve(receipt_binding.get("path", ""))
        receipt_raw = receipt_path.read_bytes()
        if hashlib.sha256(receipt_raw).hexdigest() != receipt_binding.get("sha256"):
            raise RuntimeError("response-fluency receipt hash mismatch")
        receipt = json.loads(receipt_raw)
        if receipt.get("schema") != "unison-response-fluency-receipt/v1" or \
                receipt.get("status") != "sealed":
            raise RuntimeError("response-fluency receipt is not sealed")
        artifact_binding = arm.get("artifact", {})
        if artifact_binding != receipt.get("artifact"):
            raise RuntimeError("runtime arm artifact differs from sealed receipt")
        artifact_path = resolve(artifact_binding.get("path", ""))
        if artifact_path.stat().st_size != artifact_binding.get("bytes") or \
                hashlib.sha256(artifact_path.read_bytes()).hexdigest() != \
                artifact_binding.get("sha256"):
            raise RuntimeError("registered response-fluency artifact hash mismatch")
        self.configure_fluency_store(artifact_path)
        identity = self.fluency_identity()
        expected = {
            "sha256": artifact_binding["sha256"],
            "schema": "unison-response-fluency/v1",
            "role": receipt.get("role"),
            "boundary_policy": receipt.get("boundary_policy"),
        }
        if any(identity.get(key) != value for key, value in expected.items()):
            self.configure_fluency_store(None)
            raise RuntimeError("loaded response-fluency identity differs from registration")
        self._fluency_registration = {
            "path": str(arm_path),
            "sha256": hashlib.sha256(arm_raw).hexdigest(),
            "schema": arm["schema"],
        }
        return self.fluency_identity()

    def fluency_identity(self):
        """Return the hash-bound identity of the generation surface in use."""
        self._load_fluency()
        identity = dict(self._fluency_identity or {})
        if identity and self._fluency_registration:
            identity["runtime_arm"] = dict(self._fluency_registration)
        return identity

    # ── build from the same held orbits the char engine learns from ──
    def ensure_built(self, graph, ukey):
        """Rebuild the word stores if the graph's orbits changed. Cheap at the
        live scale (the learned graph is small and grows one turn at a time)."""
        sig = (len(graph.orbits.get(ukey, [])), len(graph.orbits.get("public", [])))
        if sig == self._sig:
            return
        try:
            self._rebuild(graph, ukey)
            self._sig = sig
        except Exception:
            logger.error("word store rebuild failed", exc_info=True)

    def _rebuild(self, graph, ukey):
        stores = [defaultdict(lambda: defaultdict(int)) for _ in range(WORD_CTX_MAX + 1)]
        vocab = set()
        neigh = defaultdict(lambda: defaultdict(int))
        tok_freq = Counter()
        orbit_prompts = []
        orbit_content = []
        word_index = defaultdict(set)
        for scope in (ukey, "public"):
            for orbit in graph.orbits.get(scope, []):
                words = tokenize("".join(orbit))
                for w in words:
                    vocab.add(w.lower())
                    tok_freq[w.lower()] += 1
                # suffix stores (generation)
                for i in range(len(words) - 1):
                    nxt = words[i + 1]
                    for L in range(0, WORD_CTX_MAX + 1):
                        if i - L + 1 < 0:
                            break
                        key = tuple(w.lower() for w in words[i - L + 1:i + 1])
                        stores[L][key][nxt] += 1
                # kinship co-occurrence: each word's immediate neighbours
                for i in range(1, len(words) - 1):
                    w = words[i].lower()
                    if len(w) >= 3:
                        neigh[w][words[i - 1].lower()] += 1
                        neigh[w][words[i + 1].lower()] += 1
                # prompt index: the \x02..\x03 span of each orbit, keyed by its content words
                if END_USER in orbit:
                    e = orbit.index(END_USER)
                    prompt_chars = list(orbit[:e + 1])
                    oi = len(orbit_prompts)
                    orbit_prompts.append(prompt_chars)
                    cwset = set(_content_words(tokenize("".join(prompt_chars))))
                    orbit_content.append(cwset)
                    for w in cwset:
                        word_index[w].add(oi)
        self.stores = stores
        self.vocab = max(1, len(vocab))
        self.neigh = neigh
        self.tok_freq = tok_freq
        self.orbit_prompts = orbit_prompts
        self.orbit_content = orbit_content
        self.word_index = word_index
        # inverted kin index: context word -> words holding it (skip very common words)
        total = sum(tok_freq.values())
        common = total / (2 ** 9) if total else 0
        neigh_index = defaultdict(set)
        for w, nb in neigh.items():
            for c in nb:
                if tok_freq.get(c, 0) <= common:
                    neigh_index[c].add(w)
        self.neigh_index = neigh_index
        logger.info(f"word stores rebuilt: {sum(len(s) for s in stores)} orbits, vocab {self.vocab}, "
                    f"{len(orbit_prompts)} prompts, {len(neigh)} kin words")

    # ── fold-mix next-word distribution (exact rational) ──
    def _scored(self, context_words):
        """Return (scored, deepest_level) where scored is a list of
        (word, exact Fraction weight) — the fold-mix distribution over the next
        word. For each candidate w the weight is

            sum over holding levels L of  2^L * (count_L(w) + 1/V) / (total_L + 1)

        the fold factor 2^L weighting deeper context heavier. Candidates come
        from CONTEXTFUL levels (L>=1); the unigram level (L=0) contributes to
        each weight as the fold floor but is not itself a candidate source, so
        generation stays in-context instead of wandering into common words. If
        only the unigram level holds, its raw distribution is returned.
        """
        ctx = [w.lower() for w in context_words]
        level_data = []
        deepest = 0
        for L in range(min(WORD_CTX_MAX, len(ctx)), -1, -1):
            key = tuple(ctx[-L:]) if L else ()
            s = self.stores[L].get(key)
            if s:
                level_data.append((L, s, sum(s.values())))
                if not deepest:
                    deepest = L
            elif L == 0:
                level_data.append((0, None, 0))

        candidates = set()
        for L, s, total in level_data:
            if s and L >= 1:
                candidates.update(s.keys())
        if not candidates:
            s0 = self.stores[0].get(())
            if not s0:
                return [], 0
            tot = sum(s0.values())
            return [(w, Fraction(c, tot)) for w, c in s0.items()], 0

        floor = Fraction(1, self.vocab)
        scored = []
        for w in candidates:
            score = Fraction(0)
            for L, s, total in level_data:
                if s is None:
                    continue
                score += (Fraction(2 ** L) * (s.get(w, 0) + floor)) / (total + 1)
            scored.append((w, score))
        return scored, deepest

    def predict_next(self, context_words):
        """Argmax over the fold-mix distribution (deterministic). Kept for
        diagnostics/inspection; generation uses `sample_next`."""
        scored, deepest = self._scored(context_words)
        if not scored:
            return None, 0, 0
        best = max(scored, key=lambda x: x[1])[0]
        return best, deepest, len(scored)

    def sample_next(self, context_words, rng):
        """Sample the next word from the exact fold-mix distribution. Good for
        probability calibration (cross-entropy); for GENERATION it mixes shallow
        levels and stitches across orbits — use `sample_next_unit` instead."""
        scored, deepest = self._scored(context_words)
        if not scored:
            return None, 0, 0
        total = sum(s for _, s in scored)
        if total <= 0:
            return scored[0][0], deepest, len(scored)
        r = Fraction(rng.randrange(1_000_000_000), 1_000_000_000) * total
        acc = Fraction(0)
        pick = scored[-1][0]
        for w, s in scored:
            acc += s
            if r < acc:
                pick = w
                break
        return pick, deepest, len(scored)

    def sample_next_unit(self, context_words, rng):
        """The fold's GENERATION law: unit-capacity selection over the orbit
        hierarchy — the LONGEST held suffix (deepest context that has a match),
        continuations valued as EXACT RATIONAL SHARES, sampled.

        Unit-capacity means we commit to the single deepest matching context
        rather than mixing all depths. Within one context the walk follows a
        single orbit's flow and only branches where that deep context genuinely
        has more than one observed continuation — so generation stays coherent
        instead of stitching fragments of different orbits together. The sample
        is an exact integer draw over the observed counts: no floating point,
        no chosen number — the share is the count over the total.
        """
        ctx = [w.lower() for w in context_words]
        for L in range(min(WORD_CTX_MAX, len(ctx)), 0, -1):
            s = self.stores[L].get(tuple(ctx[-L:]))
            if s:
                total = sum(s.values())
                r = rng.randrange(total)
                acc = 0
                for w, c in s.items():
                    acc += c
                    if r < acc:
                        return w, L, len(s)
                return next(iter(s)), L, len(s)
        s0 = self.stores[0].get(())
        if not s0:
            return None, 0, 0
        total = sum(s0.values())
        r = rng.randrange(total)
        acc = 0
        for w, c in s0.items():
            acc += c
            if r < acc:
                return w, 0, len(s0)
        return None, 0, 0

    def _load_fluency(self):
        """Load one explicitly identified general-language generation surface."""
        if getattr(self, "_fluency", None) is not None:
            return self._fluency
        import hashlib, os, pickle
        configured = self._fluency_path or "word_fluency.pkl"
        path = configured if os.path.isabs(str(configured)) else os.path.join(
            os.path.dirname(__file__), str(configured))
        self._fluency = {"maxl": 0, "uni": {}, "stores": [None]}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._fluency = pickle.load(f)
                schema = self._fluency.get("schema")
                if self._fluency_path is not None and \
                        schema != "unison-response-fluency/v1":
                    raise RuntimeError("explicit fluency store has unsupported schema")
                if schema == "unison-response-fluency/v1":
                    if self._fluency.get("role") != "assistant-response" or \
                            self._fluency.get("boundary_policy") != \
                            "reset-before-and-after-every-response":
                        raise RuntimeError("response-only fluency provenance mismatch")
                if self._fluency.get("maxl", 0) < 1 or \
                        not self._fluency.get("uni") or \
                        len(self._fluency.get("stores", [])) <= self._fluency.get("maxl", 0):
                    raise RuntimeError("fluency store is empty or incomplete")
                with open(path, "rb") as identity_file:
                    digest = hashlib.sha256(identity_file.read()).hexdigest()
                self._fluency_identity = {
                    "path": os.path.abspath(path),
                    "sha256": digest,
                    "schema": schema or "legacy-word-fluency",
                    "role": self._fluency.get("role"),
                    "boundary_policy": self._fluency.get("boundary_policy"),
                    "vocabulary": len(self._fluency.get("uni", {})),
                    "max_order": self._fluency.get("maxl", 0),
                }
                logger.info(f"fluency store loaded: vocab {len(self._fluency.get('uni', {})):,}, "
                            f"depths 1..{self._fluency.get('maxl', 0)}")
            except Exception:
                self._fluency = {"maxl": 0, "uni": {}, "stores": [None]}
                self._fluency_identity = None
                logger.error("fluency load failed", exc_info=True)
        return self._fluency

    def _merged_level(self, key, L):
        """Next-word counts at level L from the FOUNDATION (conversational fluency) ONLY.
        The taught-lesson orbit stores are deliberately NOT used for generation surface —
        that was a verbatim-recall route. Lessons inform retrieval/topic (kin_route), never
        the generated words. So every generated word comes from the foundation corpus =
        generalisation, never verbatim replay of a taught orbit."""
        m = Counter()
        flu = self._fluency
        if flu and 1 <= L <= flu.get("maxl", 0):
            f = flu["stores"][L].get(key)
            if f:
                m.update(f)
        return m

    def _scored_fluent(self, context_words):
        """The fold cascade (2^L) over lessons+fluency merged at each depth. This is what
        unfold samples: the lessons give topical content, the fluency gives coherent
        general phrasing, so composition is fresh AND coherent rather than lesson-salad."""
        self._load_fluency()
        ctx = [w.lower() for w in context_words]
        candidates = set()
        level_data = []
        deepest = 0
        for L in range(len(ctx), 0, -1):
            m = self._merged_level(tuple(ctx[-L:]), L)
            if m:
                level_data.append((L, m, sum(m.values())))
                candidates.update(m.keys())
                if not deepest:
                    deepest = L
        if not candidates:
            return [], 0
        floor = Fraction(1, max(self.vocab, len(self._fluency.get("uni", {})), 2))
        bias = getattr(self, "_gen_bias", None) or frozenset()
        boost = Fraction(GEN_B * GEN_C)   # topical words (the retrieved meaning) lead
        scored = []
        for w in candidates:
            score = Fraction(0)
            for L, m, total in level_data:
                score += (Fraction(2 ** L) * (m.get(w, 0) + floor)) / (total + 1)
            if w in bias:
                score = score * boost
            scored.append((w, score))
        return scored, deepest

    def set_generation_bias(self, words):
        """Topical bias for unfold: the content words of the retrieved meaning, boosted
        so fluent composition stays ON the lesson's topic instead of wandering."""
        self._gen_bias = frozenset(w.lower() for w in words if len(w) > 3)

    def _load_coupling(self):
        """Load the broad-corpus word-coupling graph (train_eval/build_coupling.py)."""
        if getattr(self, "_coupling_graph", None) is not None:
            return self._coupling_graph
        import pickle, os
        path = os.path.join(os.path.dirname(__file__), "word_coupling.pkl")
        self._coupling_graph = {}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._coupling_graph = pickle.load(f)
                logger.info(f"coupling graph loaded: {len(self._coupling_graph):,} words")
            except Exception:
                logger.error("coupling graph load failed", exc_info=True)
        return self._coupling_graph

    def structured_unfold(self, schema_words, rng):
        """Inactive historical unfold retained as an explicit audit boundary."""
        return ""

    def _span_key(self, text):
        # Stable content address: Python's process-randomised ``hash`` made the
        # persisted learning state point at different spans after a restart.
        import hashlib
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_span_quality(self):
        if getattr(self, "_span_quality", None) is not None:
            return self._span_quality
        import pickle, os
        path = os.path.join(os.path.dirname(__file__), "span_quality.pkl")
        self._span_quality = {}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._span_quality = pickle.load(f)
            except Exception:
                logger.error("span_quality load failed", exc_info=True)
        return self._span_quality

    def reinforce_spans(self, good):
        """STAGE 3 learning (counts, zero parameters): the foundation spans that produced a
        GOOD reply gain quality; a BAD reply loses it. Retrieval then prefers spans that have
        made good replies — the engine learns which foundation language actually works, from
        feedback, over time. This is what makes generation develop coherence with use."""
        q = self._load_span_quality()
        for t in getattr(self, "_last_spans", []):
            k = self._span_key(t)
            # One observed judgement contributes one counted event.  The state
            # is not clipped by an agent-selected bound.
            q[k] = q.get(k, 0) + (1 if good else -1)
        self._span_quality_dirty = True

    def save_span_quality(self):
        if not getattr(self, "_span_quality_dirty", False):
            return
        import pickle, os
        try:
            with open(os.path.join(os.path.dirname(__file__), "span_quality.pkl"), "wb") as f:
                pickle.dump(self._span_quality, f)
            self._span_quality_dirty = False
        except Exception:
            logger.error("span_quality save failed", exc_info=True)

    def _load_retrieval(self):
        if getattr(self, "_retrieval", None) is not None:
            return self._retrieval
        import pickle, os
        path = os.path.join(os.path.dirname(__file__), "retrieval.pkl")
        self._retrieval = {}
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self._retrieval = pickle.load(f)
                logger.info(f"retrieval index loaded: {len(self._retrieval.get('sentences', [])):,} sentences")
            except Exception:
                logger.error("retrieval load failed", exc_info=True)
        return self._retrieval

    def compose_reply(self, schema_words, rng):
        """Retired agent-authored fallback entry point; never served."""
        self._last_spans = []
        return ""

    def generic_reply(self, rng):
        """Retired agent-authored fallback entry point; never served."""
        return ""

    def retrieve_and_compose(self, schema_words, rng, only=None):
        """Retired agent-authored fallback entry point; never served."""
        return ""

    def unfold_response(self, schema_words, rng):
        """Retired agent-authored fallback entry point; never served."""
        return ""

    def reload_language_stores(self):
        """Drop the cached fluency + coupling stores so the next use reloads the freshly
        rebuilt foundation from disk (after /scrape extends and rebuilds them)."""
        self._fluency = None
        self._fluency_identity = None
        self._openers = None
        self._coupling_graph = None
        logger.info("language stores dropped; will reload rebuilt foundation on next use")

    def coupling(self, word, content):
        """Fold-determined coupling of a word to the statement's meaning (coherence_value.ep):
        the fraction of the content words it co-occurs strongly with, from the broad-corpus
        coupling graph. Coheres when >= the lock 1/2 (a majority of the meaning locks in).
        Falls back to the taught-orbit kinship when the word isn't in the corpus graph."""
        g = self._load_coupling()
        w = word.lower()
        others = [o for o in content if o != w]
        if not others:
            return 1.0
        nb = g.get(w)
        if nb:
            hits = sum(1 for o in others if o in nb)
            return hits / len(others)
        ks = [self.kinship(w, o) for o in others]
        return sum(ks) / len(ks) if ks else 0.0

    def reinforce_couplings(self, content_words, delta=0.5):
        """LEARNING (coherence_value.ep): feedback rated an output good, so strengthen the
        couplings among its content words. The coupling graph is what the critic reads and
        the generator is gated by, so reinforcing it is how the engine LEARNS to generate
        more coherently over time — not by memorising the output, by strengthening the
        associations that made it right."""
        g = self._load_coupling()
        cw = [w.lower() for w in content_words if len(w) > 3]
        if len(cw) < 2:
            return
        for a in cw:
            na = g.setdefault(a, {})
            for b in cw:
                if a != b:
                    na[b] = na.get(b, 0.0) + delta
        self._coupling_dirty = True

    def weaken_couplings(self, content_words, delta=0.5):
        """Feedback rated an output bad: weaken the couplings among its content words."""
        g = self._load_coupling()
        cw = [w.lower() for w in content_words if len(w) > 3]
        for a in cw:
            na = g.get(a)
            if not na:
                continue
            for b in cw:
                if b in na:
                    na[b] = max(0.0, na[b] - delta)
        self._coupling_dirty = True

    def save_couplings(self):
        """Persist the learned coupling graph so feedback accumulates across restarts."""
        if not getattr(self, "_coupling_dirty", False):
            return
        import pickle, os
        path = os.path.join(os.path.dirname(__file__), "word_coupling.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self._coupling_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._coupling_dirty = False
        except Exception:
            logger.error("coupling save failed", exc_info=True)

    def coherence_score(self, words, context_words=None):
        """CRITIC: the fold-determined coherence value of a statement (coherence_value.ep).
        word<->statement: each content word's coupling to the rest; statement<->context:
        the content's coupling to the conversation. Both read against the lock 1/2.
        Returns (word_scale, context_scale) in [0,1]; coherent when >= 1/2."""
        content = [w.lower() for w in words if len(w) > 3]
        # A reply with (almost) no content words is not "perfectly coherent" — it is
        # empty/garbage (e.g. char-babble). Score it LOW so the critic and the feedback
        # loop never reward it.
        if len(content) < 2:
            return 0.0, 0.0
        # degenerate repetition (babble like "lo lo lo") — very few DISTINCT content words
        if len(set(content)) < max(2, len(content) // 3):
            return 0.0, 0.0
        wc = [self.coupling(w, content) for w in content]
        word_scale = sum(wc) / len(wc)
        ctx_scale = 1.0
        if context_words:
            cc = [w.lower() for w in context_words if len(w) > 3]
            if cc:
                cvs = [max((self.kinship(w, c) for c in cc), default=0.0) for w in content]
                ctx_scale = sum(cvs) / len(cvs) if cvs else 1.0
        return word_scale, ctx_scale

    def sample_next_unfold(self, context_words, rng):
        """UNFOLD, value-guided by the coherence lock (memory_abstraction.ep +
        coherence_value.ep). Compose from general fluency (capped so no lesson is walked
        out verbatim), but GATE each content-word candidate by the fold coherence lock:
        a content word that couples with the statement's meaning at >= 1/2 is kept/boosted
        (it locks in — coherent), one that couples below the lock is suppressed (it would
        run free — the wander). Function words (grammar) are never gated, so phrasing
        stays fluent. Coherence is thus enforced by the sync threshold, the fold's way,
        instead of by n-gram probability alone.
        """
        ctx = [w.lower() for w in context_words]
        capped = ctx[-UNFOLD_CTX_MAX:] if len(ctx) > UNFOLD_CTX_MAX else ctx
        scored, deepest = self._scored_fluent(capped)
        if not scored:
            return None, 0, 0
        # the statement's meaning-so-far = retrieved topical bias + content words in context
        content = set(getattr(self, "_gen_bias", None) or frozenset())
        content.update(w for w in ctx if len(w) > 3)
        content = list(content)
        LOCK = float(COHERENCE_LOCK)
        adj = []
        for w, sc in scored:
            if len(w) > 3 and content:               # a content word: gate by the lock
                coup = self.coupling(w, content)
                if coup >= LOCK:
                    sc = sc * Fraction(GEN_B)        # locks in -> coheres, boost
                elif coup <= 0.0:
                    sc = sc / Fraction(GEN_B * GEN_C)  # no coupling -> would wander, suppress
            adj.append((w, sc))
        total = Fraction(0)
        for _, s in adj:
            total += s
        if total <= 0:
            return scored[0][0], deepest, len(scored)
        r = Fraction(rng.randrange(1_000_000_000), 1_000_000_000) * total
        acc = Fraction(0)
        pick = adj[-1][0]
        for w, s in adj:
            acc += s
            if r < acc:
                pick = w
                break
        return pick, deepest, len(scored)

    # ── counted kinship (generalisation), ported from unison_chat.py ──
    def kinship(self, a, b):
        """Exact-count similarity of two words = Jaccard over their neighbour
        distributions. High when the words keep the same company (share contexts)."""
        a, b = a.lower(), b.lower()
        na, nb = self.neigh.get(a), self.neigh.get(b)
        if not na or not nb:
            return 0.0
        keys = set(na) | set(nb)
        inter = sum(min(na.get(k, 0), nb.get(k, 0)) for k in keys)
        union = sum(max(na.get(k, 0), nb.get(k, 0)) for k in keys)
        return inter / union if union else 0.0

    def kin_expand(self, words, k=KIN_K):
        """Broaden a word set with each word's top-k counted kin. Kin candidates
        are found through SHARED contexts (the inverted index) rather than scanning
        every word, then ranked by kinship and kept above KIN_FLOOR."""
        out = set(w.lower() for w in words)
        for w in list(out):
            nb = self.neigh.get(w)
            if not nb:
                continue
            cand = set()
            for c in sorted(nb, key=lambda c: self.tok_freq.get(c, 1))[:GEN_B * CTX_MAX]:
                cand |= self.neigh_index.get(c, set())
            cand.discard(w)
            scored = sorted(((self.kinship(w, o), o) for o in cand if len(o) > 3), reverse=True)
            for sc, o in scored[:k]:
                if sc > KIN_FLOOR:
                    out.add(o)
        return out

    def kin_route(self, context_chars):
        """Kinship match — runs for EVERY input, not gated to novel ones. Kinship
        participates in every match, exactly like an embedding does; a verbatim
        input simply routes back to itself.

        Each held orbit's prompt is scored against the query by content-word
        overlap PLUS counted kin (kin at half weight, the fold factor), normalised
        by the union so the TIGHTEST match wins:

            score = (|cw ∩ ocw| + 1/2·|kin ∩ ocw|) / |cw ∪ ocw|

        A verbatim prompt scores 1.0 on its own orbit (full overlap, no extras) and
        recalls its own answer. A paraphrase scores best on the taught orbit its
        content words are kin to. Returns that orbit's prompt context (usable as the
        generation context), or None if nothing overlaps at all."""
        cw = set(_content_words(tokenize("".join(context_chars))))
        if not cw or not self.orbit_content:
            return None
        kin = self.kin_expand(cw) - cw
        best_oi, best_score = None, 0.0
        for oi, ocw in enumerate(self.orbit_content):
            if not ocw:
                continue
            direct = len(cw & ocw)
            kin_hits = len(kin & ocw)
            if not direct and not kin_hits:
                continue
            score = (direct + KIN_VOTE * kin_hits) / len(cw | ocw)
            if score > best_score:
                best_score, best_oi = score, oi
        if best_oi is None:
            return None
        return self.orbit_prompts[best_oi]


_CONTROL = {STOP, END_USER, THINK_OPEN, THINK_CLOSE}
_BOUNDARY = set(" .,!?;:")


def _emit_piece(surface, tok):
    return (" " if _needs_space(surface, tok) else "") + tok


def _spell_one_word(char_ctx, char_predict, max_spell):
    """Char tier: spell a single unknown word, char-by-char, until a word
    boundary. Returns (surface_piece, records, token) or None if the char
    engine is also stuck (so the caller can stop)."""
    piece, recs = "", []
    for _ in range(max_spell):
        c, depth, cands = char_predict(char_ctx)
        if c is None or c == STOP:
            break
        char_ctx.append(c)
        if c in _CONTROL:                 # structure token: keep in context, don't spell it
            continue
        piece += c
        recs.append((depth, cands))
        if c in _BOUNDARY:                # reached the end of the word
            break
    if not piece:
        return None
    return piece, recs, piece.strip()


def generate_multiscale(context_chars, word_sample, char_predict, rng,
                        max_chars=1_000_000, max_spell=25):
    """Generate a fragment by SAMPLING whole words from the word tier's
    unit-capacity distribution (`word_sample`), letting the words build into
    segments, and dropping to the char engine only to spell a word the word tier
    has no candidate for. Sampling — not greedy argmax — is what makes it generate
    varied, coherent text conditioned on the prompt instead of collapsing to
    one path and looping.

      WORD — sample a whole word (unit-capacity: longest held suffix, exact shares).
      SEG  — the sampled words build into segments/sentences (emergent).
      CHAR — when the word tier has no candidate at all, spell one word.

    `word_sample(words, rng)` -> (word, level, num_candidates)
    `char_predict(chars)`     -> (char, depth, num_candidates)
    Returns (surface_text, records) with one (depth, candidates) per surface char.
    """
    surface = ""
    records = []
    char_ctx = list(context_chars)
    emitted = []
    # No character limit on a response: generation runs until it reaches the
    # orbit's natural STOP, dead-ends, or a real loop is detected. `max_chars` is
    # only a runaway backstop against a non-terminating walk; it is far above any
    # real response, and long responses are chunked at the send layer.
    while len(surface) < max_chars:
        words = tokenize("".join(char_ctx))
        w, L, cands = word_sample(words, rng)

        if w == STOP:
            break

        # CHAR — the word tier has no candidate here: spell one word, then resume.
        if w is None:
            got = _spell_one_word(char_ctx, char_predict, max_spell)
            if got is None:
                break
            piece, recs, tok = got
            surface += piece
            records += recs
            if tok:
                emitted.append(tok)
            if _cycled(emitted):
                break
            continue

        # End-of-user-turn marker: keep in context, don't surface.
        if w == END_USER:
            char_ctx.append(w)
            continue

        # Thinking markers surface as-is (the reasoning trace is intentional).
        if w in (THINK_OPEN, THINK_CLOSE):
            surface += w
            char_ctx.append(w)
            continue

        # WORD — emit the sampled whole word.
        piece = _emit_piece(surface, w)
        surface += piece
        char_ctx.extend(piece)
        depth = len(" ".join(words[-L:])) if L > 0 else 0
        records += [(depth, cands)] * len(piece)
        emitted.append(w)
        if _cycled(emitted):
            break
    return surface, records


# Module singleton — one word tier over the live graph.
word_engine = WordEngine()
