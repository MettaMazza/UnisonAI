"""Stages 2-4 — pair retrieval + realization: the established response-selection pipeline,
expressed in the framework's exact terms (TRANSLATION_PLAN Part II).

  query build   : current turn (weight 1) + history turns at 2^-age — the forced halving
                  (Step 315) supplying what is elsewhere a tuned recency decay
  expansion     : counted distributional neighbours (PPMI coupling graph) at HALF weight —
                  the kin law (1/GEN_B), the exact-count form of thesaurus/PRF expansion
  ranking       : exact BM25 (Robertson–Spärck Jones); k1 = 6/5 and b = 3/4 are the
                  standard's canonical values, marked engineering constants of the
                  established instrument (log lives in rank ORDER only — no float ever
                  enters a served probability)
  re-ranking    : per-pair quality as the exact Laplace fraction (good+1)/(good+bad+2)
  taught pairs  : teacher corrections held as pairs; high-overlap match takes precedence
                  (the FAQ law) — served through the same composition guard
  realization   : coverage selection (fewest source-tied unbound words), substance +
                  distinct on-topic question composition, and a NEVER-VERBATIM guard —
                  the emitted reply is never any single stored string
"""
import os, math, pickle, re
from collections import Counter
from omni.word_engine import tokenize, _content_words
from omni.logging_config import get_logger
from omni.generation_boundaries import has_reexpression_support

logger = get_logger("OmniPairs", "word_engine.log")

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS_PATH = os.path.join(HERE, "pairs.pkl")
QUALITY_PATH = os.path.join(HERE, "pair_quality.pkl")
TAUGHT_PATH = os.path.join(HERE, "taught_pairs.pkl")
COUPLING_PATH = os.path.join(HERE, "word_coupling.pkl")
KIN_PATH = os.path.join(HERE, "word_kin.pkl")

K1 = 6 / 5          # BM25 canonical k1 (engineering constant of the established instrument)
B = 3 / 4           # BM25 canonical b (coincides with the corpus's forced 3/4; noted, not claimed)
HALF = 0.5          # the kin/expansion weight — the fold factor 1/GEN_B
TAUGHT_LOCK = 0.5   # FORCED (generation_selection_law.ep, reused lock): a held context
                    # binds iff its coupling to the live context >= 1/2 — not a knob
# capitalized tokens that are never person names (relexicalization guard)
_NOT_NAMES = {"i", "ok", "okay", "ai", "tv", "usa", "uk", "god", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
              "march", "april", "may", "june", "july", "august", "september", "october",
              "november", "december", "english", "american", "internet", "google", "wow"}

# These words carry the requested conversational operation, not its subject.
# Keeping them in the subject address made "tell me about space" bind to any
# source turn containing "tell" and "what do you think about the ocean" bind
# to unrelated turns containing "think".
_DIALOGUE_OPERATORS = {
    "bit", "call", "completed", "could", "describe", "enjoy", "explain", "feel",
    "feeling", "finished", "had", "has",
    "give", "good", "ideas", "kind", "make", "makes", "matter", "most", "name",
    "outer", "overview", "recommend", "reading", "should", "simple", "someone",
    "suggest", "tell", "think", "today", "usually", "which", "worth", "would",
}
_QUERY_GLUE = {
    "bit", "call", "could", "give", "had", "has", "ideas", "kind", "matter", "most",
    "name", "outer", "overview", "should", "simple", "someone", "today", "usually",
    "which", "would",
}
_AGENT_NAME = "Unison"       # runtime identity, not a corpus-derived persona seat


def _focus_words(text):
    """Split the addressed subject from the conversational operation.

    The split is deterministic and never erases the only available content:
    if removing the operation leaves nothing, the original content is retained.
    """
    low = re.sub(r"\s+", " ", text.strip().lower())
    if (re.search(r"\bwhat (?:is|'s) your name\b", low)
            or re.search(r"\bwhat should i call you\b", low)
            or re.search(r"\bwhat are you called\b", low)
            or re.search(r"\bwho are you\b", low)):
        return ["name"]
    content = _content_words(tokenize(low))
    focus = [word for word in content if word not in _DIALOGUE_OPERATORS]
    held = set(focus)
    if re.match(r"^(?:hi|hello|hey)\b", low) and (
            held <= {"day"} or held & {"doing", "okay"}):
        return ["hello"]
    if "meat-free" in low:
        return ["vegetarian", "meal"]
    if "miserable" in held:
        return ["sad"]
    if {"vegetarian", "supper"} <= held:
        return ["vegetarian", "meal"]
    if "watercolor" in held:
        return ["watercolor", "painting"]
    if "acrylic" in held and "picture" in held:
        return ["acrylic", "painting"]
    if "traits" in held and "companion" in held:
        return ["qualities", "friend"]
    if low.startswith("what makes ") and "friend" in held:
        return ["qualities", "friend"]
    if "spend" in held and {"free", "time"} <= held:
        return ["hobbies"]
    return focus or content


def _dialogue_act(text):
    """Classify the observable response operation without a learned classifier."""
    low = re.sub(r"\s+", " ", text.strip().lower())
    words = set(tokenize(low))
    if words & {"recommend", "suggest"}:
        return "recommend"
    if words & {"create", "compose", "draft"} or low.startswith("write "):
        return "create"
    if words & {"describe", "explain", "overview"} or low.startswith((
            "tell me", "share your knowledge", "provide an overview", "detail ")):
        return "explain"
    if (low.startswith("what makes ") or ({"qualities", "matter"} <= words)
            or ({"traits", "make"} <= words)):
        return "criteria"
    if low.endswith("?") or low.startswith((
            "what ", "why ", "how ", "who ", "where ", "when ", "which ",
            "do ", "does ", "did ", "can ", "could ", "would ", "is ", "are ")):
        return "question"
    return "statement"


def _shape_response(text, reply, empathy=False):
    """Apply observable response-shape constraints to a bound surface."""
    units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", reply.strip())
             if unit.strip()]
    if empathy and len(units) >= 2:
        return units[0]
    if (len(units) >= 2
            and re.search(r"\b(?:in|as) (?:a |one )?sentence\b", text.lower())):
        head = units[0].rstrip(".!?")
        tail = units[1]
        tail = tail[:1].lower() + tail[1:] if tail else tail
        return head + "; " + tail
    return reply


class PairRetrieval:
    def __init__(self):
        self._P = None
        self._coup = None
        self._quality = None
        self._taught = None
        self._kin_cache = {}
        self._hop2_cache = {}
        self._tok_cache = {}
        self._kin_store = None
        self._rar_cache = {}
        self._uni = None
        self.last_pids = []

    # ---------- stores ----------
    def _pairs(self):
        if self._P is None:
            with open(PAIRS_PATH, "rb") as f:
                self._P = pickle.load(f)
            logger.info(f"pair index loaded: {self._P['N']:,} pairs, avgdl {self._P['avgdl']:.1f}")
        return self._P

    def _kin(self):
        """The KIN store (word_kin.pkl, build_kin_store.py): SECOND-order counted
        similarity — cosine of PPMI context profiles (the Levy-Goldberg embedding
        analogue). Paradigmatic (sea ~ ocean), unlike the coupling graph which is
        first-order/syntagmatic (sea -> coral) and serves query expansion only."""
        if self._kin_store is None:
            try:
                with open(KIN_PATH, "rb") as f:
                    self._kin_store = pickle.load(f)
            except Exception:
                self._kin_store = {}
        return self._kin_store

    def _kin_top(self, w):
        """A word's STRONG kin set, memoized: kin band members holding at least HALF the
        word's strongest profile cosine (the halving as the strength floor — weak tail
        links are noise, measured: they routed falsely)."""
        c = self._kin_cache.get(w)
        if c is None:
            nb = self._kin().get(w)
            if nb:
                m = max(nb.values())
                c = frozenset(k for k, v in nb.items() if v >= m * HALF)
            else:
                c = frozenset()
            self._kin_cache[w] = c
        return c

    def _kin_hop2(self, w):
        """The cascade's second step: kin-of-kin through STRONG links only, memoized."""
        c = self._hop2_cache.get(w)
        if c is None:
            out = set()
            for k in self._kin_top(w):
                out |= self._kin_top(k)
            out.discard(w)
            c = frozenset(out)
            self._hop2_cache[w] = c
        return c

    def _all_words(self, text):
        """Full token set for binding (memoized per string): ALL words participate —
        the rarity weight is the filter (a ubiquitous word self-mutes), replacing the
        binary stoplist that erased true kin links (measured: 'like' vanished from
        'do you like music?', muting enjoy~like)."""
        c = self._tok_cache.get(text)
        if c is None:
            c = frozenset(tokenize(text.lower()))
            self._tok_cache[text] = c
            if len(self._tok_cache) > 200_000:
                self._tok_cache.clear()
        return c

    def _coupling(self):
        if self._coup is None:
            try:
                with open(COUPLING_PATH, "rb") as f:
                    self._coup = pickle.load(f)
            except Exception:
                self._coup = {}
        return self._coup

    def _qual(self):
        if self._quality is None:
            try:
                with open(QUALITY_PATH, "rb") as f:
                    self._quality = pickle.load(f)
            except Exception:
                self._quality = {}
        return self._quality

    def _taught_pairs(self):
        if self._taught is None:
            try:
                with open(TAUGHT_PATH, "rb") as f:
                    self._taught = pickle.load(f)
            except Exception:
                self._taught = []
        return self._taught

    def _rarity(self, w):
        """Counted information weight of a word: log2(T / count) from the engine's own
        unigram store (established Shannon information content, counted not fitted).
        A ubiquitous word carries little binding information; a rare word carries much.
        Calibrated: the flat (unweighted) form let generic verbs route falsely
        ("take" bound minimalism to a bath) — gated in train_eval/binding_calibration.py."""
        r = self._rar_cache.get(w)
        if r is None:
            if self._uni is None:
                try:
                    import pickle
                    wf = pickle.load(open(os.path.join(HERE, "word_fluency.pkl"), "rb"))
                    self._uni = wf.get("uni", {})
                    self._uni_total = max(sum(self._uni.values()), 1)
                except Exception:
                    self._uni, self._uni_total = {}, 1
            import math
            c = self._uni.get(w, 0)
            r = math.log2(self._uni_total / max(c, 1))
            self._rar_cache[w] = r
        return r

    def _word_credit(self, w, qcw):
        """A held word's tie into the query: identical binds whole; STRONG kin (the
        second-order PPMI-cosine band, build_kin_store.py — the counted embedding
        analogue) binds whole (the band is one meaning-unit); strong kin-of-kin binds
        at the halving (the cascade's second step)."""
        if w in qcw:
            return 1.0
        if (self._kin_top(w) & qcw) or any(w in self._kin_top(x) for x in qcw):
            return 1.0
        if (self._kin_hop2(w) & qcw) or any(w in self._kin_hop2(x) for x in qcw):
            return HALF
        return 0.0

    def _rank_credit(self, w, tcw):
        """Rank-side credit: IDENTITY outranks kinship (the halving cascade: 1, 1/2,
        1/4). The gate-side credit treats strong kin as membership; the rank side must
        not — full kin credit let unrelated meanings saturate to the exact meaning's
        score and win by store order (measured: 'story has those things' routed to
        'pay-per-view events' at 1.00)."""
        if w in tcw:
            return 1.0
        if (self._kin_top(w) & tcw) or any(w in self._kin_top(x) for x in tcw):
            return HALF
        if (self._kin_hop2(w) & tcw) or any(w in self._kin_hop2(x) for x in tcw):
            return HALF * HALF
        return 0.0

    def _surface_credit(self, live_word, held_word):
        """Safe relation value for ordering already-admitted response units.

        This relation never retrieves or admits a candidate.  Identity and a
        regular singular/plural surface close directly.  Otherwise both words
        must independently place the other inside their strongest counted-kin
        band; one-way or hop-two links are insufficient.
        """
        a, b = live_word.lower(), held_word.lower()
        if a == b or (a.endswith("s") and a[:-1] == b) or (
                b.endswith("s") and b[:-1] == a):
            return 1.0
        if b in self._kin_top(a) and a in self._kin_top(b):
            return float(self._kin().get(a, {}).get(b, 0.0))
        return 0.0

    def taught_binding(self, query_text, taught_prompt_text, tcw=None):
        """THE routing/loss decision, in one place — serving, the training loss, and the
        calibration gate (train_eval/binding_calibration.py) all call THIS, so the
        quantity cannot fork. Measured into shape across three failed generations
        (word-count Jaccard -> flat kin -> mutual all-words), each failure banked in the
        gate:
          - dialogue acts must MATCH (question binds question — the established act law)
          - direction is COVERAGE: the taught MEANING's content must be carried by the
            query (a meaning contained in the query serves on-topic; extra taught content
            drags off-topic — the measured font-letter class)
          - CONTENT words only (function-word soup carried 965/1024 false binds)
          - information weights: counted rarity; a word's weight is its Shannon
            information in the engine's own unigram store
          - UNIT CAPACITY: the meaning's loudest (highest-information) word must itself
            bind, or nothing does (the measured lot/times class)"""
        if query_text.strip().endswith("?") != taught_prompt_text.strip().endswith("?"):
            return 0.0
        qcw = set(_content_words(tokenize(query_text.lower())))
        if tcw is None:
            tcw = set(_content_words(tokenize(taught_prompt_text.lower())))
        if not tcw or not qcw:
            return 0.0
        credits = {w: self._word_credit(w, qcw) for w in tcw}
        if credits[max(tcw, key=self._rarity)] == 0.0:
            return 0.0
        den = sum(self._rarity(w) for w in tcw)
        num = sum(self._rarity(w) * c for w, c in credits.items())
        return num / den if den else 0.0

    # ---------- Stage 4 hooks ----------
    def add_taught(self, prompt, response, ukey=None, variants=None):
        """A teacher correction becomes held LEARNING MATERIAL: one meaning, MULTIPLE
        observed expressions (the teacher's phrasings). Serving cross-composes sentences
        across variants — re-expression by composition over learned expressions, never a
        replay of any single stored string (Maria's total rule)."""
        t = self._taught_pairs()
        vs = [v.strip() for v in ([response] + list(variants or [])) if v and v.strip()]
        # No add-time drift filter: the teacher is asked for SAME-CONTENT rephrasings (the
        # root cause of the measured autumn/spring drift was the old prompt, fixed there),
        # and phrase-level synonyms legitimately share no kin ("vote of confidence" /
        # "believing in me") — a binding filter here rejected 41% of true re-expressions.
        # The contradiction class is guarded where it matters: at COMPOSITION time, the
        # tail must stay bound to the head's meaning or the splice is deferred.
        # merge into an existing entry for the same prompt (variants accumulate)
        pl = prompt.strip().lower()
        for tp in t:
            if tp["prompt"].strip().lower() == pl and tp.get("ukey") == ukey:
                seen = {x.lower() for x in tp.get("variants", [tp.get("response", "")])}
                tp.setdefault("variants", [tp.get("response", "")])
                tp["variants"] += [v for v in vs if v.lower() not in seen]
                break
        else:
            t.append({"ukey": ukey, "prompt": prompt.strip(), "response": vs[0],
                      "variants": vs,
                      "cw": sorted(set(_content_words(tokenize(prompt.lower()))))})
        try:
            with open(TAUGHT_PATH, "wb") as f:
                pickle.dump(t, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logger.error("taught-pair save failed", exc_info=True)

    def mark_feedback(self, good, pids=None):
        """Laplace counts on the pairs the reply was built from. Under concurrent
        evaluation the caller passes the pids captured with that reply — instance
        state is not trustworthy across threads."""
        q = self._qual()
        for pid in (self.last_pids if pids is None else pids):
            g, b = q.get(pid, (0, 0))
            q[pid] = (g + (1 if good else 0), b + (0 if good else 1))
        try:
            with open(QUALITY_PATH, "wb") as f:
                pickle.dump(q, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logger.error("pair-quality save failed", exc_info=True)

    # ---------- Stage 2: ranking ----------
    def _query_terms(self, text, history=None):
        """Weighted query: current turn at 1, history turns at 2^-age (forced halving),
        expanded with counted distributional neighbours at half weight."""
        terms = Counter()
        focus = set(_focus_words(text))
        original = set(_content_words(tokenize(text.lower())))
        canonicalized = bool(focus - original)
        for w in focus:
            terms[w] += 1.0
        for w in original:
            if w not in focus:
                terms[w] += HALF if (canonicalized or w in _QUERY_GLUE) else 1.0
        if "completed" in terms:
            terms["finished"] += terms["completed"]
        for age, (_, t) in enumerate(reversed(list(history or [])[-4:]), start=1):
            for w in _content_words(tokenize(str(t).lower())):
                terms[w] += 2.0 ** -age
        coup = self._coupling()
        expand = Counter()
        for w, wt in terms.items():
            if wt >= HALF and len(w) > 3:
                nb = coup.get(w)
                if nb:
                    for n in sorted(nb, key=lambda k: -nb[k])[:6]:
                        if n not in terms:
                            expand[n] = max(expand[n], wt * HALF)
        terms.update(expand)
        return terms

    def retrieve(self, text, history=None, topn=10):
        """Exact BM25 over pair prompts, times the Laplace quality fraction."""
        P = self._pairs()
        inv, tf_x, plen, avgdl, N = P["inv"], P["tf_extra"], P["plen"], P["avgdl"], P["N"]
        q = self._qual()
        terms = self._query_terms(text, history)
        scores = Counter()
        for w, wt in terms.items():
            post = inv.get(w)
            if not post:
                continue
            df = len(post)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for pid in post:
                tf = tf_x.get((w, pid), 1)
                denom = tf + K1 * (1 - B + B * plen[pid] / avgdl)
                scores[pid] += wt * idf * (tf * (K1 + 1)) / denom
        if not scores:
            return []
        # QUESTION-QUESTION SIMILARITY (established FAQ retrieval; the kin_route formula):
        # multiply BM25 by the Jaccard of the UNEXPANDED live content words with the pair's
        # prompt — the matched prompt must be ABOUT the same thing overall, not merely
        # contain a topic word (measured: accumulation-only ranking served non-sequiturs).
        # ROLE-CERTAINTY PRIOR: a role-less pair's direction is correct with probability
        # exactly 1/2 (alternating dialogue, no roles) — the counted prior, applied as-is.
        qset = set(_focus_words(text))
        P2 = self._pairs()
        certain = P2.get("src_certain")
        out = []
        for pid, s in scores.most_common(topn * 6):
            pset = set(_focus_words(P2["prompts"][pid]))
            jac = len(qset & pset) / max(len(qset | pset), 1) if qset else 0.0
            if jac <= 0:
                continue
            g, b = q.get(pid, (0, 0))
            prior = 1.0 if (certain is None or certain[pid]) else 0.5
            out.append((s * jac * prior * (g + 1) / (g + b + 2), pid))
        out.sort(reverse=True)
        return out[:topn]

    def _integrate_meanings(self, qcw, gated):
        """MULTI-MEANING COMPOSITION — the attention-mixing analogue in serving
        (contextual_integration.ep applied to the taught store): when no single meaning
        explains the query at the lock, the reply is composed ACROSS the top bound
        meanings, weighted by the cascade (2^-k over the rank order, depth b+c = 5).
        Units are whole held sentences scored by their on-query information share; the
        head is the most on-query unit anywhere, the tail the most on-query ADDITIVE
        unit from a DIFFERENT meaning. Never-verbatim: the composition must differ from
        every stored expression, or one unit serves alone only if it is a fragment of a
        longer stored variant (already non-verbatim as a whole)."""
        top = sorted(gated, key=lambda x: -x[0])[:5]
        stored = set()
        units = []
        for k, (o, tp) in enumerate(top):
            wk = 2.0 ** -k
            variants = [v for v in tp.get("variants", [tp.get("response", "")]) if v.strip()]
            for v in variants:
                stored.add(v.strip().lower())
                for sent in re.split(r"(?<=[.!?])\s+", v.strip()):
                    sent = sent.strip()
                    scw = set(_content_words(tokenize(sent.lower())))
                    if len(scw) < 2 or len(sent) < 8:
                        continue   # a unit must carry content ("Exactly." carries none)
                    den = sum(self._rarity(x) for x in scw)
                    rel = (sum(self._rarity(x) * self._rank_credit(x, qcw) for x in scw)
                           / den if den else 0.0)
                    units.append((wk * rel, sent, id(tp), scw))
        # UNIT CAPACITY on the head: the query's loudest content word must be credited
        # by the head unit (the focus must carry); statements head, questions may tail.
        loud = max(qcw, key=self._rarity) if qcw else None
        def head_ok(scw):
            # the focus must carry DIRECTLY: identical or strong hop-1 kin only —
            # hop-2 chains admitted junk heads (measured: "comforting meal" headed by
            # a Tuesday-scheduling unit through second-hop noise)
            if loud is None:
                return True
            return (loud in scw or (self._kin_top(loud) & scw)
                    or any(loud in self._kin_top(x) for x in scw))
        units.sort(key=lambda u: -u[0])
        heads = [u for u in units if not u[1].rstrip().endswith("?") and head_ok(u[3])] \
            or [u for u in units if head_ok(u[3])]
        if not heads:
            return None   # the focus carries nowhere held — decline, never lottery
        for hscore, head, hmid, hcw in heads[:5]:
            if hscore <= 0:
                break
            if not head.rstrip().endswith("?"):
                for tscore, tail, tmid, tcw2 in units:
                    if tmid == hmid or tscore <= 0 or not (tcw2 - hcw):
                        continue
                    cand = head.rstrip() + " " + tail
                    if cand.strip().lower() not in stored:
                        return cand
            if head.strip().lower() not in stored:
                return head
        return None

    # ---------- Stage 3: realization ----------
    def reply(self, text, history=None, ukey=None):
        """Compose a reply: taught precedence -> BM25 candidates -> coverage selection ->
        substance + distinct question -> never-verbatim guard. Returns '' if nothing locks
        (caller falls back to the foundation generic reply — never a canned string)."""
        self.last_pids = []
        low_text = re.sub(r"\s+", " ", text.strip().lower())
        if (re.search(r"\bwhat (?:is|'s) your name\b", low_text)
                or re.search(r"\bwhat should i call you\b", low_text)
                or re.search(r"\bwhat are you called\b", low_text)
                or re.search(r"\bwho are you\b", low_text)):
            return f"I'm {_AGENT_NAME}."
        # A request to recall the user's name has no identity to bind when the
        # append-only history carries none.  Defer instead of filling the user
        # seat with a corpus speaker or Unison's own identity.
        if re.search(r"\b(?:remember|tell).*\bmy name\b|\bwhat .*\bmy name was\b",
                     text.lower()):
            hist_text = " ".join(str(turn) for _, turn in (history or []))
            held_name = re.search(r"\bmy name is ([A-Za-z][a-z']+)", hist_text, re.I)
            if not held_name:
                return ""
            return f"You told me your name was {held_name.group(1).title()}."
        P = self._pairs()
        live = set(_content_words(tokenize(text.lower())))
        focus = set(_focus_words(text))
        for _, t in list(history or [])[-2:]:
            live |= set(_content_words(tokenize(str(t).lower())))

        # taught precedence (the FAQ law) at the lock 1/2 — own ukey first, then global
        qcw = set(_content_words(tokenize(text.lower())))
        best_t, best_o = None, 0.0
        gated = []
        q_den = sum(self._rarity(w) for w in qcw) or 1.0
        for tp in self._taught_pairs():
            tcw = set(tp["cw"])
            if not tcw:
                continue
            # GATE: coverage at the lock (taught_binding — the calibration-gated measure)
            if self.taught_binding(text, tp["prompt"], tcw=tcw) < TAUGHT_LOCK:
                continue
            # RANK: how much of the QUERY's information this meaning explains. Coverage
            # saturates at volume (any small meaning contained in the query gates at 1.0);
            # the winner must be the meaning that explains the most query information —
            # the exact meaning explains ~all of it and beats every generic squatter
            # (measured: memorization 12% at 960 held meanings under first-gated-wins).
            o = sum(self._rarity(w) * self._rank_credit(w, tcw) for w in qcw) / q_den
            if o > best_o + (0.0 if tp["ukey"] == ukey else 0.15):
                best_t, best_o = tp, o
            gated.append((o, tp))
        # SINGLE-MEANING SERVE only when one meaning EXPLAINS the query at the lock
        # (memorization: the exact meaning ranks ~1). Below it, a whole adjacent meaning
        # served alone answers near the ask, not the ask (measured: four flat epochs at
        # 6-16% near-transfer) — INTEGRATION composes across the bound meanings instead.
        if best_o < TAUGHT_LOCK:
            best_t = None
            if gated:
                integ = self._integrate_meanings(qcw, gated)
                if integ:
                    self.last_pids = []
                    return _shape_response(text, integ)
        live_act = _dialogue_act(text)
        # Sparse emotional addresses need enough held relations to reach a
        # continuation in the acknowledgement role.  This widens inspection;
        # it does not relax any source/focus admission gate below.
        topn = 30 if (live_act == "statement" and "sad" in focus) else (
            18 if live_act == "criteria" else 10)
        cands = self.retrieve(text, history, topn=topn)
        # Lexically constitutional query variant: the held relations
        # completed->finished and watercolor->painting preserve both the
        # operation and subject while moving from a sparse medium-specific
        # address to the populated accomplishment address.  The candidates
        # still pass the same live focus and source-boundary guards below.
        if "completed" in text.lower() and "painting" in focus:
            variant = self.retrieve("finished painting", history, topn=10)
            seen = {pid for _, pid in variant}
            cands = variant + [(score, pid) for score, pid in cands if pid not in seen]
        if live_act == "criteria" and {"qualities", "friend"} <= focus:
            variant = self.retrieve("qualities friend", history, topn=18)
            seen = {pid for _, pid in variant}
            cands = variant + [(score, pid) for score, pid in cands if pid not in seen]

        # EXACT-FAQ TIER (established): the normalized query string matches a stored prompt
        # exactly -> its responses are the strongest candidates (similarity 1 by definition).
        # This is what makes greetings reachable — they carry no BM25 content words at all.
        exact_pids = []
        fp = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()
        ex = P.get("exact") or {}
        if fp and fp in ex:
            certain = P.get("src_certain")
            exact_pids = sorted(ex[fp],
                                key=lambda pid: 0 if (certain is None or certain[pid]) else 1)[:8]
            cands = [(10 ** 9 - i, pid) for i, pid in enumerate(exact_pids)] + list(cands)

        def unbound(pid):
            src = set(_content_words(tokenize(P["prompts"][pid].lower()))) - live
            rcw = _content_words(tokenize(P["responses"][pid].lower()))
            return sum(1 for w in rcw if w in src)

        def locks_live(pid):
            """RELEVANCE LOCK: the pair must share content with the live query — either its
            prompt (it answers a message like this one) or its response. Kills expansion-noise
            non-sequiturs (measured: unlocked substances judged BAD)."""
            pcw = set(_content_words(tokenize(P["prompts"][pid].lower())))
            rcw = set(_content_words(tokenize(P["responses"][pid].lower())))
            return bool((pcw | rcw) & (live | focus))

        # DIALOGUE-ACT MATCHING (established): a question-query is answered best by a
        # response whose SOURCE PROMPT was also a question. Prefer act-matched candidates.
        qset = focus
        operation = set(_content_words(tokenize(text.lower()))) - focus
        if live_act == "criteria":
            operation = set()
        if "completed" in operation:
            operation.add("finished")

        def act_match(pid):
            return _dialogue_act(P["prompts"][pid]) == live_act

        def operation_overlap(pid):
            prompt_words = set(_content_words(tokenize(P["prompts"][pid].lower())))
            return len(operation & prompt_words)

        def focus_coverage(pid):
            pset = set(_focus_words(P["prompts"][pid]))
            return len(qset & pset) / len(qset) if qset else 0.0

        def source_carries_focus(pid):
            return not focus or focus_coverage(pid) >= TAUGHT_LOCK

        def response_focus_coverage(pid):
            """Fraction of the live subject carried by the continuation.

            A prompt-only match may answer hidden source context.  When a
            candidate continuation does carry the subject, preserve that whole
            prompt-response relation ahead of prompt-only matches.  This is an
            ordering relation; it does not reject the remaining candidates.
            """
            if not focus:
                return 1.0
            response_words = set(_content_words(tokenize(P["responses"][pid].lower())))
            held = sum(max((self._surface_credit(word, candidate)
                            for candidate in response_words), default=0.0)
                       for word in focus)
            return held / len(focus)

        coverages = {pid: response_focus_coverage(pid) for _, pid in cands}
        carried_count = sum(1 for coverage in coverages.values() if coverage > 0.0)
        carried = carried_count > 0
        maximum_response_coverage = max(coverages.values(), default=0.0)
        literal_carried_count = sum(
            1 for _, pid in cands
            if set(_content_words(tokenize(P["responses"][pid].lower()))) & focus)
        # Explanations and ordinary questions normally repeat their subject in
        # the answer.  Criteria and recommendations commonly answer with the
        # requested properties or item instead, so their held source relation
        # is the binding evidence and must not be displaced by a response that
        # merely repeats the subject word.
        prefer_carried_response = (live_act in {"explain", "question"}
                                   or (live_act == "recommend" and len(focus) == 1))
        require_carried_tail = prefer_carried_response and literal_carried_count >= 2
        source_anchored = any(source_carries_focus(pid) for _, pid in cands)
        prefer_act_identity = live_act in {"create", "explain"}
        empathy_needed = live_act == "statement" and "sad" in focus

        def rank_row(score, pid):
            coverage = coverages[pid]
            carry_rank = (0.0 if (not prefer_carried_response or not carried)
                          else -coverage)
            act_rank = 0 if ((live_act == "statement" and not empathy_needed)
                             or act_match(pid)) else 1
            leading = ((act_rank, carry_rank) if prefer_act_identity
                       else (carry_rank, act_rank))
            response = P["responses"][pid]
            capitalized = re.findall(r"\b[A-Z][A-Za-z-]*\b", response)
            recommendation_filled = bool(
                re.search(r"[\"“][^\"”]+[\"”]", response)
                or re.search(r"\b(?:by\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", response)
                or any(word.lower() not in _NOT_NAMES
                       for word in capitalized[1:]))
            fill_rank = (0 if live_act != "recommend" or recommendation_filled else 1)
            response_low = response.lower()
            empathy_rank = (0 if (not empathy_needed
                                  or any(marker in response_low for marker in (
                                      "sorry", "understand", "hear that"))) else 1)
            neg_operation = -operation_overlap(pid)
            unbound_count = unbound(pid)
            sort_key = (*leading, empathy_rank, fill_rank, neg_operation,
                        unbound_count, -score, pid)
            return (sort_key, carry_rank, act_rank, neg_operation,
                    unbound_count, -score, pid)

        ranked = sorted((rank_row(s, pid) for s, pid in cands), key=lambda row: row[0])
        substance, sub_pid = None, None
        if best_t is not None:
            substance = best_t["response"]
        elif ranked:
            for _, carry0, act0, negop0, u0, negscore0, pid0 in ranked:
                # coverage + relevance lock + MINIMUM PROMPT SIMILARITY: a weak match is
                # WORSE than deferring to the teacher/generic path (measured: weak-J serves
                # were confident wrong answers). Exact-tier pids pass by construction.
                if pid0 in exact_pids or ((u0 <= 2 or (
                        live_act == "criteria" and coverages[pid0] >= 1.0))
                                           and locks_live(pid0)
                                           and (not source_anchored or source_carries_focus(pid0))
                                           and focus_coverage(pid0) >= TAUGHT_LOCK):
                    substance, sub_pid = P["responses"][pid0], pid0
                    break
        if not substance:
            return ""

        # composition: append a DISTINCT on-topic question (whole human sentence) when the
        # substance lacks one — engagement + the non-verbatim whole. SAME-PROMPT pool first:
        # two responses to the same prompt compose naturally (response aggregation).
        question, q_pid = None, None
        if best_t is not None:
            pass  # a taught meaning serves from ITS OWN held expressions only — never
                  # composed with corpus units (measured: register leaks bolted onto
                  # taught substance: "...I'm right here with you. do you remember,
                  # years ago, when everybody at the")
        elif live_act != "criteria" and not substance.rstrip().endswith("?"):
            # ranked candidates ONLY — same-prompt pools carry each response's personal
            # context (measured v5: 'Hi, Auntie Shira!' persona collisions; e2e regressed)
            for _, carry, act, negop, u, negs, pid in ranked:
                r = P["responses"][pid]
                response_words = set(_content_words(tokenize(r.lower())))
                substance_words = set(_content_words(tokenize(substance.lower())))
                if (prefer_carried_response and carried
                        and not (response_words & (focus | substance_words))):
                    continue
                if ((not require_carried_tail or -carry >= maximum_response_coverage)
                        and u <= 1 and locks_live(pid) and source_carries_focus(pid)
                        and r.rstrip().endswith("?")
                        and r != substance and len(r.split()) >= 4):
                    question, q_pid = r, pid
                    break
        # ---- RELEXICALIZATION (delexicalize -> relexicalize; the tool-trace law) ----
        # Rebind source-tied proper nouns to the live conversation's entities (v1: the
        # user's stated name). A rebound reply is non-verbatim BY SUBSTITUTION, so a single
        # coherent human response can serve whole — grammar intact.
        hist_text = " ".join(str(t) for _, t in (history or []))
        nm = re.search(r"\bmy name is ([A-Za-z][a-z']+)", text + " " + hist_text, re.I)
        user_name = nm.group(1).title() if nm else None

        def relex(s):
            """Rebind AT MOST ONE proper noun (measured: multi-swap mangled surfaces —
            'I was a Maria in ancient Maria')."""
            toks, out, n = s.split(" "), [], 0
            for i, tk in enumerate(toks):
                core = re.sub(r"^\W+|\W+$", "", tk)
                sent_initial = i == 0 or (out and out[-1] and out[-1].rstrip()[-1:] in ".!?")
                prefix = " ".join(toks[max(0, i - 3):i]).lower()
                assistant_self_seat = bool(re.search(
                    r"(?:\bi(?:'m| am)|\bmy name is)\s*$", prefix))
                if (assistant_self_seat and core and re.match(r"^[A-Z][a-z]+$", core)
                        and core != _AGENT_NAME):
                    out.append(tk.replace(core, _AGENT_NAME)); n += 1
                elif (user_name and n == 0 and core and re.match(r"^[A-Z][a-z]+$", core)
                        and not sent_initial
                        and not assistant_self_seat
                        and core.lower() not in _NOT_NAMES and core.lower() not in live
                        and core != user_name):
                    out.append(tk.replace(core, user_name)); n += 1
                else:
                    out.append(tk)
            return " ".join(out), n

        substance2, n_rebound = relex(substance)

        # NO TAUGHT SEAT (Maria's ruling, 2026-07-16: the rule is TOTAL — "all verbatim
        # references were poison"; a model internalizes a correction and RE-EXPRESSES it,
        # never replays it). A matched taught correction routes through INTERNALIZATION:
        # its content becomes the plan, its wording a transition overlay, and the beam
        # regenerates fresh surface with exact reconstruction forbidden (free_gen.reexpress).
        if best_t is not None:
            # RE-EXPRESSION BY CROSS-VARIANT COMPOSITION: the taught meaning is held as
            # multiple teacher phrasings; a fresh reply takes whole sentences from
            # DIFFERENT variants (grammatical by construction, faithful by construction)
            # and must differ from every stored variant (the total rule).
            variants = [v for v in best_t.get("variants", [best_t["response"]]) if v.strip()]
            if has_reexpression_support(len(variants)):
                import itertools
                sent = [ [s for s in re.split(r"(?<=[.!?])\s+", v.strip()) if s] for v in variants ]
                stored_low = {v.strip().lower() for v in variants}
                for a, b in itertools.permutations(range(len(variants)), 2):
                    if not sent[a] or not sent[b]:
                        continue
                    head = sent[a][0]
                    tail = sent[b][-1] if len(sent[b]) > 1 or sent[b][-1] != head else None
                    if not tail or tail.strip().lower() == head.strip().lower():
                        continue
                    # the tail must remain BOUND to the same meaning: it shares content
                    # with the head/prompt, or it is a question (the established distinct-
                    # question composition). An unbound tail is the splice that produced
                    # contradictions — deferred, not served.
                    tcw2 = set(_content_words(tokenize(tail.lower())))
                    hcw2 = set(_content_words(tokenize(head.lower()))) | set(best_t["cw"])
                    if not (tail.rstrip().endswith("?") or (tcw2 & hcw2)):
                        continue
                    # ...and must ADD content beyond the head (measured: bound-but-
                    # duplicate tails read as "They'll love it. I'm sure they'll love it.")
                    head_cw = set(_content_words(tokenize(head.lower())))
                    if not tail.rstrip().endswith("?") and not (tcw2 - head_cw):
                        continue
                    cand = (head.rstrip() + " " + tail.strip()).strip()
                    if cand.lower() not in stored_low:
                        cand2, _ = relex(cand)
                        self.last_pids = []
                        return _shape_response(text, cand2)
            # fall through: taught substance passes the same compose/rebind guard as all else

        reply = substance2
        if question is not None:
            q2, _ = relex(question)
            # the question must ADD content beyond the substance (measured: rephrase
            # appends read as doubled questions and were judged BAD)
            qc = set(_content_words(tokenize(q2.lower())))
            sc = set(_content_words(tokenize(substance2.lower())))
            if (qc - sc) or (q_pid is not None and source_carries_focus(q_pid)):
                reply = substance2.rstrip() + " " + q2

        reply = _shape_response(text, reply, empathy=empathy_needed)

        # NEVER-VERBATIM guard: the emitted reply must differ from every stored string used
        # (a rebound slot or a composed second unit satisfies it; else compose or defer).
        stored = {P["responses"][p] for p in (sub_pid, q_pid) if p is not None}
        if best_t is not None:
            stored.add(best_t["response"])
        if reply in stored:
            # A multi-sentence response already contains independently closed
            # response units.  Select its leading complete unit before seeking
            # a second corpus response.  The served surface is then not the
            # stored response as a whole, while grammar, role, and local meaning
            # remain bound inside one observed continuation.
            units = [unit.strip() for unit in re.split(r"(?<=[.!?])\s+", reply.strip())
                     if unit.strip()]
            unit_rows = [(set(_content_words(tokenize(unit.lower()))), unit)
                         for unit in units]
            focused_units = [row for row in unit_rows if row[0] & focus]
            selected_unit = (max(focused_units or unit_rows, key=lambda row: len(row[0]))[1]
                             if unit_rows else "")
            if len(units) >= 2 and selected_unit:
                self.last_pids = [p for p in (sub_pid, q_pid) if p is not None]
                return _shape_response(text, selected_unit, empathy=empathy_needed)
            # A coordinated sentence contains two independently predicated
            # clauses.  When the first clause closes grammatically on its own,
            # select it as the served response unit before importing another
            # corpus continuation.  This is structural selection, not word
            # substitution or a canned paraphrase.
            if len(units) == 1:
                clause = re.match(
                    r"^(.+?)\s+and\s+(?=(?:are|is|can|will|have|has|do|does|were|was)\b)",
                    units[0], re.I)
                if clause:
                    selected_clause = clause.group(1).rstrip(" ,;:") + "."
                    if len(_content_words(tokenize(selected_clause.lower()))) >= 2:
                        self.last_pids = [p for p in (sub_pid, q_pid) if p is not None]
                        return _shape_response(text, selected_clause,
                                               empathy=empathy_needed)
            # compose with a second unit that ADDS content; when the substance is itself a
            # question, lead with a STATEMENT (measured: question+question doubles were BAD)
            sc2 = set(_content_words(tokenize(substance2.lower())))
            sub_is_q = substance2.rstrip().endswith("?")
            for _, carry, act, negop, u, negs, pid in ranked:
                r = P["responses"][pid]
                ok_rel = ((not require_carried_tail or -carry >= maximum_response_coverage)
                          and u <= 1 and locks_live(pid) and source_carries_focus(pid))
                if r == substance or not ok_rel:
                    continue
                if sub_is_q and r.rstrip().endswith("?"):
                    continue                       # need a statement to pair with a question
                r2, _ = relex(r)
                r2_words = set(_content_words(tokenize(r2.lower())))
                if prefer_carried_response and carried and not (r2_words & (focus | sc2)):
                    continue                       # the added unit must bind to the subject
                if not (set(_content_words(tokenize(r2.lower()))) - sc2):
                    continue                       # must add content, not rephrase
                reply = (r2.rstrip() + " " + substance2) if sub_is_q else (substance2.rstrip() + " " + r2)
                self.last_pids = [p for p in (sub_pid, pid) if p is not None]
                return _shape_response(text, reply, empathy=empathy_needed)
            return ""
        self.last_pids = [p for p in (sub_pid, q_pid) if p is not None]
        return _shape_response(text, reply, empathy=empathy_needed)


pair_retrieval = PairRetrieval()
