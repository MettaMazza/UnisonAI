"""F1 — the free generator: kin-context mixing (FRONTIER_PLAN, kNN-LM translated).

Generates novel surface word-by-word. At each step the current context retrieves its
nearest stored contexts from the kin datastore (counted overlap over the trailing content
window, idf-weighted); their recorded NEXT words form the retrieved distribution; positions
whose exact last token matches carry the binary factor 2 (the deeper tier — the same forced
halving/cascade shape as Rung 5e, applied at generation). That distribution is mixed with
the local fluency counts. Retrieval contributes DISTRIBUTIONS, never surface strings — the
output is composed token-by-token and is non-verbatim by construction.

Topic steering (F2, minimal v1): the query's content words and their top coupling
neighbours form the topic set; content-word candidates outside it are halved once (2^-1) —
a counted prior, the trigger-LM translation. Never used as a judge.
"""
import os, pickle, re
from collections import Counter
from omni.word_engine import word_engine, tokenize, _content_words
from omni.logging_config import get_logger

logger = get_logger("OmniFreeGen", "word_engine.log")
HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "kin_context.pkl")

TOPK_POS = 300          # retrieved positions per step
MAX_WORDS = 36
MIN_WORDS = 6
_END = {".", "!", "?"}


_PUNC = {".", ",", "!", "?", "'", ";", ":", '"', ")", "(", "-"}


class FreeGen:
    def __init__(self):
        self._S = None
        self._flu_cache = {}
        self._ls_cache = {}
        self._hits_cache = {}

    def _fluent_fast(self, window):
        """Float-ranked fluency cascade for the GENERATION hot path (ranking order only —
        never a served probability). The exact-Fraction `_scored_fluent` was measured at
        678 ms/call on an M3 Ultra (thousands of Fraction constructions per call, uncached,
        while beams share prefixes) — the latency was the tell of the defect, not the
        machine. Cached per window; candidates capped to the deepest levels' top counts."""
        key = tuple(window)
        hit = self._flu_cache.get(key)
        if hit is not None:
            return hit
        we = word_engine
        we._load_fluency()
        ctx = [w.lower() for w in window]
        out = {}
        levels_used = 0
        for L in range(min(len(ctx), 4), 0, -1):
            m = we._merged_level(tuple(ctx[-L:]), L)
            if not m:
                continue
            total = float(sum(m.values())) or 1.0
            w2 = float(2 ** L)
            items = m.most_common(80) if hasattr(m, "most_common") else \
                sorted(m.items(), key=lambda kv: -kv[1])[:80]
            for w, c in items:
                out[w] = out.get(w, 0.0) + w2 * (c / total)
            levels_used += 1
            deepest = L if levels_used == 1 else deepest
            if levels_used >= 2 and L <= 1:
                break                      # unigram tail adds noise, not signal, past depth
        res = (sorted(out.items(), key=lambda kv: -kv[1])[:64],
               (deepest if levels_used else 0))
        if len(self._flu_cache) > 20000:
            self._flu_cache.clear()
        self._flu_cache[key] = res
        return res

    def _store(self):
        if self._S is None:
            with open(STORE, "rb") as f:
                self._S = pickle.load(f)
            logger.info(f"kin-context store loaded: {len(self._S['cond']):,} conditioned keys")
        return self._S

    _CONTR_OK = {"t": None, "s": None, "d": None, "ll": None, "ve": None, "re": None, "m": None}

    def _junk_token(self, w):
        """Token-pipeline junk, blocked at EVERY emission site: bare quote tokens
        (rendered "That' m") and typo-merged contractions ("would't", "i're")."""
        return w in {"'", "\u2019"} or ("'" in w and not self._valid_contraction(w))

    def _valid_contraction(self, w):
        """A merged contraction is valid only in its English pattern — corpus typos
        ("would't", "passengers'm") merged into servable junk (measured audit)."""
        if w.count("'") != 1:
            return False
        h, t = w.split("'")
        if not h.isalpha():
            return False
        if t == "t":
            return h.endswith("n") or h in {"can", "won", "shan"}
        if t == "m":
            return h == "i"
        if t == "re":
            return h in {"you", "we", "they", "who", "there", "what", "how", "where", "when"}
        if t in {"s", "d", "ll", "ve"}:
            return True
        return t == ""

    def _live_state_ids(self, base_state_ids, gcw, vocab):
        """THE AUTOREGRESSIVE STATE (completing the contextual_integration translation):
        a transformer conditions on everything generated so far; here the words the
        engine has EMITTED diffuse into the integrated state through the same forced
        cascade — at the fold factor HALF, because own unconfirmed output is held at
        reduced confidence (the retention law, reused). The query remains the
        full-weight anchor; the live thread gains mass; competing corpus threads decay
        relatively (the measured thread-jumping class)."""
        cached = self._ls_cache.get(gcw)
        if cached is None:
            from omni.context_state import integrated_state
            g = integrated_state(" ".join(gcw), None)
            cached = {}
            for w, m in g.items():
                nid = vocab.get(w)
                if nid is not None:
                    cached[nid] = m
            if len(self._ls_cache) > 4096:
                self._ls_cache.clear()
            self._ls_cache[gcw] = cached
        if not cached:
            return base_state_ids
        merged = dict(base_state_ids)
        for nid, m in cached.items():
            merged[nid] = merged.get(nid, 0.0) + 0.5 * m
        return merged

    def _closures(self, S):
        """F4 FEEDBACK: teacher-CLOSED own generations (omni/free_closures.pkl) as a
        conditioning overlay — verified own-output is retained and reinforced (the
        retention law: only closed output re-enters; raw output never self-reinforces).
        Closed texts count at 2^2 (closed + own — two doublings; engineering factor,
        marked). Tables rebuilt when the closure store changes."""
        import os as _os, time as _time
        now = _time.monotonic()
        if now - getattr(self, "_clo_checked", 0.0) < 30.0:
            return getattr(self, "_clo_tables", None)
        self._clo_checked = now
        path = getattr(self, "_clo_path", None)
        if path is None:
            path = self._clo_path = _os.path.join(
                _os.path.dirname(_os.path.abspath(__file__)), "free_closures.pkl")
        try:
            mt = _os.path.getmtime(path)
        except OSError:
            self._clo_mtime, self._clo_tables = None, None
            return None
        if getattr(self, "_clo_mtime", None) == mt:
            return self._clo_tables
        import pickle as _pickle
        from collections import Counter as _C
        try:
            held = _pickle.load(open(path, "rb"))
        except Exception:
            return None
        vocab = S["vocab"]
        c2c, c3c = {}, {}
        for row in held:
            toks = [t.lower() for t in tokenize(row.get("closed", ""))]
            ids = [vocab.get(t, -1) for t in toks]
            tops = {vocab[w] for w in _content_words(toks) if w in vocab}
            for i in range(1, len(ids)):
                if ids[i] < 0 or ids[i - 1] < 0:
                    continue
                pv = ids[i - 2] if i >= 2 and ids[i - 2] >= 0 else None
                for tid in tops:
                    c2c.setdefault((ids[i - 1], tid), _C())[ids[i]] += 4
                    if pv is not None:
                        c3c.setdefault((pv, ids[i - 1], tid), _C())[ids[i]] += 4
        self._clo_mtime, self._clo_tables = mt, (c2c, c3c)
        return self._clo_tables

    def _topic_hits(self, S, prev_id, last_id, topic_ids, clo):
        """All topics with counts for this (prev, last) pair, memoized per pair — the
        measured hot path was millions of EMPTY per-topic lookups (57s/reply)."""
        key = (prev_id, last_id)
        hits = self._hits_cache.get(key)
        if hits is None:
            hits = []
            for tid in topic_ids:
                c = self._tier_counts(S, prev_id, last_id, tid, clo)
                if c:
                    hits.append((tid, c))
            self._hits_cache[key] = hits
        return hits

    def _tier_counts(self, S, prev_id, last_id, tid, clo=None):
        """Conditioned counts across n-gram tiers: the bigram tier plus the TRIGRAM tier
        at the binary factor 2 — two exact ordered tokens in the key outrank one, the
        same deeper-context-doubles tiering the fluency cascade and topic weighting use
        (F-lever 1: order lives in the key, established n-gram deepening). The F4
        closure overlay (verified own-output) joins both tiers at its own factor."""
        c2 = S["cond"].get((last_id, tid))
        c3 = S.get("cond3", {}).get((prev_id, last_id, tid)) if prev_id >= 0 else None
        m = dict(c2) if c2 else {}
        if c3:
            for nid, n in c3.items():
                m[nid] = m.get(nid, 0) + 2 * n
        if clo:
            c2c, c3c = clo
            e2 = c2c.get((last_id, tid))
            if e2:
                for nid, n in e2.items():
                    m[nid] = m.get(nid, 0) + n
            e3 = c3c.get((prev_id, last_id, tid)) if prev_id >= 0 else None
            if e3:
                for nid, n in e3.items():
                    m[nid] = m.get(nid, 0) + 2 * n
        return m or None

    def _topic_set(self, text, history=None):
        base = set(_content_words(tokenize(text.lower())))
        for _, t in list(history or [])[-2:]:
            base |= set(_content_words(tokenize(str(t).lower())))
        coup = word_engine._load_coupling() or {}
        topic = set(base)
        # expansion TRIMMED to top-3 per base word (v2 defect: wide expansion drifted the
        # topic — ocean -> parmesan through neighbour chains)
        for w in base:
            nb = coup.get(w)
            if nb:
                topic.update(sorted(nb, key=lambda k: -nb[k])[:3])
        return base, topic

    def generate(self, text, history=None, rng=None, max_words=MAX_WORDS):
        import random
        rng = rng or random.Random()
        S = self._store()
        vocab, words_arr, cond = S["vocab"], S["words"], S["cond"]
        base, topic = self._topic_set(text, history)
        base_ids = {vocab[w] for w in base if w in vocab}
        topic_ids = {vocab[w] for w in topic if w in vocab}
        self._hits_cache = {}

        out = []                      # generated tokens (lowercase)
        # seed: open with a topical corpus opener — sample the first word from the
        # conditioned table under a sentence-start proxy (the query's strongest topic word)
        for step in range(max_words):
            # sentence-start conditioning: the BOS marker keys topical corpus OPENERS
            last_tok = out[-1] if out else "\x02"
            last_id = vocab.get(last_tok, -1)
            prev_tok = out[-2] if len(out) >= 2 else "\x02"
            prev_id = vocab.get(prev_tok, -1)
            dist = Counter()
            clo = self._closures(S)
            if last_id >= 0:
                # TOPIC-CONDITIONED lookup, via the memoized per-(prev,last) hit list
                for tid, c in self._topic_hits(S, prev_id, last_id, topic_ids, clo):
                    w_t = 2 if tid in base_ids else 1
                    for nid, n in c.items():
                        dist[nid] += n * w_t
            # mix the local fluency substrate (deeper exact n-grams outrank, 2^L inside)
            window = ([w for w in base][:WINDOW_SEED] + out)[-6:]
            scored, _ = self._fluent_fast(window) if window else ([], 0)
            fl = Counter()
            for w, s in scored or []:
                nid = vocab.get(w.lower())
                if nid is not None:
                    fl[nid] += float(s)
            if fl:
                tot_f = sum(fl.values())
                tot_d = sum(dist.values()) or 1
                for nid, v in fl.items():
                    dist[nid] += (v / tot_f) * tot_d   # equal-mass interpolation (lambda 1/2)
            if not dist:
                break
            # structural + junk guards (this arm sampled UNFILTERED — audit find)
            for nid in list(dist):
                w = words_arr[nid]
                if self._junk_token(w) or (not out and w in _PUNC) or (
                        out and out[-1] in _PUNC and w in _PUNC):
                    del dist[nid]
            if not dist:
                break
            # sample from the top of the mixed distribution
            items = dist.most_common(24)
            total = sum(c for _, c in items)
            r = rng.random() * total
            acc, pick = 0.0, items[0][0]
            for nid, c in items:
                acc += c
                if r < acc:
                    pick = nid
                    break
            w = words_arr[pick]
            out.append(w)
            if w in _END and len(out) >= MIN_WORDS:
                break

        # surface assembly: sentence case + spacing
        s = ""
        for w in out:
            if not s:
                s = w.capitalize()
            elif w in {".", ",", "!", "?", "'", ";", ":"} or w.startswith("'"):
                s += w
            else:
                s += " " + w
        s = re.sub(r"\s+([.,!?;:])", r"\1", s).strip()
        return s


    # ---------------- F3: plan -> realize (constrained decoding) ----------------
    def _plan(self, text, history=None, topn=5):
        """The PLAN: what humans say in reply to messages like this — the top kin
        responses' content words (weighted by rank) + the dialogue act. Meaning only,
        never surface (established content planning)."""
        from omni.pair_retrieval import pair_retrieval
        P = pair_retrieval._pairs()
        cands = pair_retrieval.retrieve(text, history, topn=topn)
        plan = Counter()
        acts_q = 0
        for rank, (s, pid) in enumerate(cands):
            r = P["responses"][pid]
            if r.rstrip().endswith("?"):
                acts_q += 1
            for w in _content_words(tokenize(r.lower())):
                plan[w] += 1.0 / (1 + rank)
        want_q = acts_q * 2 >= max(len(cands), 1)          # majority act at the lock
        return [w for w, _ in plan.most_common(6)], want_q

    def generate_planned(self, text, history=None, rng=None, n_best=5):
        """F3-v2: constrained BEAM realization (deterministic coverage + log-count
        maximization — established constrained decoding; replaces sampled best-of-8, whose
        soft boost could not hold the plan). The judge never selects, only scores."""
        import random, math
        rng = rng or random.Random()
        plan, want_q = self._plan(text, history)
        S = self._store()
        vocab = S["vocab"]
        plan_ids = {vocab[w] for w in plan if w in vocab}
        best, best_score = "", -1e18
        # a couple of beam runs with different tie-noise seeds guard against one bad opener
        for k in range(max(1, n_best)):
            cand, sc = self._beam(text, history, plan_ids, want_q,
                                  random.Random(rng.randrange(1 << 30)))
            if cand and sc > best_score:
                best, best_score = cand, sc
        return best

    def reexpress(self, taught_text, query, history=None, rng=None):
        """INTERNALIZE-AND-RE-EXPRESS a taught correction (Maria's total rule: a model
        learns from a correction and regenerates — never replays, never shreds).

        v2 — DISTRIBUTIONAL PARAPHRASE (established): substitute 1-2 content words with
        their strongest SECOND-ORDER kin (words with similar neighbour distributions —
        `word_engine.kinship`, the counted substitutability measure; first-order
        co-occurrence is association, not substitutability). Grammar stays intact because
        substitutions are in-place; meaning holds because kin substitutes are
        distributionally interchangeable; non-verbatim by substitution. (v1 fed the
        correction's bigrams to the beam, which SHREDDED it — "Thank you for the great
        things?" — measured and replaced.)"""
        import random
        rng = rng or random.Random()
        toks = tokenize(taught_text)
        coup = word_engine._load_coupling() or {}
        # candidate substitutions: content words with a strong second-order kin substitute
        subs = []
        for i, t in enumerate(toks):
            w = t.lower()
            if len(w) <= 3 or not w.isalpha():
                continue
            pool = list((coup.get(w) or {}).keys())[:20]
            best_k, best_w = 0.0, None
            for c in pool:
                if c == w or len(c) <= 3:
                    continue
                k = word_engine.kinship(w, c)
                if k > best_k:
                    best_k, best_w = k, c
            if best_w and best_k >= 1.0 / 3.0:      # the memory-cycle share as the lock
                subs.append((best_k, i, best_w))
        subs.sort(reverse=True)
        out = list(toks)
        for _, i, w2 in subs[:2]:                     # substitute at most two words
            out[i] = w2 if out[i].islower() else w2.capitalize()
        cand = " ".join(out)
        cand = re.sub(r"\s+([.,!?;:'])", r"\1", cand).replace(" '", "'").strip()
        if cand.strip().lower() != taught_text.strip().lower():
            return cand
        # no substitutable word found -> sentence-level variation: drop the last sentence
        # when there are several (still the taught content, re-expressed by selection)
        parts = re.split(r"(?<=[.!?])\s+", taught_text.strip())
        if len(parts) >= 2:
            return " ".join(parts[:-1]).strip()
        return ""                                     # defer to the compose/teacher path

    def _beam(self, text, history, plan_ids, want_q, rng, width=4, expand=3,
              max_words=MAX_WORDS, overlay=None, forbidden=None):
        """Beam over the mixed conditioned distribution. Score = Σ log(1+count-mass) +
        the binary bonus per NEW plan word covered (2 per hit — coverage is worth more
        than any single step's fluency). Terminates a beam at sentence end once HALF the
        plan is covered (the lock)."""
        import math
        S = self._store()
        vocab, words_arr, cond = S["vocab"], S["words"], S["cond"]
        base, topic = self._topic_set(text, history)
        base_ids = {vocab[w] for w in base if w in vocab}
        self._hits_cache = {}
        # THE INTEGRATED CONTEXT STATE (contextual_integration.ep, ernos-verified):
        # five kin-diffusion rounds at 2^-k — order-k context relations with forced
        # weights. Admission and weighting come from S's mass, not a binary topic set.
        from omni.context_state import integrated_state
        istate = integrated_state(text, history)
        # UNIT-CAPACITY ADMISSION (forced, reused): the state's minimal strongest prefix
        # whose cumulative mass reaches the lock 1/2 IS the admitted context; the tail is
        # suppressed. (Admitting the full diffused state was also the measured hot-path
        # cost: hundreds of tail words scanned per step at ~zero mass each.)
        state_ids = {}
        acc = 0.0
        for w, m in sorted(istate.items(), key=lambda kv: -kv[1]):
            nid = vocab.get(w)
            if nid is not None:
                state_ids[nid] = m
                acc += m
                if acc >= 0.5:
                    break
        topic_ids = (set(state_ids) | {vocab[w] for w in topic if w in vocab} | plan_ids)
        _PUNC = {".", ",", "!", "?", "'", ";", ":", '"', ")", "(", "-"}

        def step_dist(out):
            last_tok = out[-1] if out else "\x02"
            last_id = vocab.get(last_tok, -1)
            prev_tok = out[-2] if len(out) >= 2 else "\x02"
            prev_id = vocab.get(prev_tok, -1)
            gcw = tuple(w for w in out if len(w) > 3 and w.isalpha())[-8:]
            live_ids = self._live_state_ids(state_ids, gcw, vocab) if gcw else state_ids
            dist = Counter()
            clo = self._closures(S)
            if last_id >= 0:
                for tid, c in self._topic_hits(S, prev_id, last_id, topic_ids, clo):
                    w_t = (2 if (tid in base_ids or tid in plan_ids) else 1) \
                        * (1.0 + live_ids.get(tid, 0.0))
                    for nid, n in c.items():
                        dist[nid] += n * w_t
                # taught-transition overlay (reexpress): the correction's own learned
                # wording as high-count local transitions
                if overlay:
                    ov = overlay.get(last_id)
                    if ov:
                        for nid, n in ov.items():
                            dist[nid] += n * 4
            window = ([w for w in base][:WINDOW_SEED] + out)[-6:]
            scored, deepest = self._fluent_fast(window) if window else ([], 0)
            if scored:
                tot_d = sum(dist.values()) or 1
                tot_f = sum(float(s) for _, s in scored) or 1
                for w, s in scored:
                    nid = vocab.get(w.lower())
                    if nid is not None:
                        dist[nid] += (float(s) / tot_f) * tot_d
                # (F3-v6 support-intersection was measured a REGRESSION and reverted: the
                # fluency store is built from the OLD unfiltered corpus, so intersecting
                # with its support let code artifacts through ("` < / dc") — the finding is
                # that the FLUENCY STORE needs rebuilding from the clean pair corpus before
                # any deeper fusion with it can help.)
            # structural guards + HARD topical constraint (F3-v3): a CONTENT word may only
            # be emitted if it belongs to plan ∪ topic — drift into corpus threads becomes
            # structurally impossible; grammar (function words) flows free. Established
            # constrained/lexically-restricted decoding.
            ents = getattr(self, "_entity_ids", None)
            if ents is None:
                ents = self._entity_ids = {vocab[w] for w in S.get("entities", []) if w in vocab}
            for nid in list(dist):
                w = words_arr[nid]
                if self._junk_token(w):
                    del dist[nid]      # token-pipeline junk (shared predicate)
                elif (not out and w in _PUNC) or (out and out[-1] in _PUNC and w in _PUNC):
                    del dist[nid]
                elif len(w) > 3 and w.isalpha() and nid not in topic_ids:
                    del dist[nid]
                elif nid in ents and nid not in base_ids:
                    del dist[nid]              # counted entity filter (F3-v5): corpus names
                    # ("sarah") are tied to their source contexts — never emitted unless the
                    # live conversation itself carries them
            return dist

        beams = [([], 0.0, frozenset())]          # (tokens, score, covered)
        finished = []
        for _ in range(max_words):
            nxt_beams = []
            for out, sc, cov in beams:
                dist = step_dist(out)
                if not dist:
                    finished.append((out, sc, cov))
                    continue
                tot = sum(dist.values())
                seen_bigrams = set(zip(out, out[1:]))          # no-repeat-bigram (F3-v4):
                for nid, c in dist.most_common(expand + 4):    # the established decoding
                    w = words_arr[nid]                          # guard against loops
                    if out and (out[-1], w) in seen_bigrams:
                        continue
                    ns = sc + math.log1p(c / tot) + rng.random() * 1e-6   # tie noise only
                    ncov = cov | ({nid} if nid in plan_ids else set())
                    if nid in plan_ids and nid not in cov:
                        ns += 2.0                  # the binary bonus per new plan word
                    nout = out + [w]
                    if w in _END and len(nout) >= MIN_WORDS and (
                            not plan_ids or len(ncov) * 2 >= len(plan_ids)):
                        finished.append((nout, ns, ncov))
                    else:
                        nxt_beams.append((nout, ns, ncov))
            if not nxt_beams:
                break
            nxt_beams.sort(key=lambda b: -b[1])
            beams = nxt_beams[:width]
        finished = [f for f in finished if f[0]] or [b for b in beams if b[0]]
        if forbidden:
            # NEVER-VERBATIM at the beam level: exact reconstruction of a stored string
            # (e.g. the taught correction) is forbidden — internalize, never replay
            finished = [f for f in finished
                        if " ".join(f[0]).strip().lower() not in forbidden
                        and re.sub(r"\s+([.,!?;:'])", r"\1", " ".join(f[0])).strip().lower()
                        not in forbidden] or []
        if not finished:
            return "", -1e18
        out, sc, cov = max(finished, key=lambda b: b[1] / max(len(b[0]), 1))
        if want_q and out and out[-1] == ".":
            out[-1] = "?"
        s = ""
        for w in out:
            if not s:
                s = w.capitalize()
            elif w in {".", ",", "!", "?", "'", ";", ":"} or w.startswith("'"):
                s += w
            else:
                s += " " + w
        return re.sub(r"\s+([.,!?;:])", r"\1", s).strip(), sc / max(len(out), 1)

    def _select_score(self, s, plan):
        """Counted selector: plan coverage + adjacency fluency (fraction of adjacent
        word pairs attested in the local bigram store). Internal, never the judge."""
        toks = [t.lower() for t in tokenize(s)]
        cw = set(_content_words(toks))
        cov = (len(cw & set(plan)) / len(plan)) if plan else 0.0
        pairs = list(zip(toks, toks[1:]))
        if not pairs:
            return cov
        ok = 0
        for a, b in pairs:
            scored, _ = word_engine._scored_fluent([a])
            if any(w.lower() == b for w, _ in (scored or [])[:60]):
                ok += 1
        return cov + ok / len(pairs)

    def _realize(self, text, history, plan_ids, want_q, rng, max_words=MAX_WORDS):
        """One constrained sample: plan words carry the binary factor; punctuation is
        structurally guarded; stops at sentence end once HALF the plan is covered (the
        lock) or at the cap. Narrow sampling (top-4) — the established low temperature."""
        S = self._store()
        vocab, words_arr, cond = S["vocab"], S["words"], S["cond"]
        base, topic = self._topic_set(text, history)
        base_ids = {vocab[w] for w in base if w in vocab}
        topic_ids = {vocab[w] for w in topic if w in vocab} | plan_ids
        covered = set()
        _PUNC = {".", ",", "!", "?", "'", ";", ":", '"', ")", "(", "-"}

        out = []
        for step in range(max_words):
            last_tok = out[-1] if out else "\x02"
            last_id = vocab.get(last_tok, -1)
            prev_tok = out[-2] if len(out) >= 2 else "\x02"
            prev_id = vocab.get(prev_tok, -1)
            gcw = tuple(w for w in out if len(w) > 3 and w.isalpha())[-8:]
            live_ids = self._live_state_ids(state_ids, gcw, vocab) if gcw else state_ids
            dist = Counter()
            clo = self._closures(S)
            if last_id >= 0:
                for tid, c in self._topic_hits(S, prev_id, last_id, topic_ids, clo):
                    w_t = (2 if (tid in base_ids or tid in plan_ids) else 1) \
                        * (1.0 + live_ids.get(tid, 0.0))
                    for nid, n in c.items():
                        dist[nid] += n * w_t
            window = ([w for w in base][:WINDOW_SEED] + out)[-6:]
            scored, _ = self._fluent_fast(window) if window else ([], 0)
            fl = Counter()
            for w, s in scored or []:
                nid = vocab.get(w.lower())
                if nid is not None:
                    fl[nid] += float(s)
            if fl:
                tot_f = sum(fl.values()) or 1
                tot_d = sum(dist.values()) or 1
                for nid, v in fl.items():
                    dist[nid] += (v / tot_f) * tot_d
            if not dist:
                break
            # PLAN CONSTRAINT: uncovered plan words get the binary factor again
            for pid_ in plan_ids - covered:
                if pid_ in dist:
                    dist[pid_] *= 2
            # structural guards: no punctuation opener, no doubled punctuation, no junk
            drop = set()
            for nid in dist:
                w = words_arr[nid]
                if self._junk_token(w):
                    drop.add(nid)
                elif (not out and w in _PUNC) or (out and out[-1] in _PUNC and w in _PUNC):
                    drop.add(nid)
            for nid in drop:
                del dist[nid]
            if not dist:
                break
            items = dist.most_common(4)                    # narrow sampling
            total = sum(c for _, c in items)
            r = rng.random() * total
            acc, pick = 0.0, items[0][0]
            for nid, c in items:
                acc += c
                if r < acc:
                    pick = nid
                    break
            w = words_arr[pick]
            out.append(w)
            if pick in plan_ids:
                covered.add(pick)
            # coverage stopping at the lock: end the sentence once HALF the plan is in
            if w in _END and len(out) >= MIN_WORDS and (
                    not plan_ids or len(covered) * 2 >= len(plan_ids)):
                break
        if want_q and out and out[-1] == ".":
            out[-1] = "?"
        s = ""
        for w in out:
            if not s:
                s = w.capitalize()
            elif w in {".", ",", "!", "?", "'", ";", ":"} or w.startswith("'"):
                s += w
            else:
                s += " " + w
        return re.sub(r"\s+([.,!?;:])", r"\1", s).strip()


WINDOW_SEED = 6
free_gen = FreeGen()
