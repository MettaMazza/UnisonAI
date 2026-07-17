"""F1 datastore — the kNN-LM translated to counts (FRONTIER_PLAN).

The established bridge (Khandelwal et al. 2020): a datastore of (context, next-token)
records; at inference the current context retrieves its nearest stored contexts and their
next-tokens form a distribution, interpolated with the local LM. The counted translation:
a context is its trailing CONTENT-WORD set plus its exact last token; similarity is counted
overlap (idf-weighted); the syntax tier (same last token) weighs 2x — the binary factor.

Built over the clean pair corpus's text (prompts + responses — already register-filtered),
so the datastore inherits the conversational register. Saves omni/kin_context.pkl:
  {"vocab": {word->id}, "words": [id->word], "nxt": array(next-token id per position),
   "last": array(last-token id per position), "inv": {content_word -> array(positions)}}
"""
import os, sys, pickle, time
from array import array
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize, _content_words

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.join(HERE, "..", "omni", "pairs.pkl")
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "kin_context.pkl"))
WINDOW = 8              # trailing content window per position
CAP_POST = 4000         # posting cap per content word

_TAILS = {"s", "t", "re", "ve", "ll", "d", "m"}


def merge_contractions(toks):
    """don / ' / t -> don't  (v1 defect: split contractions let the conditioned table mix
    tails across words — 'it' re', 'I' t know'). One token per contraction, both at store
    build and at generation."""
    out = []
    i = 0
    while i < len(toks):
        if (i + 2 < len(toks) and toks[i + 1] in {"'", "’"} and toks[i + 2].lower() in _TAILS
                and toks[i].isalpha()):
            out.append(toks[i] + "'" + toks[i + 2])
            i += 3
        else:
            out.append(toks[i])
            i += 1
    return out


def main():
    t0 = time.time()
    P = pickle.load(open(PAIRS, "rb"))
    texts = P["prompts"] + P["responses"]
    vocab, words = {}, []

    def wid(w):
        i = vocab.get(w)
        if i is None:
            i = len(words); vocab[w] = i; words.append(w)
        return i

    # TOPIC-CONDITIONED N-GRAM (v2 — established topic-mixture LM shape). Key = the exact
    # LAST TOKEN (order/grammar by construction) x a TOPIC word from the trailing window
    # (relevance by the key). Value = Counter of next tokens. The v1 order-free content-SET
    # key was measured producing function-word soup (F0 run: 0/12, "Go on on a to on") —
    # order must live in the key, exactly as kNN-LM's context states encode it.
    cond = defaultdict(Counter)        # (last_id, topic_id) -> {next_id: count}
    cond3 = defaultdict(Counter)       # (prev_id, last_id, topic_id) -> {next_id: count}
    # TRIGRAM CONDITIONING (F-lever 1): two tokens of exact order in the key — the
    # established n-gram deepening; at generation the deeper tier carries the binary
    # factor 2 over the bigram tier (the same tiering the store already uses).
    BOS = wid("\x02")                  # sentence-start marker (the <s> of every n-gram LM)
    npos = 0
    cap_num = Counter()      # counted entity signal: how often a token appears Capitalized
    cap_den = Counter()      # mid-sentence (an entity name) vs in any position at all
    for ti, text in enumerate(texts):
        toks = merge_contractions(tokenize(text))
        if len(toks) < 2:
            continue
        low = [t.lower() for t in toks]
        for j, t in enumerate(toks):
            if t[:1].isalpha() and len(t) > 2:
                cap_den[low[j]] += 1
                prev_end = j == 0 or toks[j - 1] in {".", "!", "?"}
                if t[0].isupper() and not prev_end:
                    cap_num[low[j]] += 1
        # sentence OPENERS conditioned on the text's topic — replies must START like
        # sentences (v2 defect: mid-sentence openings, no boundary conditioning)
        text_topics = set(_content_words(low))
        first_id = wid(low[0])
        for cw in text_topics:
            c = cond[(BOS, wid(cw))]
            if len(c) < 400:
                c[first_id] += 1
        for i in range(1, len(toks)):
            last_id, next_id = wid(low[i - 1]), wid(low[i])
            prev_id = wid(low[i - 2]) if i >= 2 else BOS
            for cw in set(_content_words(low[max(0, i - WINDOW):i])):
                cw_id = wid(cw)
                c = cond[(last_id, cw_id)]
                if len(c) < 400:
                    c[next_id] += 1
                c3 = cond3[(prev_id, last_id, cw_id)]
                if len(c3) < 400:
                    c3[next_id] += 1
            npos += 1
        if ti % 100000 == 0 and ti:
            print(f"  ...{ti} texts, {npos} positions, {len(cond)} keys | {time.time()-t0:.0f}s", flush=True)

    # hapax pruning (established), at BOTH levels: a key seen once carries noise — and
    # so does a count-1 next-token ENTRY inside a surviving key (measured: corpus typos
    # like "fonetwish" won thin distributions through count-1 entries; the n-gram count
    # cutoff is the established remedy)
    def prune_vals(d):
        out = {}
        for k, v in d.items():
            vv = {n: c for n, c in v.items() if c > 1} or dict(v)
            if sum(vv.values()) > 1:
                out[k] = vv
        return out
    pruned = prune_vals(cond)
    print(f"  pruned {len(cond) - len(pruned):,} hapax bigram keys -> {len(pruned):,} kept", flush=True)
    pruned3 = prune_vals(cond3)
    print(f"  pruned {len(cond3) - len(pruned3):,} hapax trigram keys -> {len(pruned3):,} kept", flush=True)
    del cond3
    # entity tokens (counted): predominantly mid-sentence-capitalized words are names/places
    # tied to their source contexts — excluded at generation unless in the live topic
    entities = {w for w, d in cap_den.items()
                if d >= 4 and cap_num.get(w, 0) * 2 >= d}          # cap-fraction >= 1/2
    print(f"  counted entity tokens: {len(entities):,}", flush=True)
    with open(OUT, "wb") as f:
        pickle.dump({"vocab": vocab, "words": words, "cond": pruned, "cond3": pruned3,
                     "entities": sorted(entities)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved kin-context store v3 (trigram): {npos:,} positions, vocab {len(words):,}, "
          f"{len(cond):,} (last,topic) keys -> {OUT} ({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
