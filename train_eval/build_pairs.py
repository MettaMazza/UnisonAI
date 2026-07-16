"""Stage 1 — the PROMPT->RESPONSE pair index with full BM25 collection statistics.

The corpus is conversational turns; consecutive turns ARE context->response pairs — the
established document unit of retrieval dialogue (response selection). This indexes each
clean, STANDALONE response keyed by the content words of its prompt, storing the statistics
the exact BM25 formula needs: per-pair term frequencies (tf>1 kept sparsely), prompt lengths,
avgdl, N. Duplicates are removed. Recombination/relexicalization at generation time keeps
the emitted reply non-verbatim.

Saves omni/pairs.pkl:
  {"responses", "prompts", "inv" (word -> [pair_id]), "tf_extra" ((word,pid) -> tf>1),
   "plen" ([content-word length per prompt]), "avgdl", "N"}
"""
import os, sys, re, pickle, time, hashlib
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize, _content_words

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conv_corpus.txt")
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "pairs.pkl"))
MAX_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 700_000
_CODE = set("{}[]<>=/\\|`~^*_#@")
_LEAK = ("main function", "prompt the user", "variable named", "best regards", "sincerely",
         "sign off", "conclude by", "as follows", "click here", "dear friend", "http", "www.",
         ".com", "@", "step 1", "step 2", "in this article", "in this tutorial")


def good_response(s):
    s = s.strip()
    w = s.split()
    if not (4 <= len(w) <= 26):
        return False
    low = s.lower()
    letters = sum(c.isalpha() or c == " " for c in s)
    if letters / max(len(s), 1) < 0.85:
        return False
    if any(c in _CODE for c in s):
        return False
    if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.02:
        return False
    if any(p in low for p in _LEAK):
        return False
    if re.search(r"\b\d{1,2}[:.]\d", s) or re.search(r"(^|\s)\d{1,2}\.\s", s):
        return False
    if re.match(r"^[A-Z][a-z]+:", s):          # dialogue "Name:" register
        return False
    # STANDALONE only: a response that leans on prior conversation reads as a non-sequitur
    # when served to a fresh message (measured: continuations judged BAD).
    if low.startswith(("and ", "but ", "so ", "also ", "or ", "then ", "- ", "yes,", "no,",
                       "sure,", "sure!", "absolutely", "certainly", "here are", "here's",
                       "as well", "another ", "additionally", "moreover", "that's ", "it's ",
                       "they ", "these ", "those ", "this ")):
        return False
    if any(f" {w} " in f" {low} " for w in ("more", "also", "another", "too")):
        return False
    return True


def good_prompt(s):
    w = s.split()
    return 2 <= len(w) <= 40 and any(c.isalpha() for c in s)


def main():
    t0 = time.time()
    responses, prompts, plen = [], [], []
    inv = defaultdict(list)
    tf_extra = {}
    seen = set()
    conv, n = [], 0

    def flush(conv):
        nonlocal n
        for i in range(len(conv) - 1):
            p, r = conv[i], conv[i + 1]
            if not (good_prompt(p) and good_response(r)):
                continue
            h = hashlib.md5((p.lower().strip() + "\x00" + r.lower().strip()).encode()).digest()
            if h in seen:                      # dedup exact pairs
                continue
            seen.add(h)
            pcw = _content_words(tokenize(p.lower()))
            if not pcw:
                continue
            pid = len(responses)
            responses.append(r.strip()); prompts.append(p.strip()); plen.append(len(pcw))
            for w, c in Counter(pcw).items():
                inv[w].append(pid)
                if c > 1:
                    tf_extra[(w, pid)] = c
            n += 1
            if n % 50000 == 0:
                print(f"  ...{n} pairs, {len(inv)} prompt-words | {time.time()-t0:.0f}s", flush=True)
            if n >= MAX_PAIRS:
                return True
        return False

    stop = False
    with open(CORPUS, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if flush(conv):
                    stop = True; break
                conv = []
                continue
            conv.append(line.strip())
        if not stop:
            flush(conv)

    inv = {w: lst[:1500] for w, lst in inv.items()}   # cap common prompt-words
    N = len(responses)
    avgdl = (sum(plen) / N) if N else 1.0
    with open(OUT, "wb") as fo:
        pickle.dump({"responses": responses, "prompts": prompts, "inv": inv,
                     "tf_extra": tf_extra, "plen": plen, "avgdl": avgdl, "N": N}, fo,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved pair index: {N:,} pairs (deduped), {len(inv):,} prompt-words, avgdl {avgdl:.1f} "
          f"-> {OUT} ({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
