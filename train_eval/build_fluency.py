"""Build the word-tier FLUENCY store: general English word n-grams (L=1..4) from the
decontaminated broad corpus, so unfold has real language to re-express WITH (not just
the 82 taught lessons). Saves to omni/word_fluency.pkl for the engine to load and blend
into generation. This is the training that turns unfold from salad into coherent
re-expression."""
import os, sys, pickle, time
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conv_corpus.txt")
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "word_fluency.pkl"))
CAP = 120_000_000    # take the whole conversational corpus (register matters more than size)
MAXL = 4


def main():
    t0 = time.time()
    text = open(CORPUS, encoding="utf-8", errors="replace").read(CAP).lower()
    words = tokenize(text)
    print(f"tokenized {len(words):,} words from {len(text)/1e6:.0f}MB in {time.time()-t0:.0f}s", flush=True)

    stores = [defaultdict(Counter) for _ in range(MAXL + 1)]
    uni = Counter()
    n = len(words)
    for i in range(n):
        w = words[i]
        uni[w] += 1
        for L in range(1, MAXL + 1):
            if i - L < 0:
                break
            stores[L][tuple(words[i - L:i])][w] += 1
        if i % 2_000_000 == 0 and i:
            print(f"  ...{i//1_000_000}M words, L4 keys {len(stores[MAXL]):,}", flush=True)

    # convert to plain dicts for compact pickling
    out = {"maxl": MAXL, "uni": dict(uni),
           "stores": [None] + [{k: dict(v) for k, v in stores[L].items()} for L in range(1, MAXL + 1)]}
    with open(OUT, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved fluency store: vocab {len(uni):,} | "
          f"L-key counts {[len(stores[L]) for L in range(1, MAXL+1)]} -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
