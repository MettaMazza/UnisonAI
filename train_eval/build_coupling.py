"""Build the word COUPLING graph (coherence_value.ep) from the broad corpus: for each
frequent word, its strongest co-occurrence neighbours (PPMI over a window). This is the
fold coupling the coherence lock reads -- so a content word can be judged coherent (>=1/2
coupled to the statement's meaning) against REAL associations, not just 82 lessons.
Saves omni/word_coupling.pkl : {word -> {neighbour -> ppmi}} for the top-F words."""
import os, sys, pickle, time, math
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conv_corpus.txt")
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "word_coupling.pkl"))
CAP = 120_000_000
F = 12000        # frequent words that get a coupling profile
WIN = 5
TOPK = 40        # neighbours kept per word


def main():
    t0 = time.time()
    words = tokenize(open(CORPUS, encoding="utf-8", errors="replace").read(CAP).lower())
    freq = Counter(words)
    top = [w for w, _ in freq.most_common(F) if len(w) > 2]
    F2 = len(top)
    idx = {w: i for i, w in enumerate(top)}
    print(f"{len(words):,} words | {F2} coupling words | {time.time()-t0:.0f}s", flush=True)

    cooc = np.zeros((F2, F2), dtype=np.float32)
    n = len(words)
    for i in range(n):
        ci = idx.get(words[i])
        if ci is None:
            continue
        lo, hi = max(0, i - WIN), min(n, i + WIN + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            cj = idx.get(words[j])
            if cj is not None:
                cooc[ci, cj] += 1.0
    tot = cooc.sum()
    rows = cooc.sum(1, keepdims=True)
    cols = cooc.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(np.maximum(cooc * tot, 1e-9) / np.maximum(rows * cols, 1e-9))
    ppmi = np.maximum(ppmi, 0.0).astype(np.float32)
    print(f"cooc+ppmi done {time.time()-t0:.0f}s; extracting top-{TOPK} neighbours", flush=True)

    graph = {}
    for i, w in enumerate(top):
        row = ppmi[i]
        if row.max() <= 0:
            continue
        order = np.argpartition(-row, min(TOPK, F2 - 1))[:TOPK]
        nb = {top[j]: float(row[j]) for j in order if row[j] > 0}
        if nb:
            graph[w] = nb
    with open(OUT, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved coupling graph: {len(graph):,} words -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
