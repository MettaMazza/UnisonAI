"""Build the word KIN store — the counted embedding-similarity analogue (Levy-Goldberg).

The coupling graph (build_coupling.py) is FIRST-order: words that co-occur (sea -> coral,
acidification). That is syntagmatic association — right for query expansion, wrong for
paraphrase binding, because synonyms almost never co-occur ("sea" and "ocean" name the same
thing; nobody says both). What embeddings actually measure is SECOND-order similarity:
two words are kin when their CONTEXT DISTRIBUTIONS are similar — and per Levy-Goldberg
(SGNS ~ PPMI factorization) that quantity is counted, not learned:

    kin(a, b) = cosine( PPMI-profile(a), PPMI-profile(b) )

Built from the clean pair corpus (the same text the engine serves from). Keeps the top
BAND = 32 kin per word (the functional band, forced). Saves omni/word_kin.pkl:
{word -> {kin_word -> cosine}}.

The binding measure (pair_retrieval.taught_binding) reads THIS store; the calibration gate
(binding_calibration.py) must PASS on it before any run believes the measure.

Run: PYTHONPATH=. python3 train_eval/build_kin_store.py
"""
import os, sys, pickle, time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = os.path.abspath(os.path.join(HERE, "..", "omni", "pairs.pkl"))
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "word_kin.pkl"))
F = 12000        # profiled vocabulary (frequent words)
WIN = 5          # context window (same as the coupling build)
BAND = 32        # kin kept per word — the functional band, 2^(b+c)


def main():
    t0 = time.time()
    P = pickle.load(open(PAIRS, "rb"))
    text = "\n".join(P["prompts"]) + "\n" + "\n".join(P["responses"])
    words = tokenize(text.lower())
    del text
    freq = Counter(words)
    top = [w for w, _ in freq.most_common(F) if len(w) > 2]
    F2 = len(top)
    idx = {w: i for i, w in enumerate(top)}
    print(f"{len(words):,} tokens | {F2} profiled words | {time.time()-t0:.0f}s", flush=True)

    cooc = np.zeros((F2, F2), dtype=np.float32)
    n = len(words)
    ids = np.array([idx.get(w, -1) for w in words], dtype=np.int32)
    for i in range(n):
        ci = ids[i]
        if ci < 0:
            continue
        hi = min(n, i + WIN + 1)
        for j in range(i + 1, hi):
            cj = ids[j]
            if cj >= 0:
                cooc[ci, cj] += 1.0
                cooc[cj, ci] += 1.0
    tot = cooc.sum()
    rows = cooc.sum(1, keepdims=True)
    cols = cooc.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(np.maximum(cooc * tot, 1e-9) / np.maximum(rows * cols, 1e-9))
    ppmi = np.maximum(ppmi, 0.0).astype(np.float32)
    del cooc
    print(f"PPMI done {time.time()-t0:.0f}s; second-order cosine ({F2}x{F2})", flush=True)

    norms = np.linalg.norm(ppmi, axis=1, keepdims=True)
    ppmi /= np.maximum(norms, 1e-9)
    sim = ppmi @ ppmi.T                       # cosine of context profiles
    np.fill_diagonal(sim, 0.0)
    print(f"cosine done {time.time()-t0:.0f}s; extracting top-{BAND} kin", flush=True)

    kin = {}
    for i, w in enumerate(top):
        row = sim[i]
        order = np.argpartition(-row, BAND)[:BAND]
        nb = {top[j]: float(row[j]) for j in order if row[j] > 0}
        if nb:
            kin[w] = nb
    with open(OUT, "wb") as f:
        pickle.dump(kin, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved kin store: {len(kin):,} words -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)
    for probe in (("sea", "ocean"), ("meal", "dinner"), ("motivation", "motivated"),
                  ("thoughts", "think"), ("evening", "night")):
        a, b = probe
        nb = kin.get(a, {})
        rank = (sorted(nb, key=lambda k: -nb[k]).index(b) + 1) if b in nb else None
        print(f"  kin check {a!r}~{b!r}: rank {rank}, cos {nb.get(b, 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
