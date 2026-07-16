"""DATA-SCALING of the engine's generalisation: proper modified-Kneser-Ney n-gram
LM with precomputed tables (O(1) lookup, scales to 100M+ tokens), swept over corpus
size. Answers: how far does DATA alone move engine-alone bpb toward the 35B bar 0.34?

Fold-mix/KN over str.find can't scale; this builds order-N tables once and applies
interpolated modified-KN (3 discounts by count, continuation-count base). Trains on
books + pg_corpus (Gutenberg) up to CAP bytes; held-out = the Tom Sawyer tail (same
slice as every other eval), measured in bits-per-byte over GPT-2 tokens.
Usage: eval_ngram_scale.py CAP_MB [ORDER]
"""
import os, sys, glob, re, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = (int(sys.argv[1]) if len(sys.argv) > 1 else 60) * 1_000_000
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EVAL_TOKS = 900


def clean(text):
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m1: text = text[m1.end():]
    if "*** END OF" in text: text = text[:text.rfind("*** END OF")]
    return text.replace("\r\n", "\n").lower()


def main():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # held-out: Tom Sawyer tail (unchanged slice)
    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    held_ids = tok(tom[cut:]).input_ids[:EVAL_TOKS]
    held_bytes = len(tok.decode(held_ids).encode("utf-8"))

    # training text: books first (in-domain), then pg_corpus up to CAP bytes
    texts = [tom[:cut]]
    nbytes = len(texts[0])
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try:
            t = clean(open(f, encoding="utf-8", errors="replace").read())
            texts.append(t); nbytes += len(t)
        except Exception: pass
    pg = os.path.join(HERE, "pg_corpus_clean.txt")   # decontaminated of held-out leakage
    if os.path.exists(pg) and nbytes < CAP:
        with open(pg, encoding="utf-8", errors="replace") as fh:
            chunk = fh.read(CAP - nbytes)
        texts.append(chunk.lower()); nbytes += len(chunk)

    # tokenise to one id stream (sentinel -1 between docs)
    ids = []
    for t in texts:
        ids.extend(tok(t).input_ids); ids.append(-1)
    print(f"train {nbytes/1e6:.0f} MB -> {len(ids):,} tokens | order {ORDER} | held {len(held_ids)} tok / {held_bytes} B", flush=True)

    # n-gram tables: for L=1..ORDER, dict suffix(tuple) -> Counter(next). L=0 = continuation unigram.
    tables = [defaultdict(Counter) for _ in range(ORDER + 1)]
    preds = defaultdict(set)                    # for continuation unigram
    N = len(ids)
    for i in range(N):
        w = ids[i]
        if w == -1: continue
        if i > 0 and ids[i - 1] != -1:
            preds[w].add(ids[i - 1])
        for L in range(1, ORDER + 1):
            if i - L < 0: break
            h = tuple(ids[i - L:i])
            if -1 in h: break
            tables[L][h][w] += 1
    N1p_dot = sum(len(s) for s in preds.values()) or 1
    cont_uni = {w: len(s) / N1p_dot for w, s in preds.items()}
    Vc = len(cont_uni) or 1

    # modified-KN discounts per order from count-of-counts
    Ds = []
    for L in range(1, ORDER + 1):
        cc = Counter()
        for h, ctr in tables[L].items():
            for v in ctr.values():
                cc[v] += 1
        n1, n2, n3, n4 = cc[1], cc[2], cc[3], cc[4]
        Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
        D1 = 1 - 2 * Y * n2 / n1 if n1 else 0.5
        D2 = 2 - 3 * Y * n3 / n2 if n2 else D1
        D3 = 3 - 4 * Y * n4 / n3 if n3 else D2
        Ds.append((max(D1, 0.1), max(D2, 0.1), max(D3, 0.1)))

    def Dof(L, c):
        d1, d2, d3 = Ds[L - 1]
        return d1 if c == 1 else d2 if c == 2 else d3

    def kn(h_tuple, w, L):
        if L == 0:
            return cont_uni.get(w, 1.0 / Vc * 0.5) + 1e-9
        ctr = tables[L].get(h_tuple)
        lower = kn(h_tuple[1:], w, L - 1)
        if not ctr:
            return lower
        tot = sum(ctr.values())
        c = ctr.get(w, 0)
        n1 = sum(1 for v in ctr.values() if v == 1)
        n2 = sum(1 for v in ctr.values() if v == 2)
        n3p = sum(1 for v in ctr.values() if v >= 3)
        d1, d2, d3 = Ds[L - 1]
        gamma = (d1 * n1 + d2 * n2 + d3 * n3p) / tot
        disc = max(c - Dof(L, c), 0.0) / tot if c > 0 else 0.0
        return disc + gamma * lower

    bits = 0.0
    for j in range(1, len(held_ids)):
        w = held_ids[j]
        h = tuple(held_ids[max(0, j - ORDER):j])
        p = kn(h, w, len(h))
        bits += -math.log2(max(p, 1e-12))
    bpb = bits / held_bytes
    print(f"\nENGINE modified-KN order-{ORDER} @ {nbytes/1e6:.0f}MB: {bpb:.4f} bits/byte", flush=True)
    print(f"  discounts D1 by order: {[round(d[0],2) for d in Ds]}", flush=True)
    print(f"  (target: 35B bar 0.34  |  GPT-2 stepping-stone 1.44  |  prev best 1.80)", flush=True)
    import json
    json.dump({"train_mb": round(nbytes/1e6, 1), "tokens": len(ids), "order": ORDER, "bpb": round(bpb, 4)},
              open(os.path.join(HERE, f"ngram_scale_{int(nbytes/1e6)}mb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
