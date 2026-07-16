"""FAIR data-scaling test: held-out drawn from the SAME distribution as the scaling
data (a slice of the decontaminated Gutenberg corpus), removed from training with a
1MB decontamination window. Trains on increasing amounts of the REST and measures
modified-KN bpb. This isolates whether DATA helps IN-DOMAIN generalisation (the fair
version of Maria's data thesis) -- unlike the Tom-Sawyer held-out where added generic
text is off-distribution. Usage: eval_ngram_indomain.py CAP_MB [ORDER]
"""
import os, sys, re, math
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = (int(sys.argv[1]) if len(sys.argv) > 1 else 60) * 1_000_000
ORDER = int(sys.argv[2]) if len(sys.argv) > 2 else 5
EVAL_TOKS = 900


def main():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    pg = open(os.path.join(HERE, "pg_corpus_clean.txt"), encoding="utf-8", errors="replace").read()
    # held-out: a 4000-char slice from ~55% through the corpus (mid a book), decontaminate a 1MB window
    h0 = int(len(pg) * 0.55)
    held_text = pg[h0:h0 + 4000]
    WIN = 15_000_000                                               # excise the WHOLE book (+/-15MB)
    train_all = pg[:max(0, h0 - WIN)] + pg[h0 + WIN:]
    # verify decontam
    spans = [held_text[i:i + 60] for i in range(0, len(held_text) - 60, 40)]
    leaks = sum(1 for s in spans if s in train_all)

    held_ids = tok(held_text).input_ids[:EVAL_TOKS]
    held_bytes = len(tok.decode(held_ids).encode("utf-8"))
    train_text = train_all[:CAP]

    ids = tok(train_text).input_ids
    print(f"train {len(train_text)/1e6:.0f} MB -> {len(ids):,} tokens | order {ORDER} | "
          f"held {len(held_ids)} tok / {held_bytes} B | leak spans {leaks}/{len(spans)}", flush=True)

    tables = [defaultdict(Counter) for _ in range(ORDER + 1)]
    preds = defaultdict(set)
    N = len(ids)
    for i in range(N):
        w = ids[i]
        if i > 0:
            preds[w].add(ids[i - 1])
        for L in range(1, ORDER + 1):
            if i - L < 0: break
            tables[L][tuple(ids[i - L:i])][w] += 1
    N1p = sum(len(s) for s in preds.values()) or 1
    cont_uni = {w: len(s) / N1p for w, s in preds.items()}
    Vc = len(cont_uni) or 1

    Ds = []
    for L in range(1, ORDER + 1):
        cc = Counter()
        for ctr in tables[L].values():
            for v in ctr.values(): cc[v] += 1
        n1, n2, n3, n4 = cc[1], cc[2], cc[3], cc[4]
        Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
        D1 = 1 - 2 * Y * n2 / n1 if n1 else 0.5
        D2 = 2 - 3 * Y * n3 / n2 if n2 else D1
        D3 = 3 - 4 * Y * n4 / n3 if n3 else D2
        Ds.append((max(D1, .1), max(D2, .1), max(D3, .1)))

    def Dof(L, c): d = Ds[L - 1]; return d[0] if c == 1 else d[1] if c == 2 else d[2]

    def kn(h, w, L):
        if L == 0:
            return cont_uni.get(w, 1.0 / Vc * 0.5) + 1e-9
        ctr = tables[L].get(h)
        low = kn(h[1:], w, L - 1)
        if not ctr: return low
        tot = sum(ctr.values()); c = ctr.get(w, 0)
        n1 = sum(1 for v in ctr.values() if v == 1)
        n2 = sum(1 for v in ctr.values() if v == 2)
        n3p = sum(1 for v in ctr.values() if v >= 3)
        d1, d2, d3 = Ds[L - 1]
        gamma = (d1 * n1 + d2 * n2 + d3 * n3p) / tot
        return (max(c - Dof(L, c), 0.0) / tot if c > 0 else 0.0) + gamma * low

    bits = 0.0
    for j in range(1, len(held_ids)):
        h = tuple(held_ids[max(0, j - ORDER):j])
        bits += -math.log2(max(kn(h, held_ids[j], len(h)), 1e-12))
    bpb = bits / held_bytes
    print(f"ENGINE modified-KN order-{ORDER} @ {len(train_text)/1e6:.0f}MB (IN-DOMAIN held): {bpb:.4f} bits/byte", flush=True)


if __name__ == "__main__":
    main()
