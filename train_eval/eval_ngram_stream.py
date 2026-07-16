"""Streaming modified-KN generalisation eval -- scales to ANY corpus size in bounded
memory. We only need counts for the n-grams the held-out queries, so we stream the
corpus once and tally ONLY those (plus a unigram table). Multiple NARRATIVE held-out
slices (averaged) for robustness; on-the-fly decontamination (skip any chunk containing
a held shingle); bits-per-byte reported at size checkpoints so one pass gives the whole
data-scaling curve. No ceilings assumed -- just the measured curve.
Usage: eval_ngram_stream.py [CORPUS_FILE] [CAP_MB] [ORDER]
"""
import os, sys, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pg_big.txt")
CAP = (int(sys.argv[2]) if len(sys.argv) > 2 else 100000) * 1_000_000
ORDER = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SLICE_FRACS = [0.20, 0.30, 0.40, 0.50, 0.62, 0.70, 0.80, 0.88]
SLEN = 4000
EVAL_TOKS = 260
CHECKPOINT = 100_000_000     # report bpb every 100MB of accepted training


def main():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    pgc = open(os.path.join(HERE, "pg_corpus_clean.txt"), encoding="utf-8", errors="replace").read()
    held_slices = [pgc[int(len(pgc) * f):int(len(pgc) * f) + SLEN] for f in SLICE_FRACS]

    Q = set(); slice_data = []; held_bytes_total = 0
    for h in held_slices:
        ids = tok(h).input_ids[:EVAL_TOKS]
        s = "".join(chr(t) for t in ids)
        for j in range(1, len(ids)):
            for L in range(1, ORDER + 1):
                if j - L >= 0: Q.add(s[j - L:j])
        b = len(tok.decode(ids).encode("utf-8"))
        held_bytes_total += b
        slice_data.append((ids, s))
    Qlast = set(c[-1] for c in Q)
    cont = {c: Counter() for c in Q}
    uni = Counter()

    shs = set()
    for h in held_slices:
        hl = h.lower()
        for i in range(0, len(hl) - 40, 17): shs.add(hl[i:i + 40])
    shs = list(shs)
    print(f"held: {len(held_slices)} narrative slices / {held_bytes_total} bytes | Q={len(Q):,} | "
          f"corpus={os.path.basename(CORPUS)} order={ORDER}", flush=True)

    def discounts():
        Ds = {}
        for L in range(1, ORDER + 1):
            cc = Counter()
            for ctx, ctr in cont.items():
                if len(ctx) == L:
                    for v in ctr.values(): cc[v] += 1
            n1, n2, n3, n4 = cc[1], cc[2], cc[3], cc[4]
            Y = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.5
            D1 = 1 - 2 * Y * n2 / n1 if n1 else 0.6
            D2 = 2 - 3 * Y * n3 / n2 if n2 else D1
            D3 = 3 - 4 * Y * n4 / n3 if n3 else D2
            Ds[L] = (max(D1, .1), max(D2, .1), max(D3, .1))
        return Ds

    def evaluate():
        TU = sum(uni.values()) or 1
        Ds = discounts()
        def Dof(L, c): d = Ds[L]; return d[0] if c == 1 else d[1] if c == 2 else d[2]
        def kn(ctx, true, L):
            if L == 0:
                return uni.get(true, 0) / TU + 1e-9
            ctr = cont.get(ctx)
            low = kn(ctx[1:], true, L - 1)
            if not ctr: return low
            tot = sum(ctr.values()); c = ctr.get(chr(true), 0)
            n1 = sum(1 for v in ctr.values() if v == 1)
            n2 = sum(1 for v in ctr.values() if v == 2)
            n3p = sum(1 for v in ctr.values() if v >= 3)
            d1, d2, d3 = Ds[L]
            gamma = (d1 * n1 + d2 * n2 + d3 * n3p) / tot
            return (max(c - Dof(L, c), 0.0) / tot if c > 0 else 0.0) + gamma * low
        bits = 0.0
        for ids, s in slice_data:
            for j in range(1, len(ids)):
                ctx = s[max(0, j - ORDER):j]
                bits += -math.log2(max(kn(ctx, ids[j], len(ctx)), 1e-12))
        return bits / held_bytes_total

    read = 0; accepted = 0; next_ckpt = CHECKPOINT; carry = ""; skipped = 0
    with open(CORPUS, encoding="utf-8", errors="replace") as f:
        while read < CAP:
            chunk = f.read(4_000_000)
            if not chunk: break
            read += len(chunk)
            low = chunk.lower()
            if any(sh in low for sh in shs):
                carry = ""; skipped += 1; continue
            ids = tok(low).input_ids
            for w in ids: uni[w] += 1
            s = carry + "".join(chr(t) for t in ids)
            for i in range(1, len(s)):
                if s[i - 1] not in Qlast:
                    continue
                nxt = s[i]
                for L in range(1, ORDER + 1):
                    if i - L < 0: break
                    ctx = s[i - L:i]
                    c = cont.get(ctx)
                    if c is not None: c[nxt] += 1
            carry = s[-ORDER:]
            accepted += len(chunk)
            if accepted >= next_ckpt:
                bpb = evaluate()
                print(f"  @{accepted//1_000_000}MB accepted ({skipped} chunks skipped): {bpb:.4f} bpb", flush=True)
                next_ckpt += CHECKPOINT
    bpb = evaluate()
    print(f"\nFINAL @{accepted//1_000_000}MB: {bpb:.4f} bits/byte  (35B ref 0.34, GPT-2 stepping-stone 1.44)", flush=True)


if __name__ == "__main__":
    main()
