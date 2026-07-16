"""Data-scaling test (fast, single-threaded — no ProcessPool). Train on 8 books
(~5.5MB), hold out the Tom Sawyer tail, measure held-out bits-per-byte under the
engine's law (longest-suffix exact shares + No-Zero floor). Optional --foldmix
blends all depths (2^L). Uses the corpus string directly via the engine's own
search functions, bypassing the live engine's parallel path.
"""
import os, sys, math, re, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation, _SENTINEL
from omni.core import GEN_B

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = 24
EVAL_CHARS = 1500
FOLDMIX = "--foldmix" in sys.argv


def clean(text):
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m1: text = text[m1.end():]
    if "*** END OF" in text: text = text[:text.rfind("*** END OF")]
    return text.replace("\r\n", "\n").lower()


def longest_k(corpus, ctx_str, cap):
    lo, hi, best = 1, min(len(ctx_str), cap), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_continuation(corpus, ctx_str[-mid:]):
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return best


def base_prob(corpus, ctx_str, c, V):
    k = longest_k(corpus, ctx_str, CTX)
    if k == 0:
        return 1.0 / V
    conts = _search_suffix_in_corpus(corpus, ctx_str[-k:], k)
    total = len(conts)
    cnt = conts.count(c)
    return (cnt + 1.0 / V) / (total + 1)


def foldmix_prob(corpus, ctx_str, c, V, uni_total, uni_cnt_fn):
    num = 0.0; wsum = 0.0; invV = 1.0 / V
    kmax = longest_k(corpus, ctx_str, CTX)
    for L in range(0, kmax + 1):
        w = float(GEN_B ** L)
        if L == 0:
            total, cnt = uni_total, uni_cnt_fn(c)
        else:
            conts = _search_suffix_in_corpus(corpus, ctx_str[-L:], L)
            total = len(conts); cnt = conts.count(c)
            if total == 0:
                break
        num += w * (cnt + invV) / (total + 1); wsum += w
    return num / wsum if wsum else invV


def main():
    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    tom_train, held = tom[:cut], tom[cut:cut + EVAL_CHARS]
    parts = []
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try: parts.append(clean(open(f, encoding="utf-8", errors="replace").read()))
        except Exception: pass
    parts.append(tom_train)
    corpus = _SENTINEL.join(parts)
    V = len(set(corpus) - {_SENTINEL})
    print(f"training: {len(parts)} books, {len(corpus):,} chars | held-out {len(held)} | vocab {V} | "
          f"{'FOLD-MIX' if FOLDMIX else 'base'} depth {CTX}", flush=True)

    # unigram counts (for fold-mix L=0)
    from collections import Counter
    uni = Counter(ch for ch in corpus if ch != _SENTINEL)
    uni_total = sum(uni.values())

    bits = 0.0
    for i in range(len(held)):
        ctx = held[max(0, i - CTX):i]
        c = held[i]
        if FOLDMIX:
            p = foldmix_prob(corpus, ctx, c, V, uni_total, lambda x: uni.get(x, 0))
        else:
            p = base_prob(corpus, ctx, c, V)
        bits += -math.log2(max(p, 1e-12))
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(held)}  bpb={bits/(i+1):.4f}", flush=True)

    bpb = bits / len(held)
    tag = "8book-foldmix" if FOLDMIX else "8book-base"
    print(f"\nOMNI ENGINE ({tag}, {len(corpus)//1000}K chars): {bpb:.4f} bits/byte (ppl {2**bpb:.2f})", flush=True)
    print(f"  vs 1-book 3.18, GPT-2 1.43", flush=True)
    json.dump({"model": f"omni-{tag}", "train_chars": len(corpus), "bits_per_byte": round(bpb, 4)},
              open(os.path.join(HERE, f"engine_{tag}_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
