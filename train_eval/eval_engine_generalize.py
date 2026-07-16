"""THE ENGINE'S OWN GENERALISATION -- no reference model, no transfer, no gate.

On a context whose long suffix is novel, the engine must still predict well from
its OWN counted structure. Its forced mechanism is the dyadic fold-mix backoff
(the 2^L cascade, Steps 313/319): blend the next-token distribution across EVERY
suffix depth L, weighted 2^L, with the No-Zero floor 1/V per level:

  p(t) = [ sum_{L=0..kmax} 2^L * (count_L(t) + 1/V)/(total_L + 1) ] / [ sum 2^L ]

A depth-8 miss still gets depth-3/2/1 signal -- generalising on unseen contexts by
structural backoff, zero external model. Measured engine-ALONE in bits-per-byte over
GPT-2's tokens (so directly comparable to GPT-2's own 1.44 and the top-suffix baseline
~3.0). This is the number that must come down for the engine to generalise.
"""
import os, sys, math, glob, re, json
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation

HERE = os.path.dirname(os.path.abspath(__file__))
SENT = chr(50257)
CTX = 16
EVAL_TOKS = 900


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


def main():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    Vtok = tok.vocab_size

    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    train_texts = [tom[:cut]]
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try: train_texts.append(clean(open(f, encoding="utf-8", errors="replace").read()))
        except Exception: pass
    held_text = tom[cut:]

    def to_tokchars(t): return "".join(chr(i) for i in tok(t).input_ids)
    corpus = SENT.join(to_tokchars(t) for t in train_texts)
    held_ids = tok(held_text).input_ids[:EVAL_TOKS]
    held_str = "".join(chr(i) for i in held_ids)
    held_bytes = len(tok.decode(held_ids).encode("utf-8"))
    uni = Counter(c for c in corpus if c != SENT)
    uni_total = sum(uni.values())
    invV = 1.0 / Vtok

    # --- kinship precompute: continuation diversity (Kneser-Ney lower order) ---
    # N1+(*t) = # distinct tokens that precede t; discount D from bigram count-of-counts.
    preds = {}                       # token -> set of distinct predecessors
    bigram = Counter()
    prev = None
    for c in corpus:
        if c == SENT:
            prev = None; continue
        if prev is not None:
            preds.setdefault(c, set()).add(prev)
            bigram[(prev, c)] += 1
        prev = c
    N1plus_dot = sum(len(s) for s in preds.values())    # total distinct bigram types
    cont_uni = {t: len(s) / N1plus_dot for t, s in preds.items()}
    n1 = sum(1 for v in bigram.values() if v == 1)
    n2 = sum(1 for v in bigram.values() if v == 2)
    D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.75    # forced-from-data discount
    print(f"corpus {len(corpus):,} tokens | held {len(held_ids)} tokens / {held_bytes} bytes | "
          f"engine-ALONE | KN discount D={D:.3f}", flush=True)

    bits_base = bits_fold = bits_kn = 0.0
    for i in range(1, len(held_ids)):
        true = chr(held_ids[i])
        ctx = held_str[max(0, i - CTX):i]
        kmax = longest_k(corpus, ctx, CTX)
        conts_by_L = {}
        for L in range(1, kmax + 1):
            conts_by_L[L] = _search_suffix_in_corpus(corpus, ctx[-L:], L)
        # base: top matching suffix only
        if kmax > 0:
            c0 = conts_by_L[kmax]; p_base = (c0.count(true) + invV) / (len(c0) + 1)
        else:
            p_base = invV
        # fold-mix: 2^L blend over depths 0..kmax
        num = wsum = 0.0
        for L in range(0, kmax + 1):
            w = float(1 << L)
            if L == 0:
                tot = uni_total; cnt = uni.get(true, 0)
            else:
                c = conts_by_L[L]; tot = len(c); cnt = c.count(true)
            num += w * (cnt + invV) / (tot + 1); wsum += w
        p_fold = num / wsum if wsum else invV
        # interpolated Kneser-Ney: recurse low->high, base = continuation unigram
        p_kn = cont_uni.get(true, invV * 0.5) + invV * 1e-3
        for L in range(1, kmax + 1):
            c = conts_by_L[L]; tot = len(c)
            if tot == 0:
                continue
            cnt = c.count(true); types = len(set(c))
            p_kn = max(cnt - D, 0.0) / tot + (D * types / tot) * p_kn
        bits_base += -math.log2(max(p_base, 1e-12))
        bits_fold += -math.log2(max(p_fold, 1e-12))
        bits_kn += -math.log2(max(p_kn, 1e-12))
        if i % 200 == 0:
            print(f"  ...{i}/{len(held_ids)}  fold={bits_fold/i:.3f}/tok  kn={bits_kn/i:.3f}/tok", flush=True)

    bb, bf, bk = bits_base / held_bytes, bits_fold / held_bytes, bits_kn / held_bytes
    print(f"\nENGINE top-suffix (baseline):  {bb:.4f} bits/byte", flush=True)
    print(f"ENGINE fold-mix (forced gen):  {bf:.4f} bits/byte  ({bb-bf:+.3f} vs baseline)", flush=True)
    print(f"ENGINE fold+kinship (KN):      {bk:.4f} bits/byte  ({bf-bk:+.3f} vs fold-mix)", flush=True)
    print(f"  (GPT-2 alone on same slice: 1.44)", flush=True)
    json.dump({"engine_top_suffix": round(bb, 4), "engine_foldmix": round(bf, 4),
               "engine_kinship_kn": round(bk, 4), "kn_discount": round(D, 4), "gpt2": 1.44},
              open(os.path.join(HERE, "engine_generalize_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
