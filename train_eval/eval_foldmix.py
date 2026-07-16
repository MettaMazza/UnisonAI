"""The engine's forced generalization: the char-tier FOLD-MIX backoff.

Instead of only the longest-suffix exact shares (which babble on a miss), blend
the next-char distribution across EVERY suffix depth L, weighted 2^L (the dyadic
cascade, Steps 313/319), with the No-Zero floor 1/V per level:

  p(c) = [ sum_L 2^L * (count_L(c) + 1/V)/(total_L + 1) ] / [ sum_L 2^L ]

A depth-8 miss still gets depth-3/2/1 signal -- the engine generalizing on unseen
contexts by its own backoff, zero external model. Measured bits-per-byte on the
same held-out slice as the base engine (3.18) and GPT-2 (1.43).
"""
import os, sys, math, re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import SynapticGraph, _search_suffix_in_corpus, _SENTINEL, _has_continuation
from omni.core import GEN_B

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "corpus_raw.txt")
CTX = 16          # max backoff depth (deep enough; deeper suffixes rarely match)
EVAL_CHARS = 4000


def clean(text):
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m1: text = text[m1.end():]
    if "*** END OF" in text: text = text[:text.rfind("*** END OF")]
    return text.replace("\r\n", "\n").lower()


def foldmix_prob(corpus, ctx_str, true_c, V):
    """Blended next-char prob of true_c under the 2^L fold-mix over depths 0..CTX."""
    num = 0.0     # sum_L 2^L * (count_L(c)+1/V)/(total_L+1)
    wsum = 0.0    # sum_L 2^L
    inv_V = 1.0 / V
    for L in range(0, CTX + 1):
        w = float(GEN_B ** L)
        if L == 0:
            # unigram: all chars in corpus
            total = len(corpus) - corpus.count(_SENTINEL)
            cnt = corpus.count(true_c)
        else:
            suf = ctx_str[-L:]
            if len(suf) < L:
                break
            conts = _search_suffix_in_corpus(corpus, suf, L)
            total = len(conts)
            if total == 0:
                # this depth (and all deeper) don't occur; stop deepening
                # but still include its floor-only contribution then break
                num += w * (inv_V) / 1.0
                wsum += w
                break
            cnt = conts.count(true_c)
        num += w * (cnt + inv_V) / (total + 1)
        wsum += w
    return num / wsum if wsum else inv_V


def main():
    text = clean(open(RAW, encoding="utf-8").read())
    cut = int(len(text) * 0.90)
    train, held = text[:cut], text[cut:cut + EVAL_CHARS]
    V = len(set(text))
    # build the corpus string the engine searches (the train text, sentinel-joined paras)
    paras = [p for p in re.split(r"\n\s*\n", train) if p.strip()]
    corpus = _SENTINEL.join(paras)
    print(f"train {len(train):,} chars | held-out {len(held):,} | vocab {V} | fold-mix depth {CTX}", flush=True)

    bits = 0.0
    for i in range(len(held)):
        ctx = held[max(0, i - CTX):i]
        p = foldmix_prob(corpus, ctx, held[i], V)
        bits += -math.log2(max(p, 1e-12))
        if (i + 1) % 1000 == 0:
            print(f"  ...{i+1}/{len(held)}  running bpb={bits/(i+1):.4f}", flush=True)

    bpb = bits / len(held)
    print(f"\nOMNI ENGINE (fold-mix): {bpb:.4f} bits/byte  (perplexity {2**bpb:.2f})", flush=True)
    print(f"  vs base engine 3.18 (top-suffix only), GPT-2 1.43", flush=True)
    json.dump({"model": "omni-engine-foldmix", "bits_per_byte": round(bpb, 4),
               "perplexity": round(2 ** bpb, 2)}, open(os.path.join(HERE, "engine_foldmix_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
