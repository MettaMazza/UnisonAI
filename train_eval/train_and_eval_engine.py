"""Train the omni engine on a corpus, measure its held-out bits-per-byte.

The engine trains by holding orbits (reading text once). Generalization is then
measured as char-level cross-entropy on a HELD-OUT (unseen) slice, under the
engine's forced No-Zero floor law:  p(c) = (count(c) + 1/V) / (total + 1).
On ASCII text, bits-per-char == bits-per-byte, so this is directly comparable to
GPT-2 / Gemma-4 (which we measure separately, tokenizer-independent).
"""
import os, sys, math, re, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fractions import Fraction
from omni.memory import SynapticGraph, exact_rational_shares
from omni.core import verify_locks

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "corpus_raw.txt")
CTX = 64          # max suffix context for eval (matches the engine's whole-conversation cap scale)
EVAL_CHARS = 4000  # held-out chars to score (speed)


def clean(text):
    # strip Project Gutenberg boilerplate
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    m2 = re.search(r"\*\*\* END OF", text)
    if m1: text = text[m1.end():]
    if m2: text = text[:text.rfind("*** END OF")]
    text = text.replace("\r\n", "\n")
    # case-fold (the engine's key law) and collapse whitespace runs lightly
    return text.lower()


def main():
    verify_locks()
    text = clean(open(RAW, encoding="utf-8").read())
    # 90/10 train / held-out split
    cut = int(len(text) * 0.90)
    train, held = text[:cut], text[cut:cut + max(EVAL_CHARS, 4000)]
    vocab = sorted(set(text))
    V = len(vocab)
    print(f"corpus {len(text):,} chars | train {len(train):,} | held-out {len(held):,} | vocab {V}", flush=True)

    # TRAIN: hold the train text as paragraph orbits
    g = SynapticGraph(save_path=os.path.join(HERE, "engine_graph.json"))
    paras = [p for p in re.split(r"\n\s*\n", train) if p.strip()]
    for p in paras:
        g.hold_orbit(list(p), ukey="public")
    g._rebuild_corpus_cache() if hasattr(g, "_rebuild_corpus_cache") else None
    print(f"trained: {sum(len(v) for v in g.orbits.values()):,} orbits held", flush=True)

    # EVAL: char-level bits-per-byte on the held-out slice, No-Zero floor law
    inv_V = Fraction(1, V)
    bits = 0.0
    n = 0
    matched = 0
    for i in range(len(held)):
        ctx = list(held[max(0, i - CTX):i])
        c = held[i]
        shares, depth, total = exact_rational_shares(ctx, g)
        if total > 0:
            cnt = 0
            for cand, fv in shares.items():
                if cand == c:
                    cnt = int(fv.val * total)  # share = count/total
                    break
            # p(c) = (count + 1/V) / (total + 1)
            p = (Fraction(cnt) + inv_V) / (total + 1)
            if depth > 0 and cnt > 0:
                matched += 1
        else:
            p = inv_V  # no suffix match -> uniform babble over vocab
        bits += -math.log2(float(p))
        n += 1
        if n % 1000 == 0:
            print(f"  ...{n}/{len(held)}  running bpb={bits/n:.4f}", flush=True)

    bpb = bits / n
    result = {"model": "omni-engine", "held_out_chars": n, "bits_per_byte": round(bpb, 4),
              "perplexity": round(2 ** bpb, 2), "suffix_hit_rate": round(matched / n, 3),
              "train_chars": len(train), "vocab": V}
    print(f"\nOMNI ENGINE: {bpb:.4f} bits/byte  (perplexity {2**bpb:.2f}, suffix-hit {matched/n:.1%})", flush=True)
    json.dump(result, open(os.path.join(HERE, "engine_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
