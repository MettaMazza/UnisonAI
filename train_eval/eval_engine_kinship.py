"""THE ENGINE'S SEMANTIC GENERALISATION -- distributional kinship, no reference.

n-gram + KN plateau ~1.4 bpb because they only match LITERAL suffixes. To reach the
35B bar (0.34) the engine must generalise by MEANING: a novel context predicts from
its distributional kin. Count-native (no neural weights):

  1. co-occurrence (PPMI) gives every token a distributional profile;
  2. cosine over those profiles => each token's semantic kin (top-M);
  3. kinship prediction: pool the forward continuations of the last token's kin,
     weighted by similarity -> p_kin(next | kin(last));
  4. interpolate with the fold-mix+KN literal-suffix estimate.

Measured engine-ALONE in bits-per-byte over GPT-2 tokens (comparable to the 0.34 bar
and the fold+KN 1.83 plateau). This tests whether semantic kinship moves the number
below the literal-n-gram wall; data then scales whatever headroom it opens.
"""
import os, sys, math, glob, re, json
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation

HERE = os.path.dirname(os.path.abspath(__file__))
SENT = chr(50257)
CTX = 16
EVAL_TOKS = 900
FVOCAB = 4000      # frequent tokens that get a distributional profile
WIN = 5            # co-occurrence window (+/-)
KIN_M = 50         # semantic neighbours per token
BETA = 0.25        # weight on the kinship term (engineering, fixed - not tuned on test)


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
    invV = 1.0 / Vtok

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

    # --- KN precompute (the 1.83 backbone) ---
    preds = {}; bigram = Counter(); fwd = defaultdict(Counter); prev = None
    freq = Counter()
    for c in corpus:
        if c == SENT:
            prev = None; continue
        freq[c] += 1
        if prev is not None:
            preds.setdefault(c, set()).add(prev); bigram[(prev, c)] += 1; fwd[prev][c] += 1
        prev = c
    N1plus = sum(len(s) for s in preds.values())
    cont_uni = {t: len(s) / N1plus for t, s in preds.items()}
    n1 = sum(1 for v in bigram.values() if v == 1); n2 = sum(1 for v in bigram.values() if v == 2)
    D = n1 / (n1 + 2 * n2) if (n1 + 2 * n2) else 0.75

    # --- distributional kinship: PPMI profiles -> cosine -> top-M kin ---
    top = [t for t, _ in freq.most_common(FVOCAB)]
    ridx = {t: i for i, t in enumerate(top)}
    F = len(top)
    cooc = np.zeros((F, F), dtype=np.float32)
    seq = [c for c in corpus]                       # includes SENT (skipped as center/neighbour)
    for i, c in enumerate(seq):
        ci = ridx.get(c)
        if ci is None or c == SENT: continue
        lo = max(0, i - WIN); hi = min(len(seq), i + WIN + 1)
        for j in range(lo, hi):
            if j == i: continue
            nj = ridx.get(seq[j])
            if nj is not None and seq[j] != SENT:
                cooc[ci, nj] += 1.0
    tot = cooc.sum()
    rows = cooc.sum(1, keepdims=True); cols = cooc.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.log(np.maximum(cooc * tot, 1e-9) / np.maximum(rows * cols, 1e-9))
    ppmi = np.maximum(ppmi, 0.0)
    norm = np.linalg.norm(ppmi, axis=1, keepdims=True); norm[norm == 0] = 1.0
    unit = ppmi / norm
    # forward continuation prob rows for the kin pooling (over frequent 'next' too)
    fwd_tot = {t: sum(fwd[t].values()) for t in top}
    print(f"corpus {len(corpus):,} tokens | held {len(held_ids)} | F={F} | KN D={D:.3f} | kinship WIN={WIN} M={KIN_M} beta={BETA}", flush=True)

    def kin_of(t):
        i = ridx.get(t)
        if i is None: return None
        sims = unit @ unit[i]
        order = np.argpartition(-sims, KIN_M + 1)[:KIN_M + 1]
        return [(top[j], float(sims[j])) for j in order if sims[j] > 0]

    kin_cache = {}
    bits_kn = bits_kin = 0.0
    for i in range(1, len(held_ids)):
        true = chr(held_ids[i])
        ctx = held_str[max(0, i - CTX):i]
        kmax = longest_k(corpus, ctx, CTX)
        conts_by_L = {L: _search_suffix_in_corpus(corpus, ctx[-L:], L) for L in range(1, kmax + 1)}
        # interpolated KN (the 1.83 backbone)
        p_kn = cont_uni.get(true, invV * 0.5) + invV * 1e-3
        for L in range(1, kmax + 1):
            c = conts_by_L[L]; t = len(c)
            if t == 0: continue
            p_kn = max(c.count(true) - D, 0.0) / t + (D * len(set(c)) / t) * p_kn
        # distributional kinship term on the last token
        last = ctx[-1] if ctx else None
        p_kin_term = 0.0
        if last is not None:
            if last not in kin_cache: kin_cache[last] = kin_of(last)
            kin = kin_cache[last]
            if kin:
                num = den = 0.0
                for s, sim in kin:
                    ft = fwd_tot.get(s, 0)
                    if ft: num += sim * (fwd[s].get(true, 0) / ft)
                    den += sim
                p_kin_term = num / den if den else 0.0
        p_kin = (1 - BETA) * p_kn + BETA * p_kin_term
        bits_kn += -math.log2(max(p_kn, 1e-12))
        bits_kin += -math.log2(max(p_kin, 1e-12))
        if i % 200 == 0:
            print(f"  ...{i}/{len(held_ids)}  kn={bits_kn/i:.3f}/tok  +kinship={bits_kin/i:.3f}/tok", flush=True)

    bk, bx = bits_kn / held_bytes, bits_kin / held_bytes
    print(f"\nENGINE fold+KN:                {bk:.4f} bits/byte", flush=True)
    print(f"ENGINE +distributional kinship:{bx:.4f} bits/byte  ({bk-bx:+.3f} vs fold+KN)", flush=True)
    print(f"  (target: 35B bar 0.34   |  GPT-2 stepping-stone 1.44)", flush=True)
    json.dump({"engine_fold_kn": round(bk, 4), "engine_kinship": round(bx, 4),
               "target_35b": 0.343, "beta": BETA, "F": F, "kin_M": KIN_M},
              open(os.path.join(HERE, "engine_kinship_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
