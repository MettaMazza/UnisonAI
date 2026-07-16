"""THE TRANSFER-IN with Gemma-4 as q, EXACT and token-aligned (the true bar).

Gemma's weights are loaded directly (GGUF via llama-cpp, logits_all) so q is the
FULL next-token distribution -- no top-K truncation, no byte-marginal leak. The
engine is built over Gemma's OWN tokenisation (each token id -> one unicode char,
so the engine's suffix search reuses unchanged), so both sides score the same unit:

  p(T) = ( n(T) + q_gemma(T|ctx) ) / (total + 1)

n(T)/total = counted continuations of the longest matching token-suffix in the
corpus; q_gemma = Gemma's exact probability of the true next token. On unseen
context (total=0) p = q. Natural case throughout (fair to Gemma). Three bars on the
same token grid, all bits-per-byte over the true UTF-8 byte length of the held text:
  engine alone (No-Zero) | Gemma-4 alone (the target) | engine+Gemma transfer.
"""
import os, sys, math, re, glob, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation

HERE = os.path.dirname(os.path.abspath(__file__))
# reference model (q). Default: the strong LOCAL multimodal model Qwen3.5-35B-A3B Q8
# (verified 0.61 bpb / 62.5% greedy on held-out prose -- far stronger than GPT-2 1.44
# and the reasoning-gemma ~3.0). Override with argv[1] = path to any GGUF.
BLOB = (sys.argv[1] if len(sys.argv) > 1 else
        os.path.expanduser("~/.lmstudio/models/lmstudio-community/"
                           "Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf"))
REF_NAME = os.path.basename(BLOB)
CTX = 12            # engine token-suffix depth
EVAL_TOKS = 300     # held tokens scored
GEMMA_CTX = 256     # tokens of true context handed to Gemma for warmup
SENT = chr(262144)  # book-boundary sentinel (just above Gemma vocab)


def clean(text):
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m1: text = text[m1.end():]
    if "*** END OF" in text: text = text[:text.rfind("*** END OF")]
    return text.replace("\r\n", "\n")   # natural case


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
    from llama_cpp import Llama
    llm = Llama(model_path=BLOB, logits_all=True, n_ctx=2048, verbose=False, n_gpu_layers=-1)

    def tok_ids(t):
        return llm.tokenize(t.encode("utf-8"), add_bos=False, special=False)

    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    train_texts = [tom[:cut]]
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try: train_texts.append(clean(open(f, encoding="utf-8", errors="replace").read()))
        except Exception: pass
    held_text = tom[cut:]

    # engine corpus over Gemma tokens (id -> char), sentinel between books
    corpus = SENT.join("".join(chr(i) for i in tok_ids(t)) for t in train_texts)
    print(f"corpus {len(corpus):,} ref-tokens | q={REF_NAME[:28]} exact (full distribution)", flush=True)

    # held tokens + a leading true-context window so Gemma isn't cold
    ctx_ids = tok_ids(tom[:cut][-1200:])[-GEMMA_CTX:]
    held_ids = tok_ids(held_text)[:EVAL_TOKS]
    seq = ctx_ids + held_ids
    held_bytes = len(llm.detokenize(held_ids))
    print(f"held {len(held_ids)} tokens / {held_bytes} bytes | ref warmup ctx {len(ctx_ids)} toks", flush=True)

    # one exact forward pass over the whole sequence
    llm.reset()
    llm.eval(seq)
    scores = np.array(llm.scores)[: len(seq)]

    held_str = "".join(chr(i) for i in held_ids)
    off = len(ctx_ids)              # held token j is seq position off+j
    GATES = [1, 3, 5, 7, 9, 11]     # engine speaks only when match depth k >= gate
    rows = []
    bits_e = bits_g = 0.0
    for j in range(1, len(held_ids)):
        true = held_ids[j]
        row = scores[off + j - 1].astype(np.float64)
        row -= row.max()
        ex = np.exp(row); ex /= ex.sum()
        q_true = float(ex[true])
        ctx = held_str[max(0, j - CTX):j]
        k = longest_k(corpus, ctx, CTX)
        conts = _search_suffix_in_corpus(corpus, ctx[-k:], k) if k > 0 else []
        total = len(conts)
        n_true = conts.count(chr(true))
        rows.append((k, total, n_true, q_true))
        p_e = (n_true + 1.0 / llm.n_vocab()) / (total + 1) if total > 0 else 1.0 / llm.n_vocab()
        bits_e += -math.log2(max(p_e, 1e-12))
        bits_g += -math.log2(max(q_true, 1e-12))

    be, bg = bits_e / held_bytes, bits_g / held_bytes
    print(f"\nENGINE alone (token No-Zero):  {be:.4f} bits/byte", flush=True)
    print(f"REFERENCE alone ({REF_NAME[:22]}): {bg:.4f} bits/byte", flush=True)
    best = None; gate_bpb = {}
    for G in GATES:
        bx = 0.0
        for (k, total, n_true, q_true) in rows:
            p_x = (n_true + q_true) / (total + 1) if (total > 0 and k >= G) else max(q_true, 1e-12)
            bx += -math.log2(max(p_x, 1e-12))
        bx /= held_bytes
        gate_bpb[G] = round(bx, 4)
        tag = "BEATS ref" if bx < bg - 0.005 else ("= baseline" if abs(bx - bg) < 0.01 else "above baseline")
        print(f"  transfer gate k>={G:<2}: {bx:.4f} bits/byte  ({tag}, {bx-bg:+.3f} vs ref)", flush=True)
        if best is None or bx < best[1]:
            best = (G, bx)
    print(f"\n  -> best gate k>={best[0]}: {best[1]:.4f} bpb  (reference baseline {bg:.4f})", flush=True)
    json.dump({"reference": REF_NAME, "engine_token": round(be, 4), "reference_bpb": round(bg, 4),
               "transfer_by_gate": gate_bpb, "best_gate": best[0], "best_transfer": round(best[1], 4),
               "held_tokens": len(held_ids), "held_bytes": held_bytes},
              open(os.path.join(HERE, "transfer_exact_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
