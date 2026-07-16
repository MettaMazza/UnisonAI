"""THE TRANSFER-IN, token-aligned (the correct instrument).

Both the engine and GPT-2 predict the SAME next GPT-2 token, so their
distributions combine cleanly (Rung 5d):

  p(tok) = ( n(tok) + q_gpt2(tok|ctx) ) / (total + 1)

The engine is a counted suffix model over GPT-2's TOKENS (each token id mapped to
one unicode char so the engine's own suffix search reuses unchanged). n(tok) =
counted continuations of the longest held token-suffix; q = GPT-2's next-token
distribution (one forward pass over the held sequence, all positions). total =
counted continuations. q=uniform reproduces the base engine. Bits-per-byte uses
the true UTF-8 byte length of the held text, so it is comparable to GPT-2 alone.
"""
import os, sys, math, re, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation

HERE = os.path.dirname(os.path.abspath(__file__))
SENT = chr(50257)   # a non-vocab token id as the boundary sentinel
CTX = 12            # token-suffix depth
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
    import torch, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    train_texts = [tom[:cut]]
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try: train_texts.append(clean(open(f, encoding="utf-8", errors="replace").read()))
        except Exception: pass
    held_text = tom[cut:]

    # tokenise -> map each token id to one char; sentinel between books
    def to_tokchars(t):
        return "".join(chr(i) for i in tok(t).input_ids)
    corpus = SENT.join(to_tokchars(t) for t in train_texts)
    held_ids = tok(held_text).input_ids[:EVAL_TOKS]
    held_str = "".join(chr(i) for i in held_ids)
    held_bytes = len(tok.decode(held_ids).encode("utf-8"))
    Vtok = model.config.vocab_size
    print(f"corpus {len(corpus):,} tokens | held-out {len(held_ids)} tokens / {held_bytes} bytes", flush=True)

    # GPT-2 next-token distribution at every held position (one pass, sliding if long)
    ids_t = torch.tensor([held_ids])
    with torch.no_grad():
        logits = model(ids_t).logits[0]           # [T, V]
    logq = F.log_softmax(logits.double(), dim=-1)  # log q at each position -> predicts next
    q = logq.exp().numpy()

    # One pass: record per-position (k, total, n_true, q_true). Then evaluate the
    # transfer under several depth GATES G -- the engine's counts are used only when
    # the matched context depth k >= G (a specific-enough memory to be trustworthy);
    # below G we defer fully to q. G=1 is the ungated forced transfer.
    GATES = [1, 3, 5, 7, 9, 11]
    rows = []
    bits_e = bits_g = 0.0
    for i in range(1, len(held_ids)):
        true = held_ids[i]
        ctx = held_str[max(0, i - CTX):i]
        k = longest_k(corpus, ctx, CTX)
        conts = _search_suffix_in_corpus(corpus, ctx[-k:], k) if k > 0 else []
        total = len(conts)
        n_true = conts.count(chr(true))
        q_true = float(q[i - 1, true])
        rows.append((k, total, n_true, q_true))
        p_e = (n_true + 1.0 / Vtok) / (total + 1) if total > 0 else 1.0 / Vtok
        bits_e += -math.log2(max(p_e, 1e-12))
        bits_g += -math.log2(max(q_true, 1e-12))

    be, bg = bits_e / held_bytes, bits_g / held_bytes
    print(f"\nENGINE alone (token n-gram):   {be:.4f} bits/byte", flush=True)
    print(f"GPT-2 alone (baseline):        {bg:.4f} bits/byte", flush=True)
    best = None
    gate_bpb = {}
    for G in GATES:
        bx = 0.0
        for (k, total, n_true, q_true) in rows:
            if total > 0 and k >= G:
                p_x = (n_true + q_true) / (total + 1)
            else:
                p_x = max(q_true, 1e-12)          # defer fully to GPT-2
            bx += -math.log2(max(p_x, 1e-12))
        bx /= held_bytes
        gate_bpb[G] = round(bx, 4)
        tag = "BEATS GPT-2" if bx < bg else ("= baseline" if abs(bx - bg) < 0.01 else "above baseline")
        print(f"  transfer gate k>={G:<2}: {bx:.4f} bits/byte  ({tag}, {bx-bg:+.3f} vs GPT-2)", flush=True)
        if best is None or bx < best[1]:
            best = (G, bx)
    print(f"\n  -> best gate k>={best[0]}: {best[1]:.4f} bpb  (GPT-2 baseline {bg:.4f})", flush=True)
    json.dump({"engine": round(be, 4), "gpt2": round(bg, 4),
               "transfer_by_gate": gate_bpb, "best_gate": best[0],
               "best_transfer": round(best[1], 4)},
              open(os.path.join(HERE, "transfer_tokens_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
