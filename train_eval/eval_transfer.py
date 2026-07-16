"""THE TRANSFER-IN (Rung 5d), byte-aligned: the engine's counted shares where it
has seen the context, a reference model's distribution q where it hasn't.

  p(byte) = ( n(byte) + q(byte|ctx) ) / (total + 1)

n(byte) = counted continuations of the longest held suffix (0 if unseen);
q(byte) = the reference's next-BYTE distribution (marginalised from its tokens),
which carries the reference's generalisation. total = counted continuations.
When total=0 (unseen context), p = q exactly -- the engine borrows the reference's
generalisation on the tail it would otherwise babble. q=uniform reproduces the
base engine (self-test). Reference here: GPT-2 (local, fast). Measured in
bits-per-byte on the same held-out slice (engine alone 3.18, GPT-2 alone 1.43).
"""
import os, sys, math, re, glob, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import _search_suffix_in_corpus, _has_continuation, _SENTINEL

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = 24
GPT2_CTX_CHARS = 200
EVAL_CHARS = 800


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

    # token id -> first byte of its decoded text (for byte marginalisation)
    byte_decoder = {v: k for k, v in tok.byte_encoder.items()} if hasattr(tok, "byte_encoder") else None
    import functools
    @functools.lru_cache(maxsize=None)
    def first_byte(tid):
        s = tok.convert_ids_to_tokens(tid)
        # GPT-2 byte-BPE: map the unicode-escaped token back to bytes, take first
        try:
            bs = bytearray(tok.byte_decoder[c] for c in s)
            return bs[0] if bs else None
        except Exception:
            return None

    # precompute first-byte for every token id
    V = model.config.vocab_size
    fb = [first_byte(t) for t in range(V)]
    import numpy as np
    fb_arr = np.array([b if b is not None else -1 for b in fb])

    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    cut = int(len(tom) * 0.90)
    train = [tom[:cut]]
    for f in sorted(glob.glob(os.path.join(HERE, "book_*.txt"))):
        try: train.append(clean(open(f, encoding="utf-8", errors="replace").read()))
        except Exception: pass
    corpus = _SENTINEL.join(train)
    held = tom[cut:cut + EVAL_CHARS]
    Vc = len(set(corpus) - {_SENTINEL})
    print(f"corpus {len(corpus):,} chars | held-out {len(held)} | transfer q=GPT-2", flush=True)

    def gpt2_byte_dist(ctx_str):
        ids = tok(ctx_str, return_tensors="pt").input_ids
        if ids.shape[1] == 0:
            ids = tok(" ", return_tensors="pt").input_ids
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
        p = F.softmax(logits.double(), dim=-1).numpy()
        q = np.zeros(256)
        np.add.at(q, fb_arr[fb_arr >= 0], p[fb_arr >= 0])
        s = q.sum()
        return q / s if s > 0 else q

    bits_x = 0.0   # engine + transfer
    bits_e = 0.0   # engine alone (base)
    for i in range(len(held)):
        c = held[i]; cb = ord(c) if ord(c) < 256 else None
        ctx = held[max(0, i - CTX):i]
        k = longest_k(corpus, ctx, CTX)
        conts = _search_suffix_in_corpus(corpus, ctx[-k:], k) if k > 0 else []
        total = len(conts)
        n_true = conts.count(c)
        # engine-alone prob (No-Zero floor)
        p_e = (n_true + 1.0 / Vc) / (total + 1) if total > 0 else 1.0 / Vc
        # transfer-in: reshape floor by GPT-2's byte dist
        q = gpt2_byte_dist(held[max(0, i - GPT2_CTX_CHARS):i])
        q_true = q[cb] if cb is not None else 0.0
        p_x = (n_true + q_true) / (total + 1) if total > 0 else max(q_true, 1e-9)
        bits_e += -math.log2(max(p_e, 1e-12))
        bits_x += -math.log2(max(p_x, 1e-12))
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(held)}  engine={bits_e/(i+1):.3f}  transfer={bits_x/(i+1):.3f}", flush=True)

    be, bx = bits_e / len(held), bits_x / len(held)
    print(f"\nENGINE alone:        {be:.4f} bits/byte", flush=True)
    print(f"ENGINE + GPT-2 q:    {bx:.4f} bits/byte  (GPT-2 alone 1.43)", flush=True)
    print(f"  -> transfer-in {'BEATS' if bx < 1.43 else 'does not yet beat'} GPT-2", flush=True)
    json.dump({"engine": round(be, 4), "engine_plus_gpt2_transfer": round(bx, 4), "gpt2_alone": 1.43},
              open(os.path.join(HERE, "transfer_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
