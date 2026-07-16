"""Bits-per-byte of GPT-2 and Gemma-4 on the SAME held-out slice as the engine.
Tokenizer-independent: total bits / total UTF-8 bytes, so it's a fair cross-model
comparison. GPT-2 via transformers (sliding window); Gemma-4 via Ollama logprobs.
"""
import os, sys, math, re, json

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "corpus_raw.txt")
EVAL_CHARS = 4000


def clean(text):
    m1 = re.search(r"\*\*\* START OF.*?\*\*\*", text, re.S)
    if m1: text = text[m1.end():]
    if "*** END OF" in text: text = text[:text.rfind("*** END OF")]
    return text.replace("\r\n", "\n").lower()


def held_out():
    text = clean(open(RAW, encoding="utf-8").read())
    cut = int(len(text) * 0.90)
    return text[cut:cut + EVAL_CHARS]


def gpt2_bpb(text):
    import torch, torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = tok(text, return_tensors="pt").input_ids[0]
    win, stride = 1024, 512
    nll = 0.0; ntok = 0
    with torch.no_grad():
        for start in range(0, len(ids), stride):
            chunk = ids[start:start + win]
            if len(chunk) < 2:
                break
            logits = model(chunk.unsqueeze(0)).logits[0]
            lp = F.log_softmax(logits[:-1].double(), dim=-1)
            tgt = chunk[1:]
            # only score the NEW tokens in this window (avoid double-counting overlap)
            first_new = 0 if start == 0 else (win - stride) - 1
            sel = range(max(first_new, 0), len(tgt))
            for t in sel:
                nll += -lp[t, tgt[t]].item()
                ntok += 1
            if start + win >= len(ids):
                break
    total_bits = nll / math.log(2)
    return total_bits / len(text.encode("utf-8"))


def ollama_bpb(text, model="gemma-4-31b:latest"):
    """Bits-per-byte via Ollama's per-token logprobs on the raw completion."""
    import urllib.request
    body = json.dumps({
        "model": model, "prompt": text, "stream": False,
        "options": {"num_predict": 0, "temperature": 0},
        "raw": True, "logprobs": True,
    }).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    except Exception as e:
        return None, f"ollama error: {e}"
    # Ollama returns prompt-eval logprobs when logprobs=true (newer builds). Try to read them.
    lps = resp.get("prompt_logprobs") or resp.get("logprobs")
    if not lps:
        return None, "no prompt logprobs returned by this Ollama build"
    nll = -sum(x for x in lps if x is not None)
    total_bits = nll / math.log(2)
    return total_bits / len(text.encode("utf-8")), "ok"


def main():
    text = held_out()
    print(f"held-out: {len(text)} chars, {len(text.encode('utf-8'))} bytes", flush=True)
    out = {}
    which = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    if which in ("gpt2", "all"):
        b = gpt2_bpb(text)
        out["gpt2"] = round(b, 4)
        print(f"GPT-2:   {b:.4f} bits/byte  (perplexity {2**b:.2f})", flush=True)
    if which in ("gemma", "all"):
        b, msg = ollama_bpb(text)
        if b is not None:
            out["gemma-4-31b"] = round(b, 4)
            print(f"Gemma-4: {b:.4f} bits/byte  (perplexity {2**b:.2f})", flush=True)
        else:
            print(f"Gemma-4: unavailable ({msg})", flush=True)
    json.dump(out, open(os.path.join(HERE, "baseline_bpb.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
