"""RUNG 3: cash the law as compression. Registered in PROTOCOL.md.
Fold-basis truncation of GPT-2's law-bearing matrices vs matched-budget
uniform quantization, judged by next-token KL + top-1 agreement on a
fixed prompt set. Usage: python3 rung3_compress.py"""
import numpy as np, torch, sys, copy
sys.path.insert(0, ".")
from spectral_probe import fwht

PROMPTS = [
    "The capital of France is", "Two plus two equals",
    "The theory of relativity was developed by", "Water is made of hydrogen and",
    "In the beginning was the", "The stock market fell because",
    "To be or not to be, that is the", "The chess player moved her queen to",
    "Neural networks learn by adjusting their", "The speed of light is approximately",
    "Once upon a time there lived a", "The chemical symbol for gold is",
    "Photosynthesis converts sunlight into", "The largest planet in our solar system is",
    "She opened the door and saw", "The algorithm terminates when"]

def fold_truncate(w, keep_frac):
    """Keep top-k Walsh coefficients of the flattened matrix, invert."""
    shape = w.shape
    v = w.astype(np.float64).ravel()
    n = 1 << int(np.floor(np.log2(len(v))))
    head, tail = v[:n].copy(), v[n:].copy()
    spec = fwht(head.copy())
    k = max(1, int(n * keep_frac))
    idx = np.argpartition(np.abs(spec), n - k)[:n - k]
    spec[idx] = 0.0
    rec = fwht(spec) / n
    out = np.concatenate([rec, tail]).reshape(shape)
    return out.astype(np.float32)

def quantize_matched(w, keep_frac):
    """Uniform round-to-nearest at the bit-width matching the truncation's
    storage: k coeffs x (log2(n) index bits + 16 value bits) over n weights."""
    n = w.size
    bits_per_weight = keep_frac * (np.log2(n) + 16)
    levels = max(2, int(2 ** min(16, bits_per_weight)))
    lo, hi = w.min(), w.max()
    q = np.round((w - lo) / (hi - lo) * (levels - 1))
    return (q / (levels - 1) * (hi - lo) + lo).astype(np.float32)

@torch.no_grad()
def judge(model, tok, ref_logits):
    kls, agree = [], []
    for i, p in enumerate(PROMPTS):
        ids = tok(p, return_tensors="pt")
        lg = model(**ids).logits[0, -1]
        ref = ref_logits[i]
        lp, rp = torch.log_softmax(lg, -1), torch.softmax(ref, -1)
        kls.append(torch.sum(rp * (torch.log(rp + 1e-12) - lp)).item())
        agree.append(int(lg.argmax() == ref.argmax()))
    return float(np.mean(kls)), float(np.mean(agree))

if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.eval()
    with torch.no_grad():
        ref_logits = [base(**tok(p, return_tensors="pt")).logits[0, -1].clone() for p in PROMPTS]
    print("reference logits computed on 16 fixed prompts", flush=True)

    def run(mode, target, keep):
        m = copy.deepcopy(base)
        for L in range(12):
            w = m.transformer.h[L].mlp.c_fc.weight.data.numpy() if target == "c_fc" \
                else m.transformer.h[L].mlp.c_proj.weight.data.numpy()
            new = fold_truncate(w, keep) if mode == "fold" else quantize_matched(w, keep)
            if target == "c_fc":
                m.transformer.h[L].mlp.c_fc.weight.data = torch.from_numpy(new)
            else:
                m.transformer.h[L].mlp.c_proj.weight.data = torch.from_numpy(new)
        kl, ag = judge(m, tok, ref_logits)
        print(f"{mode:6s} {target:6s} keep={keep:7.4f}  KL={kl:8.4f}  top1-agree={ag:5.2f}", flush=True)
        return kl, ag

    print("=== LAW-BEARING CLASS (c_fc, all 12 layers) ===", flush=True)
    wins = 0
    for keep in (0.5, 0.25, 0.125, 0.0625):
        fk, fa = run("fold", "c_fc", keep)
        qk, qa = run("quant", "c_fc", keep)
        if fk < qk and fa >= qa:
            wins += 1
    print(f"fold beats matched quantization at {wins}/4 budgets (rung taken at >=2)", flush=True)
    print("=== CONTROL: LAW-QUIET CLASS (c_proj) at keep=0.25 ===", flush=True)
    run("fold", "c_proj", 0.25)
    run("fold", "c_fc", 0.25)
    print("RUNG 3 COMPLETE", flush=True)
