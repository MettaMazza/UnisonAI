import numpy as np, sys
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS
from safetensors import safe_open

SEED = 20260706

def margin(w, nulls=3):
    w = w.astype(np.float64).ravel()
    n = 1 << int(np.floor(np.log2(len(w))))
    w = w[:n].copy()
    real = concentration(fwht(w.copy()), FRACS)
    rng = np.random.default_rng(SEED)
    nm = {f: 0.0 for f in FRACS}
    for s in range(nulls):
        c = concentration(fwht(w[rng.permutation(n)].copy()), FRACS)
        for f in FRACS:
            nm[f] = max(nm[f], c[f])
    f0 = FRACS[0]
    return real[f0] / nm[f0]

def both_packings(name, w):
    rm = margin(w)
    cm = margin(np.ascontiguousarray(w.T))
    print(f"{name:55s} row {rm:7.2f}x   col {cm:7.2f}x", flush=True)
    return max(rm, cm)

if __name__ == "__main__":
    print("=== ARM D: GPT-2 (full-precision canonical LLM) ===", flush=True)
    f = safe_open("gpt2_model.safetensors", framework="numpy")
    best = []
    best.append(both_packings("wte (token embedding)", f.get_tensor("wte.weight")))
    for L in range(12):
        for part in ("c_fc", "c_proj"):
            k = f"h.{L}.mlp.{part}.weight"
            best.append(both_packings(f"L{L} mlp.{part}", f.get_tensor(k)))
    print(f"GPT-2 best margin: {max(best):.2f}x over {len(best)} tensors", flush=True)

    print("=== ARM E: Llama-3.1-8B dequantized (Q4_K/Q6_K survival) ===", flush=True)
    import gguf
    from gguf.quants import dequantize
    r = gguf.GGUFReader("/Users/mettamazza/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    tens = {t.name: t for t in r.tensors}
    for L in (0, 8, 16, 24, 31):
        for part in ("ffn_gate", "ffn_down"):
            t = tens[f"blk.{L}.{part}.weight"]
            w = dequantize(t.data, t.tensor_type).reshape(tuple(int(x) for x in t.shape))
            both_packings(f"L{L} {part} [{t.tensor_type.name}]", w)
    print("ARMS D+E COMPLETE", flush=True)
