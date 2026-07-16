"""RUNG 2f: THE SCALING SURVEY -- the law-fingerprint across the frontier
scale axis, 4B -> ~1T, from Maria's local library. Registered in PROTOCOL.md.
Per model: expansion (gate/up) FFN tensors at available depths; for MoE,
individual expert slices. Locked battery, seed 20260706."""
import numpy as np, sys, gc
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS
import gguf
from gguf.quants import dequantize

SEED = 20260706
LIB = "/Volumes/One Touch/models library/GGUF_Models"
MODELS = [
    ("Qwen3.5-4B",      4e9,   f"{LIB}/Qwen3.5-4B-Q4_K_M.gguf"),
    ("Llama-3.1-8B",    8e9,   f"{LIB}/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    ("gpt-oss-20b",     2e10,  f"{LIB}/gpt-oss-20b-Q4_K_M.gguf"),
    ("Qwen3.6-27B",     2.7e10,f"{LIB}/Qwen3.6-27B-Q4_K_M.gguf"),
    ("Llama-3.3-70B",   7e10,  f"{LIB}/Llama-3.3-70B-Instruct-Q4_K_M.gguf"),
    ("gpt-oss-120b",    1.2e11,f"{LIB}/gpt-oss-120b-Q4/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf"),
    ("Qwen3-235B",      2.35e11, None),  # resolve shard below
    ("Qwen3-480B",      4.8e11,f"{LIB}/Qwen3-Coder-480B/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf"),
    ("DeepSeek-R1-671B",6.7e11,f"{LIB}/DeepSeek-R1-Q4/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf"),
    ("Kimi-K2.6-1T",    1e12,  f"{LIB}/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf"),
]
import glob, os
q235 = sorted(glob.glob(f"{LIB}/Qwen3-235B/**/*.gguf", recursive=True))
MODELS = [(n, p, q235[0] if n == "Qwen3-235B" and q235 else f) for (n, p, f) in MODELS]

def margin(w, nulls=3):
    v = w.astype(np.float64).ravel()
    n = 1 << int(np.floor(np.log2(len(v))))
    if n > 1 << 26:
        n = 1 << 26   # cap per-tensor cost; registered fractions are of the probed space
    v = v[:n].copy()
    real = concentration(fwht(v.copy()), FRACS)
    rng = np.random.default_rng(SEED)
    nm = {f: 0.0 for f in FRACS}
    for s in range(nulls):
        c = concentration(fwht(v[rng.permutation(n)].copy()), FRACS)
        for f in FRACS:
            nm[f] = max(nm[f], c[f])
    f0 = FRACS[0]
    return real[f0] / nm[f0]

def probe_model(name, params, path):
    try:
        r = gguf.GGUFReader(path)
    except Exception as e:
        print(f"{name}: READ FAIL {e}", flush=True)
        return
    cands = [t for t in r.tensors
             if ("ffn_gate" in t.name or "ffn_up" in t.name) and "inp" not in t.name
             and "norm" not in t.name]
    if not cands:
        print(f"{name}: no ffn tensors in shard", flush=True)
        return
    picks = [cands[0], cands[len(cands) // 2], cands[-1]][:3]
    best = 0.0
    for t in picks:
        try:
            w = dequantize(t.data, t.tensor_type)
            shape = tuple(int(x) for x in t.shape)
            w = w.reshape(-1) if len(shape) < 3 else w.reshape(shape[0] if shape[0] < shape[-1] else -1, -1)[0]
            m = margin(np.asarray(w))
            best = max(best, m)
            tag = "expert0" if "exps" in t.name else "dense"
            print(f"  {name:18s} {t.name:34s} [{tag}] margin {m:7.2f}x", flush=True)
            del w
            gc.collect()
        except Exception as e:
            print(f"  {name} {t.name}: FAIL {e}", flush=True)
    print(f"SCALE-POINT {name} ({params:.0e} params): best margin {best:.2f}x", flush=True)

if __name__ == "__main__":
    for name, params, path in MODELS:
        if path and os.path.exists(path):
            probe_model(name, params, path)
        else:
            print(f"{name}: path missing, skipped", flush=True)
    print("RUNG 2f COMPLETE", flush=True)
