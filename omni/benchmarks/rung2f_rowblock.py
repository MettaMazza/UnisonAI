"""Corrected scale-aware instrument (registered): per-row-block spectra,
median block margin. Control: GPT-2 (flat-window said 12.7x). Suspects:
gpt-oss-20b (flat said ~1x), Qwen3.6-27B (flat said 1.7x)."""
import numpy as np, sys
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS
SEED = 20260706
BLOCK = 1 << 22

def block_margin(v, rng):
    real = concentration(fwht(v.copy()), FRACS)
    nm = {f: 0.0 for f in FRACS}
    for s in range(3):
        c = concentration(fwht(v[rng.permutation(len(v))].copy()), FRACS)
        for f in FRACS:
            nm[f] = max(nm[f], c[f])
    f0 = FRACS[0]
    return real[f0] / nm[f0]

def probe_rowblocks(name, w):
    v = w.astype(np.float64)
    rows_per_block = max(1, min(v.shape[0], BLOCK // v.shape[1]))
    n_fit = 1 << int(np.floor(np.log2(rows_per_block * v.shape[1])))
    rng = np.random.default_rng(SEED)
    margins = []
    r = 0
    while r < v.shape[0] and r + rows_per_block <= v.shape[0] and len(margins) < 12:
        blk = v[r:r + rows_per_block].ravel()[:n_fit].copy()
        margins.append(block_margin(blk, rng))
        r += rows_per_block
    med = float(np.median(margins))
    print(f"{name:40s} blocks={len(margins)}  MEDIAN {med:6.2f}x  (min {min(margins):.2f} max {max(margins):.2f})", flush=True)

if __name__ == "__main__":
    from safetensors import safe_open
    f = safe_open("gpt2_model.safetensors", framework="numpy")
    probe_rowblocks("CONTROL GPT-2 h.0.mlp.c_fc", f.get_tensor("h.0.mlp.c_fc.weight"))
    probe_rowblocks("CONTROL GPT-2 wte", f.get_tensor("wte.weight"))
    import gguf
    from gguf.quants import dequantize
    LIB = "/Volumes/One Touch/models library/GGUF_Models"
    import glob
    q235 = sorted(glob.glob(f"{LIB}/Qwen3-235B/**/*.gguf", recursive=True))
    for name, path, keys in (
        ("gpt-oss-20b", f"{LIB}/gpt-oss-20b-Q4_K_M.gguf", None),
        ("gpt-oss-120b", f"{LIB}/gpt-oss-120b-Q4/Q4_K_M/gpt-oss-120b-Q4_K_M-00001-of-00002.gguf", None),
        ("Qwen3-235B", q235[0] if q235 else "", None),
        ("Qwen3-480B", f"{LIB}/Qwen3-Coder-480B/Q4_K_M/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00006.gguf", None),
        ("DeepSeek-R1-671B", f"{LIB}/DeepSeek-R1-Q4/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf", None),
        ("Kimi-K2.6-1T", f"{LIB}/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00001-of-00014.gguf", None)):
        if not path: continue
        r = gguf.GGUFReader(path)
        cands = [t for t in r.tensors if ("ffn_gate" in t.name or "ffn_up" in t.name)
                 and t.name.endswith(".weight")
                 and "inp" not in t.name and "norm" not in t.name and "bias" not in t.name]
        for t in (cands[0], cands[len(cands)//2]):
            w = np.asarray(dequantize(t.data, t.tensor_type)).ravel()
            shape = tuple(int(x) for x in t.shape)
            if len(shape) == 3:
                per = shape[1] * shape[2]
                w = w[:per].reshape(shape[1], shape[2])
            else:
                w = w.reshape(shape[-2], shape[-1])
            probe_rowblocks(f"{name} {t.name}", w)
    print("ROWBLOCK CHECK COMPLETE", flush=True)
