"""RUNG 2g: THE BASIS HUNT. The quiet models' law is expressed somewhere --
find the coordinates. Fixed menu of data-independent reorderings only
(registered); wake = margin > 2x under any map. Objects: Qwen3-27B gate
(quiet at 1.03x aligned), Kimi-K2.6 shexp (quiet at 1.06x aligned).
Control: GPT-2 c_fc must stay awake under bit-reversal (theorem)."""
import numpy as np, sys
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS
import gguf
from gguf.quants import dequantize
SEED = 20260706
N = 1 << 22

def margin(v, rng):
    real = concentration(fwht(v.copy()), FRACS)
    nm = {f: 0.0 for f in FRACS}
    for s in range(3):
        c = concentration(fwht(v[rng.permutation(len(v))].copy()), FRACS)
        for f in FRACS:
            nm[f] = max(nm[f], c[f])
    return real[FRACS[0]] / nm[FRACS[0]]

def maps(n):
    idx = np.arange(n, dtype=np.int64)
    nbits = int(np.log2(n))
    rev = np.zeros_like(idx)
    for b in range(nbits):
        rev |= ((idx >> b) & 1) << (nbits - 1 - b)
    gray = idx ^ (idx >> 1)
    yield "identity", idx
    yield "bit-reversal(self-test)", rev
    yield "gray-code", gray
    yield "x3 mod n", (3 * idx) % n
    yield "x5 mod n", (5 * idx) % n
    yield "affine 3i+1", (3 * idx + 1) % n

def hunt(name, w2d):
    v_row = w2d.astype(np.float64).ravel()
    n = min(N, 1 << int(np.floor(np.log2(len(v_row)))))
    rng = np.random.default_rng(SEED)
    base = v_row[:n].copy()
    col = np.ascontiguousarray(w2d.T).astype(np.float64).ravel()[:n].copy()
    bt64 = w2d.reshape(w2d.shape[0] // 64, 64, w2d.shape[1]).transpose(1, 0, 2).astype(np.float64).ravel()[:n].copy() if w2d.shape[0] % 64 == 0 else None
    results = []
    for mname, m in maps(n):
        results.append((mname, margin(base[m], rng)))
    results.append(("transpose", margin(col, rng)))
    if bt64 is not None:
        results.append(("block-transpose-64", margin(bt64, rng)))
    for mname, mg in results:
        flag = " <-- WAKE" if mg > 2.0 and "self-test" not in mname and "identity" not in mname else ""
        print(f"  {name:28s} {mname:24s} {mg:6.2f}x{flag}", flush=True)

if __name__ == "__main__":
    from safetensors import safe_open
    f = safe_open("gpt2_model.safetensors", framework="numpy")
    hunt("CONTROL GPT-2 c_fc L0", f.get_tensor("h.0.mlp.c_fc.weight"))
    LIB = "/Volumes/One Touch/models library/GGUF_Models"
    jobs = (("Qwen3-27B gate L0", f"{LIB}/Qwen3.6-27B-Q4_K_M.gguf", "blk.0.ffn_gate.weight"),
            ("Kimi-1T shexp L5", f"{LIB}/Kimi-K2.6/UD-Q4_K_XL/Kimi-K2.6-UD-Q4_K_XL-00002-of-00014.gguf", "blk.5.ffn_gate_shexp.weight"))
    for label, path, key in jobs:
        r = gguf.GGUFReader(path)
        tens = {t.name: t for t in r.tensors}
        t = tens[key]
        w = np.asarray(dequantize(t.data, t.tensor_type))
        shape = tuple(int(x) for x in t.shape)
        w = w.reshape(shape[-2], shape[-1]) if len(shape) == 2 else w.reshape(-1, shape[-1])
        hunt(label, w)
    print("RUNG 2g COMPLETE", flush=True)
