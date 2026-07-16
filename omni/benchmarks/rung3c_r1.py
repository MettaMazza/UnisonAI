"""RUNG 3c: fold-basis compression retested on the RIGHT patient --
DeepSeek-R1's loud tensors (43-47x every block). Registered in PROTOCOL.md.
Metric (registered): reconstruction fidelity -- relative MSE + energy
retained -- vs matched-storage uniform quantization, keep = 0.25 / 0.125,
per row-block (the corrected instrument's own coordinates)."""
import numpy as np, sys
sys.path.insert(0, ".")
from spectral_probe import fwht
import gguf
from gguf.quants import dequantize

BLOCK = 1 << 22
R1 = "/Volumes/One Touch/models library/GGUF_Models/DeepSeek-R1-Q4/DeepSeek-R1-Q4_K_M/DeepSeek-R1-Q4_K_M-00001-of-00009.gguf"

def fold_block(v, keep):
    n = len(v)
    spec = fwht(v.copy())
    k = max(1, int(n * keep))
    idx = np.argpartition(np.abs(spec), n - k)[:n - k]
    spec[idx] = 0.0
    return fwht(spec) / n

def quant_block(v, bits):
    levels = max(2, int(2 ** bits))
    lo, hi = v.min(), v.max()
    if hi == lo:
        return v.copy()
    q = np.round((v - lo) / (hi - lo) * (levels - 1))
    return q / (levels - 1) * (hi - lo) + lo

def compare(name, w):
    v = w.astype(np.float64)
    rows_per_block = max(1, min(v.shape[0], BLOCK // v.shape[1]))
    n_fit = 1 << int(np.floor(np.log2(rows_per_block * v.shape[1])))
    blocks = []
    r = 0
    while r + rows_per_block <= v.shape[0] and len(blocks) < 8:
        blocks.append(v[r:r + rows_per_block].ravel()[:n_fit].copy())
        r += rows_per_block
    for keep in (0.25, 0.125):
        bits = keep * (np.log2(n_fit) + 16)
        f_mse, q_mse, f_en = [], [], []
        for b in blocks:
            e = np.sum(b * b)
            fr = fold_block(b, keep)
            qr = quant_block(b, bits)
            f_mse.append(np.sum((b - fr) ** 2) / e)
            q_mse.append(np.sum((b - qr) ** 2) / e)
            f_en.append(np.sum(fr * fr) / e)
        fm, qm = float(np.median(f_mse)), float(np.median(q_mse))
        verdict = "FOLD WINS" if fm < qm else "quant wins"
        print(f"{name:34s} keep={keep:5.3f} (={bits:4.1f}b)  fold-relMSE {fm:.4e}  quant-relMSE {qm:.4e}  -> {verdict}", flush=True)

if __name__ == "__main__":
    r = gguf.GGUFReader(R1)
    tens = {t.name: t for t in r.tensors}
    for key in ("blk.0.ffn_gate.weight", "blk.5.ffn_gate_shexp.weight", "blk.3.ffn_up_shexp.weight"):
        if key in tens:
            t = tens[key]
            w = np.asarray(dequantize(t.data, t.tensor_type))
            shape = tuple(int(x) for x in t.shape)
            w = w.reshape(shape[-2], shape[-1]) if len(shape) == 2 else w.reshape(-1, shape[-1])
            compare(f"R1 {key}", w)
    print("RUNG 3c COMPLETE", flush=True)
