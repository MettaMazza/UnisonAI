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

def blockquant(v, qblock=32, bits=4):
    x = v.reshape(-1, qblock)
    scale = np.abs(x).max(axis=1, keepdims=True)
    scale[scale == 0] = 1
    levels = (1 << (bits - 1)) - 1
    q = np.round(x / scale * levels)
    return (q / levels * scale).ravel()

def compare(name, w):
    v = w.astype(np.float64)
    rows_per_block = max(1, min(v.shape[0], BLOCK // v.shape[1]))
    n_fit = 1 << int(np.floor(np.log2(rows_per_block * v.shape[1])))
    keep = 4.5 / (np.log2(n_fit) + 16)
    blocks, r = [], 0
    while r + rows_per_block <= v.shape[0] and len(blocks) < 8:
        blocks.append(v[r:r + rows_per_block].ravel()[:n_fit].copy())
        r += rows_per_block
    f_mse, q_mse = [], []
    for b in blocks:
        e = np.sum(b * b)
        f_mse.append(np.sum((b - fold_block(b, keep)) ** 2) / e)
        q_mse.append(np.sum((b - blockquant(b)) ** 2) / e)
    fm, qm = float(np.median(f_mse)), float(np.median(q_mse))
    print(f"{name:34s} keep={keep:.4f}  fold {fm:.4e}  blockquant4.5b {qm:.4e}  -> {'FOLD WINS' if fm < qm else 'blockquant wins'}", flush=True)

if __name__ == "__main__":
    r = gguf.GGUFReader(R1)
    tens = {t.name: t for t in r.tensors}
    for key in ("blk.0.ffn_gate.weight", "blk.5.ffn_gate_shexp.weight", "blk.3.ffn_up_shexp.weight"):
        t = tens[key]
        w = np.asarray(dequantize(t.data, t.tensor_type))
        shape = tuple(int(x) for x in t.shape)
        w = w.reshape(shape[-2], shape[-1]) if len(shape) == 2 else w.reshape(-1, shape[-1])
        compare(f"R1 {key}", w)
    print("RUNG 3c-II COMPLETE", flush=True)
