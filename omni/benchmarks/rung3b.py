"""RUNG 3b: the chess constructions applied to weight compression.
Arm A: packing sweep (row / column / Morton) for truncation quality.
Arm B: spectrum + exceptions vs quantization at matched bits.
Registered in PROTOCOL.md before any measurement."""
import numpy as np, torch, sys, copy
sys.path.insert(0, ".")
from spectral_probe import fwht
from rung3_compress import PROMPTS, judge

def morton_perm(rows, cols):
    """Bit-interleaved (Z-order) permutation for a rows x cols matrix,
    both dimensions powers of two required for exact interleave; we use
    the largest 2^a x 2^b sub-grid and append the remainder in row order."""
    ra = 1 << int(np.floor(np.log2(rows)))
    ca = 1 << int(np.floor(np.log2(cols)))
    r = np.arange(ra, dtype=np.int64)
    c = np.arange(ca, dtype=np.int64)
    def spread(x, nbits, offset):
        out = np.zeros_like(x)
        for b in range(nbits):
            out |= ((x >> b) & 1) << (2 * b + offset)
        return out
    rb, cb = int(np.log2(ra)), int(np.log2(ca))
    nb = max(rb, cb)
    R = spread(r, rb, 1)[:, None]
    C = spread(c, cb, 0)[None, :]
    z = (R + C).ravel()
    order = np.argsort(z, kind="stable")
    grid = (np.arange(ra)[:, None] * cols + np.arange(ca)[None, :]).ravel()
    inside = grid[order]
    all_idx = np.arange(rows * cols, dtype=np.int64)
    mask = np.zeros(rows * cols, dtype=bool)
    mask[inside] = True
    return np.concatenate([inside, all_idx[~mask]])

def truncate_in_perm(w, keep_frac, perm=None):
    shape = w.shape
    v = w.astype(np.float64).ravel()
    if perm is not None:
        v = v[perm]
    n = 1 << int(np.floor(np.log2(len(v))))
    head, tail = v[:n].copy(), v[n:].copy()
    spec = fwht(head.copy())
    k = max(1, int(n * keep_frac))
    idx = np.argpartition(np.abs(spec), n - k)[:n - k]
    spec[idx] = 0.0
    rec = np.concatenate([fwht(spec) / n, tail])
    if perm is not None:
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        rec = rec[inv]
    return rec.reshape(shape).astype(np.float32)

def spectrum_plus_exceptions(w, coeff_frac, exc_frac):
    """Chess compact-exact: top-k spectrum + top-m exact residual corrections."""
    shape = w.shape
    v = w.astype(np.float64).ravel()
    n = 1 << int(np.floor(np.log2(len(v))))
    head, tail = v[:n].copy(), v[n:].copy()
    spec = fwht(head.copy())
    k = max(1, int(n * coeff_frac))
    idx = np.argpartition(np.abs(spec), n - k)[:n - k]
    spec[idx] = 0.0
    rec = fwht(spec) / n
    resid = head - rec
    m = max(1, int(n * exc_frac))
    exc = np.argpartition(np.abs(resid), n - m)[n - m:]
    rec[exc] = head[exc]
    return np.concatenate([rec, tail]).reshape(shape).astype(np.float32)

def quantize_bits(w, bits):
    levels = max(2, int(2 ** bits))
    lo, hi = w.min(), w.max()
    q = np.round((w - lo) / (hi - lo) * (levels - 1))
    return (q / (levels - 1) * (hi - lo) + lo).astype(np.float32)

if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.eval()
    with torch.no_grad():
        ref = [base(**tok(p, return_tensors="pt")).logits[0, -1].clone() for p in PROMPTS]

    def apply_all(fn):
        m = copy.deepcopy(base)
        for L in range(12):
            w = m.transformer.h[L].mlp.c_fc.weight.data.numpy()
            m.transformer.h[L].mlp.c_fc.weight.data = torch.from_numpy(fn(w))
        return judge(m, tok, ref)

    print("=== ARM A: packing sweep (keep=0.25 / 0.125) ===", flush=True)
    W0 = base.transformer.h[0].mlp.c_fc.weight.data.numpy()
    mp = morton_perm(*W0.shape)
    for keep in (0.25, 0.125):
        kl, ag = apply_all(lambda w: truncate_in_perm(w, keep))
        print(f"row-major  keep={keep:6.3f}  KL={kl:8.4f}  agree={ag:5.2f}", flush=True)
        kl, ag = apply_all(lambda w: truncate_in_perm(w, keep, perm=np.argsort(
            (np.arange(w.size).reshape(w.shape).T).ravel(), kind="stable") if False else
            np.arange(w.size).reshape(w.shape).T.ravel().argsort(kind="stable")))
        print(f"col-major  keep={keep:6.3f}  KL={kl:8.4f}  agree={ag:5.2f}", flush=True)
        kl, ag = apply_all(lambda w: truncate_in_perm(w, keep, perm=mp))
        print(f"morton     keep={keep:6.3f}  KL={kl:8.4f}  agree={ag:5.2f}", flush=True)

    print("=== ARM B: spectrum + exceptions vs quantization at matched bits ===", flush=True)
    n_per = 1 << int(np.floor(np.log2(W0.size)))
    for bits in (4, 3):
        budget_entries = bits * W0.size / 32.0   # 32-bit entries affordable
        for split_name, cf in (("75/25", 0.75), ("50/50", 0.5), ("25/75", 0.25)):
            coeff_frac = cf * budget_entries / n_per
            exc_frac = (1 - cf) * budget_entries / n_per
            kl, ag = apply_all(lambda w: spectrum_plus_exceptions(w, coeff_frac, exc_frac))
            print(f"spec+exc {bits}b split={split_name}  KL={kl:8.4f}  agree={ag:5.2f}", flush=True)
        kl, ag = apply_all(lambda w: quantize_bits(w, bits))
        print(f"quantize {bits}b                KL={kl:8.4f}  agree={ag:5.2f}", flush=True)
    print("RUNG 3b COMPLETE", flush=True)
