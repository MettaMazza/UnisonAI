import numpy as np, time
from safetensors import safe_open
import sys
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, bit_reverse_perm, FRACS

SEED = 20260706
SD = "/Volumes/One Touch/models library/Creative_Models/v1-5-pruned-emaonly.safetensors"

def battery(w, nulls=3):
    w = w.astype(np.float64).ravel()
    nbits = int(np.floor(np.log2(len(w))))
    n = 1 << nbits
    w = w[:n].copy()
    real = concentration(fwht(w.copy()), FRACS)
    rng = np.random.default_rng(SEED)
    null_max = {f: 0.0 for f in FRACS}
    for s in range(nulls):
        c = concentration(fwht(w[rng.permutation(n)].copy()), FRACS)
        for f in FRACS:
            null_max[f] = max(null_max[f], c[f])
    f0 = FRACS[0]
    return real[f0], null_max[f0], real[f0] / null_max[f0]

if __name__ == "__main__":
    f = safe_open(SD, framework="numpy")
    rows = []
    for k in f.keys():
        sh = f.get_slice(k).get_shape()
        if len(sh) == 2 and sh[0] * sh[1] >= 1 << 20:
            r, nl, margin = battery(f.get_tensor(k))
            cls = ("embedding" if "embedding" in k else
                   "attention" if any(t in k for t in ("attn", ".to_q", ".to_k", ".to_v", "q_proj", "k_proj", "v_proj", "out_proj")) else
                   "ff" if any(t in k for t in ("ff.", "mlp", "fc")) else "other")
            rows.append((margin, cls, k, r, nl))
            print(f"{margin:8.2f}x  [{cls:9s}] {k}", flush=True)
    rows.sort(reverse=True)
    print("\n=== TOP 10 BY MARGIN ===", flush=True)
    for margin, cls, k, r, nl in rows[:10]:
        print(f"{margin:8.2f}x [{cls}] {k} real={r:.6f} null={nl:.6f}", flush=True)
    cls_m = {}
    for margin, cls, k, r, nl in rows:
        cls_m.setdefault(cls, []).append(margin)
    print("\n=== CLASS MEDIANS ===", flush=True)
    for cls, ms in cls_m.items():
        print(f"{cls}: median {sorted(ms)[len(ms)//2]:.2f}x over {len(ms)} tensors", flush=True)
    print("RUNG 2 ARM A COMPLETE", flush=True)
