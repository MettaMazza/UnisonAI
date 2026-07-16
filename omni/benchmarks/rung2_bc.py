import numpy as np
from safetensors import safe_open
import sys
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS

SEED = 20260706
SD = "/Volumes/One Touch/models library/Creative_Models/v1-5-pruned-emaonly.safetensors"
STRONG = "cond_stage_model.transformer.text_model.encoder.layers.3.mlp.fc1.weight"
THIN = "model.diffusion_model.output_blocks.8.1.transformer_blocks.0.ff.net.0.proj.weight"

def battery(w, nulls=3):
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

if __name__ == "__main__":
    f = safe_open(SD, framework="numpy")
    print("=== ARM B: packings (margin at 6.1e-5) ===", flush=True)
    for name, key in (("STRONG(text fc1 L3)", STRONG), ("THIN(vision ff)", THIN)):
        w = f.get_tensor(key)
        print(f"{name} row-major   : {battery(w):8.2f}x", flush=True)
        print(f"{name} column-major: {battery(np.asfortranarray(w).T.copy()):8.2f}x", flush=True)
    print("=== ARM C: untrained negative control (He-init, matched shapes) ===", flush=True)
    rng = np.random.default_rng(SEED + 1)
    for name, key in (("shape-of-STRONG", STRONG), ("shape-of-THIN", THIN)):
        sh = f.get_slice(key).get_shape()
        he = rng.normal(0, np.sqrt(2.0 / sh[1]), size=sh)
        print(f"{name}: {battery(he):8.2f}x  (must sit ~1.0x)", flush=True)
    print("RUNG 2 ARMS B+C COMPLETE", flush=True)
