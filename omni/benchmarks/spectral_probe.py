"""FOLD-AI Rung 1: trained neural weights in the fold's spectral basis.
Pre-registered in PROTOCOL.md (fixed before any spectrum was computed).
Usage: python3 spectral_probe.py
"""
import sys, os, time
import numpy as np

SEED = 20260706
FRACS = (6.1e-5, 4.9e-4, 3.9e-3)
SAE = "/Users/mettamazza/.ernosagent/sae_training/gemma4_sae_1m.safetensors"


def fwht(a):
    """In-place float64 Walsh-Hadamard, natural order."""
    n = len(a)
    h = 1
    while h < n:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h].copy()
        a[:, :h] += a[:, h:]
        a[:, h:] = x - a[:, h:]
        a = a.reshape(-1)
        h *= 2
    return a


def concentration(spec, fracs):
    e = spec.astype(np.float64) ** 2
    total = e.sum()
    e.sort()
    e = e[::-1]
    out = {}
    for f in fracs:
        k = max(1, int(len(e) * f))
        out[f] = float(e[:k].sum() / total)
    return out


def bit_reverse_perm(nbits):
    idx = np.arange(1 << nbits, dtype=np.int64)
    rev = np.zeros_like(idx)
    for b in range(nbits):
        rev |= ((idx >> b) & 1) << (nbits - 1 - b)
    return rev


def probe(name, w):
    w = w.astype(np.float64).ravel()
    nbits = int(np.floor(np.log2(len(w))))
    n = 1 << nbits
    w = w[:n].copy()
    print(f"\n=== {name}: {n} coefficients (2^{nbits}) ===", flush=True)
    t0 = time.time()
    real = concentration(fwht(w.copy()), FRACS)

    # theorem-forced self-test: bit-reversal is F2-linear -> identical C(k)
    st = concentration(fwht(w[bit_reverse_perm(nbits)].copy()), FRACS)
    ok = all(abs(st[f] - real[f]) < 1e-9 for f in FRACS)
    print(f"self-test (bit-reversal, must match exactly): {'PASS' if ok else 'FAIL -- RUN VOID'}", flush=True)

    rng = np.random.default_rng(SEED)
    null_max = {f: 0.0 for f in FRACS}
    for s in range(5):
        p = rng.permutation(n)
        c = concentration(fwht(w[p].copy()), FRACS)
        for f in FRACS:
            null_max[f] = max(null_max[f], c[f])
    g = rng.normal(w.mean(), w.std(), n)
    gauss = concentration(fwht(g), FRACS)

    print(f"{'fraction':>10} {'real':>10} {'shuffle-max':>12} {'gaussian':>10} {'verdict':>10}", flush=True)
    for f in FRACS:
        beyond = real[f] > null_max[f] and real[f] > gauss[f]
        print(f"{f:>10.1e} {real[f]:>10.6f} {null_max[f]:>12.6f} {gauss[f]:>10.6f} "
              f"{'STRUCTURE' if beyond else 'null-level':>10}", flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    from safetensors import safe_open
    f = safe_open(SAE, framework="numpy")
    for key in ("W_enc", "W_dec"):
        probe(f"SAE {key} (Maria's trained run)", f.get_tensor(key))

    # Kokoro-82M: the largest 2D matrices from the local HF snapshot
    import glob
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/*.safetensors")) + \
        glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/*/*.pth"))
    if snaps and snaps[0].endswith(".safetensors"):
        kf = safe_open(snaps[0], framework="numpy")
        mats = sorted(((np.prod(kf.get_slice(k).get_shape()), k) for k in kf.keys()
                       if len(kf.get_slice(k).get_shape()) == 2), reverse=True)[:2]
        for _, k in mats:
            probe(f"Kokoro {k}", kf.get_tensor(k))
    elif snaps:
        import torch
        sd = torch.load(snaps[0], map_location="cpu", weights_only=True)
        flat = {}
        def walk(d, pre=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    walk(v, pre + k + ".")
                elif hasattr(v, "ndim") and v.ndim == 2:
                    flat[pre + k] = v
        walk(sd)
        mats = sorted(((v.numel(), k) for k, v in flat.items()), reverse=True)[:2]
        for _, k in mats:
            probe(f"Kokoro {k}", flat[k].numpy())
    print("\nRUNG 1 COMPLETE", flush=True)
