"""LLM PRESENCE SUITE (pre-registered here, before data): the EXACT protocol
of the original 18/18 presence verdict -- 5 seeded shuffle-nulls of the same
tensor + a moment-matched Gaussian yardstick at the 3 registered fractions,
with the theorem-forced bit-reversal self-test per tensor -- applied to the
canonical LLM: GPT-2 full precision, the knowledge-storage class named by
the expansion-direction law (token embedding + all 12 MLP expansion
matrices = 13 tensors, 39 registered checks). Verdict rule (fixed): a check
passes iff real concentration exceeds BOTH nulls; the tensor passes iff all
3 fractions pass; report N/39 and M/13. Seed 20260706, as always."""
import numpy as np, sys, time, os
sys.path.insert(0, ".")
from spectral_probe import fwht, concentration, FRACS, bit_reverse_perm
from safetensors import safe_open

SEED = 20260706
MODEL_FILE = "gpt2_model.safetensors"
if not os.path.exists(MODEL_FILE):
    # ONE-COMMAND REPLICATION: fetch the public GPT-2 weights (the exact
    # tensors the registered protocol reads) from the official repository,
    # so a fresh clone reproduces the 13/13, 39/39 verdict with this one
    # command and nothing else.
    url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    print(f"{MODEL_FILE} not found -- fetching GPT-2 weights (~548 MB) from\n  {url}", flush=True)
    import urllib.request
    def _hook(b, bs, t):
        if t > 0 and b % 256 == 0:
            print(f"  {b * bs / 1e6:.0f}/{t / 1e6:.0f} MB", end="\r", flush=True)
    urllib.request.urlretrieve(url, MODEL_FILE + ".part", _hook)
    os.replace(MODEL_FILE + ".part", MODEL_FILE)
    print(f"\n  fetched: {os.path.getsize(MODEL_FILE) / 1e6:.0f} MB", flush=True)
f = safe_open(MODEL_FILE, framework="numpy")
names = [("wte (token embedding)", "wte.weight")] + \
        [(f"L{L} mlp.c_fc (expansion)", f"h.{L}.mlp.c_fc.weight") for L in range(12)]
checks = tensors = 0
total_checks = 0
for label, key in names:
    w = f.get_tensor(key).astype(np.float64).ravel()
    n = 1 << int(np.floor(np.log2(len(w))))
    w = w[:n].copy()
    nbits = int(np.log2(n))
    real = concentration(fwht(w.copy()), FRACS)
    st = concentration(fwht(w[bit_reverse_perm(nbits)].copy()), FRACS)
    assert all(abs(st[q] - real[q]) < 1e-9 for q in FRACS), "self-test FAIL -- run void"
    rng = np.random.default_rng(SEED)
    null_max = {q: 0.0 for q in FRACS}
    for s in range(5):
        c = concentration(fwht(w[rng.permutation(n)].copy()), FRACS)
        for q in FRACS:
            null_max[q] = max(null_max[q], c[q])
    g = concentration(fwht(rng.normal(w.mean(), w.std(), n)), FRACS)
    passed = [real[q] > null_max[q] and real[q] > g[q] for q in FRACS]
    checks += sum(passed)
    total_checks += len(FRACS)
    tensors += all(passed)
    print(f"{label:32s} {'PASS' if all(passed) else 'FAIL'} "
          f"({sum(passed)}/3 fractions; margin {real[FRACS[0]]/max(null_max[FRACS[0]],1e-12):.1f}x)", flush=True)
print(f"\nLLM PRESENCE VERDICT: {tensors}/13 tensors, {checks}/{total_checks} registered checks", flush=True)
