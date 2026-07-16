"""RUNG 5-NATIVE: THE FOLD-NATIVE SEED vs the trained transformer, on the
task. Registered in PROTOCOL.md. The engine: knowledge = contexts stored
ONCE as exact held orbits (the tablebase pattern for text); machinery =
unit-capacity selection over the orbit hierarchy (longest held suffix
first); values = exact rational shares. Zero gradient steps, zero trained
parameters. The transformer twin consumed the same text ~11x over in
48,000 gradient readings; this engine reads it once."""
import numpy as np, glob, time
from collections import defaultdict

CTX_MAX = 12          # longest stored orbit (suffix) length
paths = sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/papers/*.md")) + \
        sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/*.md"))
text = "".join(open(p, errors="ignore").read() for p in paths)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
n_split = int(0.9 * len(data))
train_d, val_d = data[:n_split], data[n_split:]
V = len(chars)
print(f"corpus: {len(text)} chars, vocab {V}; engine reads training text ONCE", flush=True)

t0 = time.time()
# THE ORBIT STORE: for each suffix length, held orbits -> continuation counts.
stores = [defaultdict(lambda: defaultdict(int)) for _ in range(CTX_MAX + 1)]
tup = tuple(train_d.tolist())
for i in range(len(tup) - 1):
    nxt = tup[i + 1]
    for L in range(0, CTX_MAX + 1):
        if i - L + 1 < 0:
            break
        stores[L][tup[i - L + 1:i + 1]][nxt] += 1
build_s = time.time() - t0
n_orbits = sum(len(s) for s in stores)
print(f"orbit store built in {build_s:.0f}s -- {n_orbits} held orbits", flush=True)

# PREDICTION: unit-capacity selection over the orbit hierarchy -- the
# LONGEST held suffix is the single integrated focus (the lock); its exact
# continuation shares are the distribution; hierarchy fallback is the fold
# to the next-longest held orbit. Exact rational shares, smoothed only by
# the forced antipodal floor (one count of the whole alphabet -- every
# symbol remains reachable, no zero, the No-Zero axiom).
def predict(context):
    for L in range(min(CTX_MAX, len(context)), -1, -1):
        key = tuple(context[-L:]) if L > 0 else ()
        s = stores[L].get(key)
        if s:
            total = sum(s.values())
            probs = np.full(V, 1.0 / (total + V) / V * 1.0)
            base = np.full(V, 1.0 / (total + V) / V)
            # exact shares with the alphabet floor: (count + 1/V-share)
            probs = np.full(V, (1.0 / V) / (total + 1.0))
            for c, n in s.items():
                probs[c] += n / (total + 1.0)
            return probs
    return np.full(V, 1.0 / V)

t1 = time.time()
rng = np.random.default_rng(999)
CTX_EVAL, BATCH, ROUNDS = 128, 32, 20
losses = []
for _ in range(ROUNDS):
    ix = rng.integers(CTX_MAX, len(val_d) - CTX_EVAL - 1, BATCH)
    for i in ix:
        seq = val_d[i:i + CTX_EVAL + 1]
        for j in range(CTX_EVAL):
            p = predict(seq[max(0, j - CTX_MAX):j + 1][:-1].tolist() + [int(seq[j])])
            losses.append(-np.log(max(p[int(seq[j + 1])], 1e-12)))
val_loss = float(np.mean(losses))
print(f"eval in {time.time()-t1:.0f}s over {len(losses)} predictions", flush=True)
print(f"\nTRANSFORMER TWIN (trained, 48,000 gradient readings): 1.8878", flush=True)
print(f"FOLD-NATIVE SEED (text read ONCE, zero training):      {val_loss:.4f}", flush=True)
print("VERDICT:", "FOLD-NATIVE WINS/TIES -- the rung is taken" if val_loss <= 1.8878 + 0.005
      else f"transformer ahead by {val_loss-1.8878:+.4f} -- recorded with the efficiency axes", flush=True)
print(f"axes: build {build_s:.0f}s vs ~420s training x3 seeds; passes over data 1 vs ~11; fact-edit cost: write one orbit vs retrain", flush=True)
print("RUNG 5-NATIVE COMPLETE", flush=True)
