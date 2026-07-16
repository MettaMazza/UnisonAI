"""RUNG 5b: the orbit engine at word scale vs the trained twin.
Registered in PROTOCOL.md."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, re, time
from collections import defaultdict

CTX_MAX = 5
STEPS, BATCH, CTX_T, DIM, HEADS, LAYERS, LR = 1500, 32, 64, 128, 4, 4, 3e-4
SEEDS = (1, 2, 3)

files = sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/**/*.md", recursive=True)) + \
        sorted(glob.glob("/Users/mettamazza/Desktop/SFTOM/**/*.md", recursive=True))
files = [f for f in files if "/language/" not in f and "/.git/" not in f]
text = "\n".join(open(f, errors="ignore").read() for f in files)
toks = re.findall(r"\w+|[^\w\s]", text)
from collections import Counter
cnt = Counter(toks)
vocab_words = [w for w, c in cnt.items() if c >= 3]
stoi = {w: i for i, w in enumerate(vocab_words)}
RARE = len(stoi)
V = RARE + 1
ids = np.array([stoi.get(t, RARE) for t in toks], dtype=np.int64)
n_split = int(0.9 * len(ids))
train_d, val_d = ids[:n_split], ids[n_split:]
print(f"tokens {len(ids)}, vocab {V} (rare-mapped)", flush=True)

# ---- FOLD ENGINE ----
t0 = time.time()
stores = [defaultdict(lambda: defaultdict(int)) for _ in range(CTX_MAX + 1)]
tup = tuple(train_d.tolist())
for i in range(len(tup) - 1):
    nxt = tup[i + 1]
    for L in range(0, CTX_MAX + 1):
        if i - L + 1 < 0: break
        stores[L][tup[i - L + 1:i + 1]][nxt] += 1
print(f"orbit store: {sum(len(s) for s in stores)} orbits in {time.time()-t0:.0f}s (one reading)", flush=True)

def predict(ctx):
    for L in range(min(CTX_MAX, len(ctx)), -1, -1):
        s = stores[L].get(tuple(ctx[-L:]) if L else ())
        if s:
            total = sum(s.values())
            p = np.full(V, (1.0 / V) / (total + 1.0))
            for c, n in s.items(): p[c] += n / (total + 1.0)
            return p
    return np.full(V, 1.0 / V)

t1 = time.time()
rng = np.random.default_rng(999)
losses = []
for _ in range(20):
    ix = rng.integers(CTX_MAX, len(val_d) - 65, 32)
    for i in ix:
        seq = val_d[i:i + 65]
        for j in range(64):
            p = predict(seq[max(0, j - CTX_MAX + 1):j + 1].tolist())
            losses.append(-np.log(max(p[int(seq[j + 1])], 1e-12)))
fold_loss = float(np.mean(losses))
print(f"FOLD ENGINE held-out CE: {fold_loss:.4f}  (eval {time.time()-t1:.0f}s)", flush=True)

# ---- TRANSFORMER TWIN ----
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
    def forward(self, x, mask):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, DIM)
        self.wpe = nn.Embedding(CTX_T, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, V, bias=False)
    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx) + self.wpe(torch.arange(T))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))

td, vd = torch.from_numpy(train_d), torch.from_numpy(val_d)
def batch(d, rng):
    ix = torch.from_numpy(rng.integers(0, len(d) - CTX_T - 1, BATCH))
    x = torch.stack([d[i:i + CTX_T] for i in ix])
    y = torch.stack([d[i + 1:i + CTX_T + 1] for i in ix])
    return x, y

tw = []
for seed in SEEDS:
    torch.manual_seed(seed)
    rng2 = np.random.default_rng(seed)
    m = Tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    t2 = time.time()
    for s in range(STEPS):
        x, y = batch(td, rng2)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    vr = np.random.default_rng(999)
    with torch.no_grad():
        vl = np.mean([F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
                      for x, y in (batch(vd, vr) for _ in range(20))])
    tw.append(vl)
    print(f"TWIN seed={seed} val={vl:.4f} ({time.time()-t2:.0f}s)", flush=True)

print(f"\nFOLD ENGINE (one reading, zero training): {fold_loss:.4f}", flush=True)
print(f"TRANSFORMER TWIN mean:                    {np.mean(tw):.4f}", flush=True)
print("VERDICT:", "FOLD WINS/TIES -- rung taken" if fold_loss <= np.mean(tw) + 0.005
      else "twin wins -- recorded", flush=True)

# generation sample
itos = {i: w for w, i in stoi.items()}
itos[RARE] = "<rare>"
grng = np.random.default_rng(7)
ctx = [stoi.get(w, RARE) for w in re.findall(r"\w+|[^\w\s]", "The fold is")]
out = ["The fold is"]
for _ in range(120):
    p = predict(ctx)
    c = int(grng.choice(V, p=p / p.sum()))
    out.append(itos[c]); ctx.append(c)
print("\nFIRST WORDS (word-level):\n" + " ".join(out), flush=True)
print("RUNG 5b COMPLETE", flush=True)
