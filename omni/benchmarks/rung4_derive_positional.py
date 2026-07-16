"""RUNG 4: the first derivation gate. Learned positional embeddings vs the
DERIVED Walsh positional code (zero parameters), identical tiny transformers,
identical data (the SFTOM corpus's own text), identical budgets, 3 seeds.
Registered in PROTOCOL.md before any run."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, os, time

DEV = "cpu"
CTX, DIM, HEADS, LAYERS = 128, 128, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-4
SEEDS = (1, 2, 3)

# --- data: the corpus's own text ---
paths = sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/papers/*.md")) + \
        sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/*.md"))
text = "".join(open(p, errors="ignore").read() for p in paths)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_split = int(0.9 * len(data))
train_d, val_d = data[:n_split], data[n_split:]
V = len(chars)
print(f"corpus: {len(text)} chars, vocab {V}", flush=True)

def walsh_code(ctx, dim):
    """The derived positional code: Walsh functions of the position index.
    code[p, j] = (-1)^popcount(p & j) scaled by 1/sqrt(dim). Zero parameters:
    every entry forced by the dyadic characters."""
    P = np.arange(ctx)[:, None]
    J = np.arange(dim)[None, :]
    bits = np.zeros((ctx, dim), dtype=np.int64)
    x = P & J
    while x.any():
        bits += x & 1
        x >>= 1
    return torch.tensor(((-1.0) ** bits) / np.sqrt(dim), dtype=torch.float32)

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
    def __init__(self, derived_pos):
        super().__init__()
        self.wte = nn.Embedding(V, DIM)
        self.derived = derived_pos
        if derived_pos:
            self.register_buffer("pos", walsh_code(CTX, DIM))
        else:
            self.wpe = nn.Embedding(CTX, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.lnf = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, V, bias=False)
    def forward(self, idx):
        T = idx.shape[1]
        x = self.wte(idx)
        x = x + (self.pos[:T] if self.derived else self.wpe(torch.arange(T)))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.lnf(x))

def batch(d, rng):
    ix = torch.from_numpy(rng.integers(0, len(d) - CTX - 1, BATCH))
    x = torch.stack([d[i:i + CTX] for i in ix])
    y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
    return x, y

def run(derived, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Tiny(derived)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    t0 = time.time()
    for s in range(STEPS):
        x, y = batch(train_d, rng)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    vrng = np.random.default_rng(999)
    with torch.no_grad():
        vl = np.mean([F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1)).item()
                      for x, y in (batch(val_d, vrng) for _ in range(20))])
    n_pos = 0 if derived else CTX * DIM
    print(f"{'DERIVED-walsh' if derived else 'LEARNED-wpe '} seed={seed}  val={vl:.4f}  "
          f"pos-params={n_pos}  ({time.time()-t0:.0f}s)", flush=True)
    return vl

if __name__ == "__main__":
    L = [run(False, s) for s in SEEDS]
    D = [run(True, s) for s in SEEDS]
    print(f"\nLEARNED mean val loss: {np.mean(L):.4f}", flush=True)
    print(f"DERIVED mean val loss: {np.mean(D):.4f}", flush=True)
    print("VERDICT:", "DERIVED WINS (or ties) -- the rung is taken" if np.mean(D) <= np.mean(L) + 0.005
          else "learned wins -- recorded", flush=True)
    print("RUNG 4 COMPLETE", flush=True)
