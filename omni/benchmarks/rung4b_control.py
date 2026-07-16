"""RUNG 4b: THE ATTENTION GATE. Dyadic-cascade attention (forced by the
unit-capacity theorem; zero parameters, zero choices) vs trained softmax
attention (the recorded Rung-4 baseline). Registered in PROTOCOL.md."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, time
FROZEN_SOFTMAX = True

CTX, DIM, HEADS, LAYERS = 128, 128, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-4
SEEDS = (1, 2, 3)
HD = DIM // HEADS

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

def cascade_weights(scores, mask):
    """Dyadic cascade over the ranking of masked scores. scores: (B,H,T,T).
    weight of rank r (0-based) = 2^-(r+1); final valid rank takes the
    closing remainder 2^-n_valid extra -- total exactly 1."""
    s = scores.masked_fill(mask, float("-inf"))
    order = torch.argsort(s, dim=-1, descending=True)
    ranks = torch.argsort(order, dim=-1)
    n_valid = (~mask).sum(-1, keepdim=True)          # (1,1,T,1)
    w = torch.pow(2.0, -(ranks.float() + 1.0))
    is_last = ranks == (n_valid - 1)
    w = w + is_last.float() * torch.pow(2.0, -n_valid.float())
    return w.masked_fill(mask, 0.0)

class CascadeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qk = nn.Linear(DIM, 2 * DIM, bias=False)
        for p in self.qk.parameters():
            p.requires_grad = False               # random comparison directions, recorded
        self.v = nn.Linear(DIM, DIM, bias=False)
        self.out = nn.Linear(DIM, DIM, bias=False)
    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k = self.qk(x).split(DIM, dim=-1)
        q = q.view(B, T, HEADS, HD).transpose(1, 2)
        k = k.view(B, T, HEADS, HD).transpose(1, 2)
        v = self.v(x).view(B, T, HEADS, HD).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / (HD ** 0.5)
        if FROZEN_SOFTMAX:
            w = torch.softmax(scores.masked_fill(mask.view(1, 1, T, T), float("-inf")), dim=-1)
        else:
            w = cascade_weights(scores, mask.view(1, 1, T, T))
        y = (w @ v).transpose(1, 2).reshape(B, T, DIM)
        return self.out(y)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.attn = CascadeAttention()
        self.ff = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Linear(4 * DIM, DIM))
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff(self.ln2(x))

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(V, DIM)
        self.wpe = nn.Embedding(CTX, DIM)
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

def batch(d, rng):
    ix = torch.from_numpy(rng.integers(0, len(d) - CTX - 1, BATCH))
    x = torch.stack([d[i:i + CTX] for i in ix])
    y = torch.stack([d[i + 1:i + CTX + 1] for i in ix])
    return x, y

def run(seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Tiny()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
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
    print(f"{'FROZEN-SOFTMAX' if FROZEN_SOFTMAX else 'CASCADE'} seed={seed}  val={vl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return vl

if __name__ == "__main__":
    C = [run(s) for s in SEEDS]
    print(f"\ntrained-softmax 1.8878 | cascade 2.6010 | frozen-softmax {np.mean(C):.4f}", flush=True)
    print("ATTRIBUTION: frozen-directions cost =", f"{np.mean(C)-1.8878:+.4f}",
          "| cascade-law cost =", f"{2.6010-np.mean(C):+.4f}", flush=True)
    print("RUNG 4b-CONTROL COMPLETE", flush=True)
