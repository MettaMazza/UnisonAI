"""RUNG 5c: THE LOUD-SUBSPACE TRANSFER TEST (pre-registered here, before data).

THE CLAIM UNDER TEST (the paper's two strongest results, joined): Rung 1/2
located the trained law in the dyadically-LOUD tensors (embeddings + FFN
expansions; attention at chance). Rung 5b measured the trained twin ahead of
the counted engine at word scale (3.50 vs 4.51). If loud = lawful = the
purchased advantage, then the twin's advantage must LIVE in its dyadically-
loud coefficient subspace and nowhere else.

PROTOCOL (fixed in advance):
- Arena: identical to rung5b (same corpus, vocab, split, twin architecture,
  seed 1, eval seed 999).
- Transform: per-row 1D Walsh-Hadamard over DIM=128 (rows are power of two;
  no padding, no windowing).
- Variants at matched coefficient budgets k in {16, 32, 64} of 128 per row:
    LOUD    -- keep each row's top-k |coefficient|, zero the rest, invert.
    RANDOM  -- keep k coefficients chosen uniformly per row (seeded), invert.
- Tensor classes: LOUD CLASS = wte + every ffn expansion (ff[0]) + head;
  QUIET CLASS control = attention in_proj weights only.
- SELF-TESTS (theorem-forced, must pass exactly or the run is void):
  (a) WHT involution: H(H(x))/N == x to machine precision;
  (b) truncation at k=128 must reproduce baseline CE exactly.
- VERDICT RULE: the claim is SUPPORTED if, on the loud class, at every
  budget k: CE(LOUD_k) - CE(full) < CE(RANDOM_k) - CE(full); and the same
  margin on the quiet class is smaller (attention carries less of the law).
  Readout: the budget at which LOUD_k still beats the counted engine's
  4.5071 -- the size of the purchase, in coefficients.
Result file: rung5c_results.txt. Negative result recorded in full if it lands.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, re, time
from collections import defaultdict, Counter

CTX_T, DIM, HEADS, LAYERS = 64, 128, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-4
SEED = 1
FOLD_CE = 4.5071   # rung5b's recorded counted-engine reference

files = sorted(glob.glob("/Users/mettamazza/Desktop/Smithian Fold Theory/**/*.md", recursive=True)) + \
        sorted(glob.glob("/Users/mettamazza/Desktop/SFTOM/**/*.md", recursive=True))
files = [f for f in files if "/language/" not in f and "/.git/" not in f]
text = "\n".join(open(f, errors="ignore").read() for f in files)
toks = re.findall(r"\w+|[^\w\s]", text)
cnt = Counter(toks)
stoi = {w: i for i, w in enumerate([w for w, c in cnt.items() if c >= 3])}
RARE = len(stoi); V = RARE + 1
ids = np.array([stoi.get(t, RARE) for t in toks], dtype=np.int64)
n_split = int(0.9 * len(ids))
train_d, val_d = ids[:n_split], ids[n_split:]
print(f"tokens {len(ids)}, vocab {V}", flush=True)

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
    return (torch.stack([d[i:i + CTX_T] for i in ix]),
            torch.stack([d[i + 1:i + CTX_T + 1] for i in ix]))

torch.manual_seed(SEED)
rng2 = np.random.default_rng(SEED)
m = Tiny()
opt = torch.optim.AdamW(m.parameters(), lr=LR)
t0 = time.time()
for s in range(STEPS):
    x, y = batch(td, rng2)
    loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"twin trained in {time.time()-t0:.0f}s", flush=True)
m.eval()

def val_ce(model):
    vr = np.random.default_rng(999)
    with torch.no_grad():
        return float(np.mean([F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item()
                              for x, y in (batch(vd, vr) for _ in range(20))]))

# ---- per-row Walsh-Hadamard over DIM ----
H = np.array([[1.0]])
while H.shape[0] < DIM:
    H = np.block([[H, H], [H, -H]])
Ht = torch.from_numpy(H).float()
def wht_rows(W):    # rows of length DIM
    return W @ Ht
def iwht_rows(C):
    return (C @ Ht) / DIM

# SELF-TEST (a): involution to machine precision
_x = torch.randn(7, DIM)
assert torch.allclose(iwht_rows(wht_rows(_x)), _x, atol=1e-4), "WHT involution FAILED -- run void"
print("self-test (a) involution: PASS", flush=True)

LOUD_NAMES = ["wte.weight", "head.weight"] + [f"blocks.{i}.ff.0.weight" for i in range(LAYERS)]
QUIET_NAMES = [f"blocks.{i}.attn.in_proj_weight" for i in range(LAYERS)]

base_state = {k: v.clone() for k, v in m.state_dict().items()}
base_ce = val_ce(m)
print(f"baseline twin CE: {base_ce:.4f}   (counted engine reference {FOLD_CE})", flush=True)

def truncate(W, k, mode, seed=0):
    C = wht_rows(W)
    if mode == "loud":
        thr = torch.kthvalue(C.abs(), C.shape[1] - k + 1, dim=1, keepdim=True).values
        mask = C.abs() >= thr
    else:
        g = torch.Generator().manual_seed(seed)
        idx = torch.rand(C.shape, generator=g).argsort(dim=1) < k
        mask = idx
    return iwht_rows(C * mask)

def run_variant(names, k, mode):
    sd = {kk: vv.clone() for kk, vv in base_state.items()}
    for n in names:
        sd[n] = truncate(base_state[n], k, mode, seed=k)
    m.load_state_dict(sd)
    return val_ce(m)

# SELF-TEST (b): k = DIM must reproduce baseline exactly
ce_full = run_variant(LOUD_NAMES, DIM, "loud")
assert abs(ce_full - base_ce) < 5e-3, f"self-test (b) FAILED: {ce_full} vs {base_ce}"
print(f"self-test (b) k=128 reproduces baseline: PASS ({ce_full:.4f})", flush=True)

lines = [f"RUNG 5c RESULTS -- baseline {base_ce:.4f}, counted engine {FOLD_CE}"]
support = True
for k in (16, 32, 64):
    ce_l = run_variant(LOUD_NAMES, k, "loud")
    ce_r = run_variant(LOUD_NAMES, k, "random")
    ce_lq = run_variant(QUIET_NAMES, k, "loud")
    ce_rq = run_variant(QUIET_NAMES, k, "random")
    loud_margin = ce_r - ce_l
    quiet_margin = ce_rq - ce_lq
    ok = (ce_l < ce_r) and (quiet_margin < loud_margin)   # BOTH registered conjuncts
    support = support and ok
    beats = "BEATS counted engine" if ce_l < FOLD_CE else "below counted engine"
    ln = (f"k={k:3d}/128  LOUD-class: loud {ce_l:.4f} vs random {ce_r:.4f} (margin {loud_margin:+.4f}) "
          f"| QUIET-class: loud {ce_lq:.4f} vs random {ce_rq:.4f} (margin {quiet_margin:+.4f}) "
          f"| loud_k {beats} | {'OK' if ok else 'X'}")
    print(ln, flush=True)
    lines.append(ln)
verdict = "SUPPORTED" if support else "REFUSED"
lines.append(f"VERDICT (pre-registered rule): {verdict} -- the twin's advantage "
             f"{'lives in' if support else 'is NOT confined to'} the dyadically-loud subspace")
print(lines[-1], flush=True)
open("/Users/mettamazza/Desktop/Smithian Fold Theory/fold_ai/rung5c_results.txt", "w").write("\n".join(lines) + "\n")
print("RUNG 5c COMPLETE", flush=True)
