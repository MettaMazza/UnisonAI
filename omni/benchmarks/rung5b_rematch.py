"""RUNG 5b REMATCH: today's engine, today's arena, twin retrained.

THE QUESTION. Rung 5b (rung5b_words.py, committed 2026-07) measured the
NEWBORN counted engine against a trained transformer twin at word scale
and the twin led: fold CE 4.5071 vs twin mean 3.4967 (seeds 1,2,3).
That measurement predates today's architecture: context depth 6
(= GEN_B*GEN_C -- was 5) and the prose flood (the prebuilt Gutenberg
store merged at wake). The rematch asks: at today's depth and today's
volume, where does the gap stand?

TWO SUBSTRATE FACTS, handled honestly (both discovered by this run's
own self-checks, both recorded):
  1. THE CORPUS IS A LIVING DOCUMENT. The June arena held 2,514,818
     tokens (vocab 16,125); today's glob yields more. A committed twin
     CE from June text is NOT the same measure as today's -- so the
     twin is RETRAINED here, same architecture, same seeds (1,2,3),
     same protocol, on today's text. The June records are quoted
     beside today's for the trail; the VERDICT compares same-text
     numbers only.
  2. THE PROSE STORE GROWS LIVE (the flight's store-rebuild loop).
     Both this script and rung5d_transfer.py read the FROZEN snapshot
     store_rung5_snapshot.pkl (one cp of store.pkl, book count
     recorded in the output) so the two scripts share one substrate
     and their cross-script identity test can hold.

THE FOLD SIDE (today's engine mechanism, verified against the live
source this session): orbit store at depths 0..6 via the engine's own
key law (case-folded context tuples, unison_chat._key with
STORE_BOUND=0), original-case successors, exact counts; the arena
TRAIN SPLIT written in; the PROSE SNAPSHOT merged by the engine's own
merge law (counts ADD at depths 0..3, build_store.py PCTX=3); the
No-Zero floor exactly as rung 5b: p = (1/V)/(total+1) + n/(total+1).

CONTAMINATION RULE (registered): the prose store is built from
fold_ai/diet/*.txt ONLY (Project Gutenberg -- verified in
build_store.py:40), DISJOINT from the arena's val split. Corpus
wake-reading and teacher lessons are EXCLUDED from the fold side: both
can quote val-split text. Only the train split and the flood enter.

DECOMPOSITION (registered): the fold side bundles TWO deltas (depth
5->6 and the flood), so the same eval also runs with the flood
withheld -- each delta measured, not guessed.

VERDICT RULE (fixed before running): report fold CE beside today's
twin mean. Gap closed or flipped -> the word-scale loss falls. Not
closed -> the residual is rung 5d's target. Committed either way.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, re, time, pickle
from collections import defaultdict, Counter

CTX_ENGINE, PCTX = 6, 3
CTX_T, DIM, HEADS, LAYERS = 64, 128, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-4
SEEDS = (1, 2, 3)
BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"
JUNE_FOLD, JUNE_TWIN = 4.5071, 3.4967   # the committed June-arena records

# ---- THE ARENA (identical construction to rung5b_words.py) ----
files = sorted(glob.glob(BASE + "/**/*.md", recursive=True)) + \
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
train_words, val_words = toks[:n_split], toks[n_split:]
print(f"tokens {len(ids)}, vocab {V} (rare-mapped; June arena was 2514818/16125)", flush=True)

# ---- FOLD ENGINE, today's law ----
def _key(tup): return tuple(x.lower() for x in tup)
t0 = time.time()
stores = [defaultdict(lambda: defaultdict(int)) for _ in range(CTX_ENGINE + 1)]
for i in range(len(train_words) - 1):
    nxt = train_words[i + 1]
    for L in range(0, CTX_ENGINE + 1):
        if i - L + 1 < 0:
            break
        stores[L][_key(tuple(train_words[i - L + 1:i + 1]))][nxt] += 1
print(f"train-split orbits: {sum(len(s) for s in stores)} in {time.time()-t0:.0f}s (one reading)", flush=True)

t0 = time.time()
def _ddint(): return defaultdict(int)
class _StoreUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _ddint if name == "_ddint" else super().find_class(module, name)
with open(BASE + "/fold_ai/store_rung5_snapshot.pkl", "rb") as f:
    _st = _StoreUnpickler(f).load()
prose = _st["stores"]
assert _st.get("bound", 0) == 0, "store built bounded -- key law differs, refusing"
assert all("diet/" in x and x.endswith(".txt") for x in _st["ingested"]), \
    "prose store ingested non-diet files -- contamination, refusing"
print(f"prose SNAPSHOT loaded: {sum(len(s) for s in prose)} orbits from "
      f"{len(_st['ingested'])} Gutenberg books in {time.time()-t0:.0f}s", flush=True)

def predict(ctx_words, flood=True):
    for L in range(min(CTX_ENGINE, len(ctx_words)), -1, -1):
        k = _key(tuple(ctx_words[-L:])) if L else ()
        s1 = stores[L].get(k)
        s2 = prose[L].get(k) if (flood and L <= PCTX) else None
        if s1 or s2:
            p = np.zeros(V)
            total = 0
            for s in (s1, s2):
                if s:
                    for w, n in s.items():
                        p[stoi.get(w, RARE)] += n
                        total += n
            p = (p + 1.0 / V) / (total + 1.0)
            return p
    return np.full(V, 1.0 / V)

# ---- EVAL (protocol of rung 5b: rng 999, 20 x 32 x 64, draw bound 5) ----
rng = np.random.default_rng(999)
SEQS = []
for _ in range(20):
    ix = rng.integers(5, len(val_d) - 65, 32)
    for i in ix:
        SEQS.append((val_d[i:i + 65], val_words[i:i + 65]))

def eval_ce(flood):
    losses = []
    for seq_ids, seq_w in SEQS:
        for j in range(64):
            p = predict(seq_w[max(0, j - CTX_ENGINE + 1):j + 1], flood=flood)
            losses.append(-np.log(max(p[int(seq_ids[j + 1])], 1e-12)))
    return float(np.mean(losses))

t1 = time.time()
fold_loss = eval_ce(flood=True)
print(f"FOLD ENGINE (today) held-out CE: {fold_loss:.4f}  (eval {time.time()-t1:.0f}s)", flush=True)
t1 = time.time()
noflood_loss = eval_ce(flood=False)
print(f"DECOMPOSITION depth-6, train only (flood withheld): {noflood_loss:.4f}", flush=True)

# ---- THE TWIN, retrained on today's text (same arch/seeds/protocol) ----
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
def batch(d, rng_):
    ix = torch.from_numpy(rng_.integers(0, len(d) - CTX_T - 1, BATCH))
    return (torch.stack([d[i:i + CTX_T] for i in ix]),
            torch.stack([d[i + 1:i + CTX_T + 1] for i in ix]))

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
    print(f"TWIN(today) seed={seed} val={vl:.4f} ({time.time()-t2:.0f}s)", flush=True)
twin_today = float(np.mean(tw))

print(f"\nJUNE RECORD  newborn engine: {JUNE_FOLD:.4f}   twin mean: {JUNE_TWIN:.4f}   (June text)", flush=True)
print(f"TODAY  twin mean (seeds 1,2,3):     {twin_today:.4f}", flush=True)
print(f"TODAY  engine, depth 6 + flood:     {fold_loss:.4f}   <- the registered rematch number", flush=True)
print(f"TODAY  engine, depth 6, no flood:   {noflood_loss:.4f}   (decomposition)", flush=True)
print("VERDICT:", "FOLD WINS/TIES -- the word-scale loss falls" if fold_loss <= twin_today + 0.005
      else f"twin still leads by {fold_loss - twin_today:.4f} nats on today's text -- residual is rung 5d's target",
      flush=True)
print("RUNG 5b REMATCH COMPLETE", flush=True)
