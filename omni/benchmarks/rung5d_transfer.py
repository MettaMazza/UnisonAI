"""RUNG 5d: THE TRANSFER-IN (pre-registered here, before data).

THE CLAIM UNDER TEST (the constructive converse of rung 5c): 5c proved
the trained twin's advantage LIVES in its dyadically-loud coefficient
subspace -- truncating to the top 16/128 fold coefficients kept the law
(margins +2.18/+3.71/+3.77 nats over random at k=16/32/64). If that is
true content and not an artifact of where we looked, it must be
TAKEABLE: wired into the counted engine as a counted prior, it must
close the word-scale gap -- and a random-truncated twin must not.

THE MECHANISM (zero new parameters). The engine's No-Zero floor puts
total mass 1/(total+1) on unseen continuations -- the forced unseen
share, fixed by the counted law, not adjustable. Today it is spread
UNIFORMLY (1/V each). 5d reshapes that SAME mass by the truncated
twin's next-token distribution q(w|ctx):

    p(w|ctx) = n(w)/(total+1) + q(w|ctx)/(total+1)

The counted part is untouched. The floor's total mass is untouched.
There is no lambda, no blend knob, no temperature: the only thing that
changes is the SHAPE of mass the engine already owed to the unseen,
and that shape is measured fold content, not a fitted quantity.
q = uniform reproduces today's engine exactly (self-test c).

ARENA: identical to rung 5b/5c (same corpus glob, vocab>=3, 90/10
split, twin seed 1, eval rng 999, 20 x 32 x 64 positions -- the same
positions as rung5b_rematch.py, drawn with the arena's registered
bound). Engine side = today's engine exactly as the rematch holds it:
depth-6 orbit store over the train split + the Gutenberg prose flood,
the engine's own case-folded key law. SUBSTRATE PIN: the flood is read
from the FROZEN snapshot store_rung5_snapshot.pkl -- the live
store.pkl grows continuously under the flight's rebuild loop, and the
cross-script identity test (self-test c) can only hold on a shared
frozen substrate. The twin-comparison readout quotes the rematch's
same-day retrained twin mean (parsed from its results file), never the
June record: the corpus is a living document and only same-text
numbers may face each other.

ARMS (matched budget, all through the identical pipeline):
    UNIFORM        q = 1/V            -- baseline, today's engine
    LOUD-k         q = twin truncated to top-k |WHT| per row, k in {16,32,64}
    RANDOM-k       q = twin truncated to k random coefficients (seed=k), the null
    FULL           q = untruncated twin -- upper reference

SELF-TESTS (theorem-forced; any failure voids the run):
  (a) WHT involution to machine precision (as 5c);
  (b) the k=128 loud arm must equal the FULL arm (truncation at full
      budget is the identity);
  (c) the UNIFORM arm must reproduce rung5b_rematch's fold CE exactly
      (same positions, same formula -- cross-script identity).

VERDICT RULE (fixed in advance): SUPPORTED iff at EVERY budget k,
CE(loud-k) < CE(random-k). Readout beside the verdict: % of the
uniform-to-full gap closed at each k, and whether any loud arm carries
today's engine past the twin's own 3.4967. Negative result recorded in
full if it lands. Result file: rung5d_results.txt.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, re, time, pickle, os
from collections import defaultdict, Counter

CTX_T, DIM, HEADS, LAYERS = 64, 128, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-4
SEED = 1
CTX_ENGINE, PCTX = 6, 3
BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"

# ---- THE ARENA (identical to 5b/5c) ----
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
print(f"tokens {len(ids)}, vocab {V}", flush=True)

# ---- THE TWIN (identical to 5c, seed 1) ----
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

td = torch.from_numpy(train_d)
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

# ---- WHT machinery (identical to 5c) ----
H = np.array([[1.0]])
while H.shape[0] < DIM:
    H = np.block([[H, H], [H, -H]])
Ht = torch.from_numpy(H).float()
def wht_rows(W): return W @ Ht
def iwht_rows(C): return (C @ Ht) / DIM
_x = torch.randn(7, DIM)
assert torch.allclose(iwht_rows(wht_rows(_x)), _x, atol=1e-4), "self-test (a) FAILED -- run void"
print("self-test (a) involution: PASS", flush=True)

LOUD_NAMES = ["wte.weight", "head.weight"] + [f"blocks.{i}.ff.0.weight" for i in range(LAYERS)]
base_state = {k: v.clone() for k, v in m.state_dict().items()}

def truncate(W, k, mode, seed=0):
    C = wht_rows(W)
    if mode == "loud":
        thr = torch.kthvalue(C.abs(), C.shape[1] - k + 1, dim=1, keepdim=True).values
        mask = C.abs() >= thr
    else:
        g = torch.Generator().manual_seed(seed)
        mask = torch.rand(C.shape, generator=g).argsort(dim=1) < k
    return iwht_rows(C * mask)

def load_variant(k, mode):
    sd = {kk: vv.clone() for kk, vv in base_state.items()}
    if mode != "full":
        for n in LOUD_NAMES:
            sd[n] = truncate(base_state[n], k, mode, seed=k)
    m.load_state_dict(sd)

# ---- TODAY'S ENGINE (identical to rung5b_rematch.py) ----
def _key(tup): return tuple(x.lower() for x in tup)
t0 = time.time()
stores = [defaultdict(lambda: defaultdict(int)) for _ in range(CTX_ENGINE + 1)]
for i in range(len(train_words) - 1):
    nxt = train_words[i + 1]
    for L in range(0, CTX_ENGINE + 1):
        if i - L + 1 < 0:
            break
        stores[L][_key(tuple(train_words[i - L + 1:i + 1]))][nxt] += 1
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
print(f"engine ready: {sum(len(s) for s in stores)} train orbits + prose flood "
      f"({len(_st['ingested'])} books) in {time.time()-t0:.0f}s", flush=True)

# ---- EVAL POSITIONS (identical draw to 5b/rematch: rng 999) ----
rng = np.random.default_rng(999)
SEQS = []                        # (seq_ids[65], seq_words[65])
for _ in range(20):
    ix = rng.integers(5, len(val_d) - 65, 32)
    for i in ix:
        SEQS.append((val_d[i:i + 65], val_words[i:i + 65]))
print(f"eval positions: {len(SEQS)} sequences x 64 predictions", flush=True)

# ---- counted part, precomputed ONCE (arm-independent): for every
#      position, the serving level's target count and total ----
t0 = time.time()
COUNTED = []   # per sequence: list of (n_target, total) for j = 0..63
for seq_ids, seq_w in SEQS:
    row = []
    for j in range(64):
        ctx = seq_w[max(0, j - CTX_ENGINE + 1):j + 1]
        n_t, total = 0, 0
        for L in range(min(CTX_ENGINE, len(ctx)), -1, -1):
            k = _key(tuple(ctx[-L:])) if L else ()
            s1 = stores[L].get(k)
            s2 = prose[L].get(k) if L <= PCTX else None
            if s1 or s2:
                tgt_w, tgt_id = seq_w[j + 1], int(seq_ids[j + 1])
                for s in (s1, s2):
                    if s:
                        total += sum(s.values())
                        # the arena's event space is rare-mapped: a RARE
                        # target is credited with the POOLED rare mass,
                        # exactly as the rematch builds p over ids
                        if tgt_id == RARE:
                            n_t += sum(n for w2, n in s.items() if w2 not in stoi)
                        else:
                            n_t += s.get(tgt_w, 0)
                break
        row.append((n_t, total))
    COUNTED.append(row)
print(f"counted channel precomputed in {time.time()-t0:.0f}s", flush=True)

def arm_ce(q_mode, k=None):
    """CE of the hybrid: counted part + floor mass 1/(total+1) shaped by q.
    q_mode 'uniform' needs no twin; else the loaded twin serves q."""
    losses = []
    with torch.no_grad():
        for (seq_ids, seq_w), row in zip(SEQS, COUNTED):
            if q_mode == "uniform":
                qs = None
            else:
                x = torch.from_numpy(seq_ids[:64]).unsqueeze(0)
                logits = m(x)[0]                        # (64, V), causal
                probs = F.softmax(logits.double(), dim=1).numpy()
            for j in range(64):
                n_t, total = row[j]
                q_t = (1.0 / V) if q_mode == "uniform" else float(probs[j][int(seq_ids[j + 1])])
                p_t = (n_t + q_t) / (total + 1.0)
                losses.append(-np.log(max(p_t, 1e-12)))
    return float(np.mean(losses))

# ---- SELF-TEST (c): uniform arm == rematch fold CE (cross-script identity)
ce_uniform = arm_ce("uniform")
print(f"UNIFORM floor (today's engine): {ce_uniform:.4f}", flush=True)
TWIN_TODAY = None
_rr = BASE + "/fold_ai/rung5b_rematch_results.txt"
if os.path.exists(_rr):
    _rrt = open(_rr).read()
    m_re = re.search(r"held-out CE: ([0-9.]+)", _rrt)
    if m_re:
        assert abs(ce_uniform - float(m_re.group(1))) < 1e-3, \
            f"self-test (c) FAILED: {ce_uniform} vs rematch {m_re.group(1)} -- run void"
        print(f"self-test (c) uniform == rematch ({m_re.group(1)}): PASS", flush=True)
    m_tw = re.search(r"twin mean \(seeds 1,2,3\):\s+([0-9.]+)", _rrt)
    if m_tw:
        TWIN_TODAY = float(m_tw.group(1))
else:
    print("self-test (c): rematch results not present yet -- identity by construction, recheck after", flush=True)

# ---- SELF-TEST (b): k=128 loud arm == FULL arm ----
load_variant(None, "full")
ce_full = arm_ce("twin")
load_variant(DIM, "loud")
ce_k128 = arm_ce("twin")
assert abs(ce_k128 - ce_full) < 1e-3, f"self-test (b) FAILED: {ce_k128} vs {ce_full} -- run void"
print(f"self-test (b) k=128 == full ({ce_full:.4f}): PASS", flush=True)

# ---- THE ARMS ----
lines = [f"RUNG 5d RESULTS -- uniform floor {ce_uniform:.4f}, full-twin-shaped floor {ce_full:.4f}"
         + (f", same-day twin mean {TWIN_TODAY}" if TWIN_TODAY else "")]
gap = ce_uniform - ce_full
support = True
for k in (16, 32, 64):
    load_variant(k, "loud")
    ce_l = arm_ce("twin")
    load_variant(k, "random")
    ce_r = arm_ce("twin")
    ok = ce_l < ce_r
    support = support and ok
    closed_l = (ce_uniform - ce_l) / gap * 100 if gap > 0 else float("nan")
    closed_r = (ce_uniform - ce_r) / gap * 100 if gap > 0 else float("nan")
    beats = " | loud-shaped engine BEATS the same-day twin" if (TWIN_TODAY and ce_l < TWIN_TODAY) else ""
    ln = (f"k={k:3d}/128  loud-shaped {ce_l:.4f} ({closed_l:.1f}% of gap) vs "
          f"random-shaped {ce_r:.4f} ({closed_r:.1f}%){beats} | {'OK' if ok else 'X'}")
    print(ln, flush=True)
    lines.append(ln)
verdict = "SUPPORTED" if support else "REFUSED"
lines.append(f"VERDICT (pre-registered rule): {verdict} -- the twin's loud fold content "
             f"{'transfers INTO the counted engine as a counted prior' if support else 'did NOT transfer under this rule'}")
print(lines[-1], flush=True)
open(BASE + "/fold_ai/rung5d_results.txt", "w").write("\n".join(lines) + "\n")
print("RUNG 5d COMPLETE", flush=True)
