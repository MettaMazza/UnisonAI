"""RUNG 5e: THE FOLD'S OWN STACK (pre-registered here, before data).

THE STANDING RESIDUAL. After the rematch and rung 5d (verdict
SUPPORTED at every chain anchoring), the word-scale gap stands open;
this script parses the SAME-RUN chain's numbers live from the results
files (the corpus is a living document -- every chain re-anchors to
one text state, and the identity self-test voids a moved arena). The
instruction: force the LARGEST amount of the PROVEN structures at it --
all of them if possible -- everything zero-parameter, counted, forced;
verdicts recorded raw.

THE PROVEN STRUCTURES ENTERED (each already established elsewhere in
the corpus or this campaign, none invented for this rung):

1. FOLD-FACTOR LEVEL MIXING. The engine's own attention law: influence
   halves with each step of distance -- "the decay is the fold factor 2
   itself, not a tuned constant" (unison_chat.py, session attention).
   Applied to the context hierarchy: instead of hard first-hit backoff,
   EVERY level that holds the context contributes, weighted 2^level --
   deepest context heaviest, each shallower level halved. A mixture of
   proper distributions with forced weights; when only one level holds,
   it collapses to hard backoff exactly (self-test b).

2. KIN-SHAPED FLOOR. The keystone: counted kinship (exact co-occurrence
   shares) is the proven zero-parameter similarity object. The forced
   unseen mass 1/(total+1) is reshaped by the last context word's
   counted neighbour distribution -- itself protected by the same
   No-Zero form, self-similarly: q(t) = (neigh(t) + 1/V)/(S + 1), which
   sums to one exactly (self-test c). Neighbour counts come from the
   arena train split plus the Gutenberg snapshot, by the engine's own
   build law. No training, no external model, nothing chosen.

3. LOUD-TRANSFER FLOOR (rung 5d, verdict SUPPORTED). The twin's
   dyadically-loud content at k=64, as a floor shape -- the strongest
   proven extraction, stacked here on top of the fold mixing.

ARMS (all at both flood settings; same eval positions as 5b/rematch/5d,
rng 999, frozen snapshot store):
    A0 baseline      hard backoff, uniform floor  == the rematch engine
    A1 fold-mix      2^level mixing, uniform floor
    A2 kin floor     hard backoff, kin-shaped floor
    A3 mix + kin     both fold-native structures stacked
    A4 mix + loud64  fold mixing + the 5d extraction as the floor shape

SELF-TESTS (any failure voids the run):
  (a) A0 must reproduce the committed rematch numbers exactly
      (cross-script identity, parsed live from the results file);
  (b) fold-mix collapses to hard backoff when one level holds
      (constructed case, exact);
  (c) the kin floor shape sums to 1 over the vocabulary (one full
      position, numerical).

VERDICT RULE (fixed in advance): report every arm's CE beside the
same-day twin mean and 5d's k=64 reference (both parsed live from
their results files). State plainly which arms close gap and whether any
reaches the twin. No spin in either direction; the numbers are
committed as they land. Result file: rung5e_results.txt.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import glob, re, time, pickle, os
from collections import defaultdict, Counter

CTX_ENGINE, PCTX = 6, 3
CTX_T, DIM, HEADS, LAYERS = 64, 128, 4, 4
STEPS, BATCH, LR, SEED = 1500, 32, 3e-4, 1
K_LOUD = 64
BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"

# ---- THE ARENA (identical to 5b/rematch/5d) ----
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

# ---- ENGINE SUBSTRATE: train orbits + frozen snapshot (the rematch law) --
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
assert _st.get("bound", 0) == 0 and all("diet/" in x for x in _st["ingested"])
print(f"substrate: {sum(len(s) for s in stores)} train orbits + snapshot "
      f"({len(_st['ingested'])} books) in {time.time()-t0:.0f}s", flush=True)

# ---- KINSHIP SUBSTRATE (the engine's build law: train + snapshot neigh) --
t0 = time.time()
NEIGH = defaultdict(_ddint)
for i in range(1, len(train_words) - 1):
    w = train_words[i].lower()
    if len(w) >= 3:
        NEIGH[w][train_words[i - 1].lower()] += 1
        NEIGH[w][train_words[i + 1].lower()] += 1
for w, nb in _st["neigh"].items():
    for o, c in nb.items():
        NEIGH[w][o] += c
# THE CASE-FOLD BRIDGE (counted, no choice): the kin ledger is case-folded
# (the engine's key law) while the arena's events are case-sensitive vocab
# ids. Each folded neighbour's count is split across its case variants by
# their OBSERVED unigram counts -- measured shares -- and the remainder
# (variants below the vocab threshold) flows to the RARE bucket. With the
# No-Zero form on top, the shape sums to 1 exactly (self-test c).
_LOW_TOTAL = defaultdict(int)          # lower form -> total TRAIN-split count (all case variants)
_cnt_train = Counter(train_words)      # the bridge is TRAIN-ONLY (the contamination rule, enforced)
for t, c in _cnt_train.items():
    _LOW_TOTAL[t.lower()] += c
_VSHARE = {t: _cnt_train.get(t, 0) / _LOW_TOTAL[t.lower()] if _LOW_TOTAL.get(t.lower()) else 0.0
           for t in stoi}              # vocab variant's measured TRAIN share
_VCOVER = defaultdict(float)           # lower form -> total vocab-covered share
for t in stoi:
    _VCOVER[t.lower()] += _VSHARE[t]
_NSUM, _NRARE = {}, {}
def kin_shape(last, tgt_w, tgt_id):
    nb = NEIGH.get(last)
    if nb is None:
        return 1.0 / V
    if last not in _NSUM:
        _NSUM[last] = sum(nb.values())
        _NRARE[last] = sum(c * (1.0 - _VCOVER.get(w2, 0.0)) for w2, c in nb.items())
    S = _NSUM[last]
    if tgt_id == RARE:
        n = _NRARE[last]
    else:
        n = nb.get(tgt_w.lower(), 0) * _VSHARE.get(tgt_w, 0.0)
    return (n + 1.0 / V) / (S + 1.0)

# ---- EVAL POSITIONS + per-level counted channel, precomputed once ----
rng = np.random.default_rng(999)
SEQS = []
for _ in range(20):
    ix = rng.integers(5, len(val_d) - 65, 32)
    for i in ix:
        SEQS.append((val_d[i:i + 65], val_words[i:i + 65]))
t0 = time.time()
POS = []   # per sequence, per j: (levels=[(n_t, total)...deepest-first], last_word)
for seq_ids, seq_w in SEQS:
    row = []
    for j in range(64):
        ctx = seq_w[max(0, j - CTX_ENGINE + 1):j + 1]
        tgt_w, tgt_id = seq_w[j + 1], int(seq_ids[j + 1])
        levels = []
        for L in range(min(CTX_ENGINE, len(ctx)), -1, -1):
            k = _key(tuple(ctx[-L:])) if L else ()
            entry = []
            for flood in (False, True):
                s1 = stores[L].get(k)
                s2 = prose[L].get(k) if (flood and L <= PCTX) else None
                n_t, total = 0, 0
                for s in (s1, s2):
                    if s:
                        total += sum(s.values())
                        if tgt_id == RARE:
                            n_t += sum(n for w2, n in s.items() if w2 not in stoi)
                        else:
                            n_t += s.get(tgt_w, 0)
                entry.append((n_t, total))
            levels.append((L, entry))
        row.append((levels, ctx[-1].lower() if ctx else "", tgt_w, tgt_id))
    POS.append(row)
print(f"counted channel (both flood settings, all levels) in {time.time()-t0:.0f}s", flush=True)

# ---- THE TWIN for the loud floor (identical to 5c/5d, seed 1) ----
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
def batch(d, rng_):
    ix = torch.from_numpy(rng_.integers(0, len(d) - CTX_T - 1, BATCH))
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
m.eval()
print(f"twin trained in {time.time()-t0:.0f}s", flush=True)
H = np.array([[1.0]])
while H.shape[0] < DIM:
    H = np.block([[H, H], [H, -H]])
Ht = torch.from_numpy(H).float()
def truncate(W, k):
    C = W @ Ht
    thr = torch.kthvalue(C.abs(), C.shape[1] - k + 1, dim=1, keepdim=True).values
    return ((C * (C.abs() >= thr)) @ Ht) / DIM
sd = {kk: vv.clone() for kk, vv in m.state_dict().items()}
for n in ["wte.weight", "head.weight"] + [f"blocks.{i}.ff.0.weight" for i in range(LAYERS)]:
    sd[n] = truncate(sd[n], K_LOUD)
m.load_state_dict(sd)
t0 = time.time()
LOUD_QT = []   # per sequence: q_t per j from the loud-truncated twin
with torch.no_grad():
    for seq_ids, seq_w in SEQS:
        x = torch.from_numpy(seq_ids[:64]).unsqueeze(0)
        probs = F.softmax(m(x)[0].double(), dim=1).numpy()
        LOUD_QT.append([float(probs[j][int(seq_ids[j + 1])]) for j in range(64)])
print(f"loud-64 floor shapes in {time.time()-t0:.0f}s", flush=True)

# ---- THE ARMS ----
def arm_ce(flood, mix, floor):
    """floor: 'uniform' | 'kin' | 'loud'. mix: False = hard backoff."""
    fi = 1 if flood else 0
    losses = []
    for si, row in enumerate(POS):
        for j, (levels, last, tgt_w, tgt_id) in enumerate(row):
            def q_t():
                if floor == "uniform":
                    return 1.0 / V
                if floor == "kin":
                    return kin_shape(last, tgt_w, tgt_id)
                return LOUD_QT[si][j]
            if not mix:
                p_t = None
                for L, entry in levels:              # deepest first
                    n_t, total = entry[fi]
                    if total > 0 or L == 0:
                        p_t = (n_t + q_t()) / (total + 1.0)
                        break
                if p_t is None:
                    p_t = 1.0 / V
            else:
                num = den = 0.0
                q = q_t()
                for L, entry in levels:
                    n_t, total = entry[fi]
                    if total > 0 or L == 0:
                        w = 2.0 ** L                 # the fold factor, forced
                        num += w * (n_t + q) / (total + 1.0)
                        den += w
                p_t = num / den if den else 1.0 / V
            losses.append(-np.log(max(p_t, 1e-12)))
    return float(np.mean(losses))

# ---- SELF-TESTS ----
_rr = open(BASE + "/fold_ai/rung5b_rematch_results.txt").read()
REM_F = float(re.search(r"held-out CE: ([0-9.]+)", _rr).group(1))
REM_N = float(re.search(r"train only \(flood withheld\): ([0-9.]+)", _rr).group(1))
TWIN = float(re.search(r"twin mean \(seeds 1,2,3\):\s+([0-9.]+)", _rr).group(1))
a0f, a0n = arm_ce(True, False, "uniform"), arm_ce(False, False, "uniform")
assert abs(a0f - REM_F) < 1e-3 and abs(a0n - REM_N) < 1e-3, \
    f"self-test (a) FAILED: {a0f}/{a0n} vs rematch {REM_F}/{REM_N} -- run void"
print(f"self-test (a) baseline == rematch ({a0f:.4f}/{a0n:.4f}): PASS", flush=True)
# (b) one holding level: mixing collapses to hard backoff exactly
# (levels with total==0 and L>0 contribute nothing; L=0 always holds --
#  the collapse case is: only L=0 holds)
_lv0 = [(1, [(0, 0), (0, 0)]), (0, [(5, 9), (5, 9)])]
_hard0 = (5 + 1.0 / V) / 10.0
num = den = 0.0
for L, entry in _lv0:
    n_t, total = entry[0]
    if total > 0 or L == 0:
        w = 2.0 ** L
        num += w * (n_t + 1.0 / V) / (total + 1.0)
        den += w
assert abs(num / den - _hard0) < 1e-15, "self-test (b) FAILED -- run void"
print("self-test (b) single-level collapse: PASS", flush=True)
# (c) kin shape sums to 1 over the event space EXACTLY (one real position):
# the case-fold bridge distributes each folded neighbour by measured shares,
# so the total is (S + V*(1/V)) / (S + 1) = 1 by construction
_last = next(l for row in POS for (lv, l, tw, ti) in row if NEIGH.get(l))
mass = sum(kin_shape(_last, w, stoi[w]) for w in stoi) + kin_shape(_last, "\x00", RARE)
assert abs(mass - 1.0) < 1e-6, f"self-test (c) FAILED: mass {mass} -- run void"
print(f"self-test (c) kin floor mass {mass:.10f} == 1: PASS", flush=True)

# ---- RUN ALL ARMS ----
_5d = open(BASE + "/fold_ai/rung5d_results.txt").read()
_5dref = re.search(r"k= 64/128  loud-shaped ([0-9.]+)", _5d).group(1)
lines = [f"RUNG 5e RESULTS -- same-day twin {TWIN}, 5d loud-hard reference {_5dref} (parsed live)",
         f"A0 baseline        flood {a0f:.4f}   no-flood {a0n:.4f}   (== rematch, identity held)"]
results = {}
for name, mix, floor in (("A1 fold-mix       ", True, "uniform"),
                         ("A2 kin floor      ", False, "kin"),
                         ("A3 mix + kin      ", True, "kin"),
                         ("A4 mix + loud64   ", True, "loud")):
    cf, cn = arm_ce(True, mix, floor), arm_ce(False, mix, floor)
    results[name.strip()] = (cf, cn)
    ln = f"{name}flood {cf:.4f}   no-flood {cn:.4f}"
    for v, tag in ((cf, "flood"), (cn, "no-flood")):
        if v <= TWIN + 0.005:
            ln += f"   <- REACHES the twin ({tag})"
    print(ln, flush=True)
    lines.append(ln)
best = min(min(v) for v in results.values())
lines.append(f"BEST fold-stack arm: {best:.4f} vs twin {TWIN} -- "
             + ("the word-scale gap CLOSES within the fold" if best <= TWIN + 0.005
                else f"gap remaining {best - TWIN:.4f} nats, recorded"))
print(lines[-1], flush=True)
open(BASE + "/fold_ai/rung5e_results.txt", "w").write("\n".join(lines) + "\n")
print("RUNG 5e COMPLETE", flush=True)
