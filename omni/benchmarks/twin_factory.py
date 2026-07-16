"""TWIN FACTORY -- what and when training writes the law (Rungs 6a + 6b
as one instrument, registered).

THE MATRIX (6a): tiny twins of the recorded Rung-4/5 recipe, ONE ingredient
varied per twin, everything else pinned to the baseline:

    baseline   AdamW, lr 3e-4 constant, no weight decay, no dropout,
               sequential data order, seed 1  (the recorded 1.8878 recipe
               at char scale / the 5b recipe at word scale)
    optimizer  SGD+momentum(0.9) in place of AdamW
    wdecay     weight decay 0.1
    lrsched    cosine decay to 0.1x
    dropout    p = 0.1 on ff + attention output
    dataorder  shuffled document order (seed-derived)
    rlpass     a post-training REINFORCE pass: reward = negative CE on a
               held slice (the reasoning-RL ingredient in miniature)

THE CLOCK (6b): every twin checkpoints at steps 2^k (dyadic sampling of
the training axis -- the fold's own resolution of a growth process,
log-cost) and every checkpoint's loud-class tensors (ff expansion + wte)
run the locked battery. Output: margin(step; ingredient).

THE STREAM: at each checkpoint the accumulated gradient of the same
tensors over the last 8 batches runs the same battery -- is the UPDATE
STREAM loud before the weights are? (the gradient-stream spectrometer).

Verdict surface (registered): per ingredient, the deposition curve
margin(2^k); the ingredient(s) whose curve departs from the baseline's
identify WHAT writes the law; the k at which margin exceeds the wake bar
identifies WHEN. Wake bar 2x (the 2g bar). Negative curves recorded.
"""
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run, battery

BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"
SFTOM = "/Users/mettamazza/Desktop/SFTOM"
CTX_T, DIM, HEADS, LAYERS = 64, 128, 4, 4
STEPS, BATCH, LR = 2048, 32, 3e-4          # 2^11 steps: 12 dyadic checkpoints
SEED = 1                                    # the recorded twin seed
GRAD_WINDOW = 8

INGREDIENTS = ("baseline", "optimizer", "wdecay", "lrsched", "dropout",
               "dataorder", "rlpass")

REG = {
    "name": "twin-factory-6ab",
    "objects": [f"tiny twins (recorded 4L/128d recipe, seed {SEED}), one per "
                f"ingredient: {', '.join(INGREDIENTS)}; loud-class tensors "
                "(ff expansion + wte) + gradient accumulations"],
    "statistic": "locked battery margin at checkpoints 2^k, k=0..11, on weights "
                 "and on the 8-batch gradient accumulation",
    "verdict_rule": "WHAT: ingredient whose deposition curve departs from the "
                    "baseline curve beyond the seed band; WHEN: first k with "
                    "margin > wake bar; gradient stream louder earlier than "
                    "weights = the law arrives through the updates",
    "margin_clause": "wake bar 2x (the 2g bar); curve departure requires >= 2 "
                     "consecutive checkpoints outside the baseline's min/max band",
}


def build_corpus():
    import glob
    import re
    files = sorted(glob.glob(BASE + "/**/*.md", recursive=True)) + \
        sorted(glob.glob(SFTOM + "/**/*.md", recursive=True))
    files = [f for f in files if "/.git/" not in f and "/language/" not in f]
    text = "\n".join(open(f, errors="ignore").read() for f in files)
    toks = re.findall(r"\w+|[^\w\s]", text)[:2_000_000]
    cnt = Counter(toks)
    stoi = {w: i for i, w in enumerate([w for w, c in cnt.items() if c >= 3])}
    ids = np.array([stoi.get(t, len(stoi)) for t in toks], dtype=np.int64)
    return ids, len(stoi) + 1


def make_twin(V, ingredient):
    import torch
    import torch.nn as nn

    drop = 0.1 if ingredient == "dropout" else 0.0

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
            self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True, dropout=drop)
            self.ff = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(4 * DIM, DIM))
        def forward(self, x, mask):
            a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                             attn_mask=mask, need_weights=False)
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
            import torch as t
            T = idx.shape[1]
            x = self.wte(idx) + self.wpe(t.arange(T))
            mask = t.triu(t.ones(T, T, dtype=t.bool), 1)
            for b in self.blocks:
                x = b(x, mask)
            return self.head(self.lnf(x))

    torch.manual_seed(SEED)
    m = Tiny()
    if ingredient == "optimizer":
        opt = torch.optim.SGD(m.parameters(), lr=1e-2, momentum=0.9)
    elif ingredient == "wdecay":
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.1)
    else:
        opt = torch.optim.AdamW(m.parameters(), lr=LR)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=LR * 0.1)
             if ingredient == "lrsched" else None)
    return m, opt, sched


def loud_tensors(m):
    yield "wte", m.wte.weight
    for i, b in enumerate(m.blocks):
        yield f"blocks.{i}.ff.0", b.ff[0].weight


def probe_state(run, ingredient, step, m, grads):
    for name, W in loud_tensors(m):
        rec = battery(W.detach().numpy().ravel(), n_shuffle=3, comparators=False)
        run.record(instrument="deposition", ingredient=ingredient, step=step,
                   object=name, kind="weights", margin=rec["margin"],
                   beyond=sum(rec["beyond_nulls"]))
        if name in grads and grads[name] is not None:
            g = battery(grads[name].ravel(), n_shuffle=3, comparators=False)
            run.record(instrument="deposition", ingredient=ingredient, step=step,
                       object=name, kind="grad-accum", margin=g["margin"],
                       beyond=sum(g["beyond_nulls"]))


def train_twin(run, ingredient, ids, V):
    import torch
    import torch.nn.functional as F
    m, opt, sched = make_twin(V, ingredient)
    n_split = int(0.9 * len(ids))
    train_ids = ids[:n_split].copy()
    if ingredient == "dataorder":
        blocks = train_ids[:len(train_ids) // 4096 * 4096].reshape(-1, 4096)
        np.random.default_rng(SEED).shuffle(blocks)
        train_ids = blocks.ravel()
    td = torch.from_numpy(train_ids)
    rng = np.random.default_rng(SEED)

    checkpoints = {1 << k for k in range(0, 12)}
    grads = {}
    grad_accum = {name: None for name, _ in loud_tensors(m)}
    t0 = time.time()
    print(f"\n[{ingredient}] training {STEPS} steps...", flush=True)
    probe_state(run, ingredient, 0, m, {})
    for s in range(1, STEPS + 1):
        ix = torch.from_numpy(rng.integers(0, len(td) - CTX_T - 1, BATCH))
        x = torch.stack([td[i:i + CTX_T] for i in ix])
        y = torch.stack([td[i + 1:i + CTX_T + 1] for i in ix])
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        if s > STEPS - GRAD_WINDOW or (s & (s - 1)) == 0:
            named = dict(loud_tensors(m))
            for name, W in named.items():
                g = W.grad
                if g is not None:
                    grad_accum[name] = (g.detach().numpy().copy()
                                        if grad_accum[name] is None
                                        else grad_accum[name] + g.detach().numpy())
        opt.step()
        if sched:
            sched.step()
        if s in checkpoints:
            grads = {k: v for k, v in grad_accum.items()}
            probe_state(run, ingredient, s, m, grads)
            grad_accum = {name: None for name, _ in loud_tensors(m)}
            print(f"  [{ingredient}] checkpoint 2^{int(np.log2(s))} = {s} "
                  f"(loss {loss.item():.3f}, {time.time()-t0:.0f}s)", flush=True)

    if ingredient == "rlpass":
        # the reasoning-RL ingredient in miniature: REINFORCE on the same
        # twin, reward = negative CE of the sampled continuation on a held
        # slice; 256 post-training steps, checkpointed dyadically
        held = torch.from_numpy(ids[n_split:])
        print(f"  [{ingredient}] RL pass (REINFORCE, 256 steps)...", flush=True)
        for s in range(1, 257):
            ix = torch.from_numpy(rng.integers(0, len(held) - CTX_T - 1, 8))
            x = torch.stack([held[i:i + CTX_T] for i in ix])
            y = torch.stack([held[i + 1:i + CTX_T + 1] for i in ix])
            logits = m(x)
            logp = F.log_softmax(logits, dim=-1)
            samp = torch.distributions.Categorical(logits=logits).sample()
            reward = -F.cross_entropy(
                logits.reshape(-1, V), y.reshape(-1), reduction="none"
            ).reshape(samp.shape).detach()
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)
            rl_loss = -(logp.gather(-1, samp.unsqueeze(-1)).squeeze(-1) * reward).mean()
            opt.zero_grad()
            rl_loss.backward()
            opt.step()
            if (s & (s - 1)) == 0:
                probe_state(run, ingredient, STEPS + s, m, {})
        print(f"  [{ingredient}] RL pass done", flush=True)

    # the recorded-recipe reproduction check rides on the baseline twin
    if ingredient == "baseline":
        with torch.no_grad():
            val = ids[n_split:]
            vd = torch.from_numpy(val)
            losses = []
            vr = np.random.default_rng(999)
            for _ in range(64):
                i = int(vr.integers(0, len(vd) - CTX_T - 1))
                x = vd[i:i + CTX_T].unsqueeze(0)
                y = vd[i + 1:i + CTX_T + 1].unsqueeze(0)
                losses.append(F.cross_entropy(m(x).reshape(-1, V),
                                              y.reshape(-1)).item())
            ce = float(np.mean(losses))
        run.record(instrument="calibration", ingredient="baseline",
                   heldout_ce=ce, note="word-scale recipe; the char-scale "
                   "1.8878 record is a different arena -- this CE is the "
                   "factory's own pinned reference for future re-runs")
        print(f"  [baseline] held-out CE {ce:.4f} (pinned as the factory reference)",
              flush=True)


def main():
    run = Run(REG)
    print("[twin-factory] building corpus...", flush=True)
    ids, V = build_corpus()
    print(f"  {len(ids)} tokens, vocab {V}", flush=True)
    for ing in INGREDIENTS:
        train_twin(run, ing, ids, V)
    print("\nTWIN FACTORY COMPLETE -- deposition curves in results.jsonl "
          "(instrument='deposition')", flush=True)


if __name__ == "__main__":
    main()
