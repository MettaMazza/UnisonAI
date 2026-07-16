"""REASONING DECODE -- deriving reasoning from the black box (registered).

Three instruments:

  P1 ACTIVATION SPECTROMETER. Forward-pass hidden states captured by hook,
     fed to the locked battery: is the COMPUTATION dyadically loud where the
     WEIGHTS are loud? First science row: GPT-2 (weights-loud carrier,
     c_fc margins 3.4-12.7x) -- MLP pre-activation vectors on registered
     prompts, per-layer battery margin beside the recorded weight margin.
     The 32B reasoning/sibling pair (2h) is REGISTERED as this instrument's
     next object; it needs an activation-exposing runtime for 32B GGUFs,
     recorded here as the named dependency, not silently dropped.

  P2 ATTENTION-CASCADE DECODER. The theorem-forced selection law (Claim
     XI-2) is the dyadic cascade: ranked foci hold 1/2, 1/4, 1/8, ...,
     final candidate takes the closing remainder so the total is exactly
     One. Statistic: total-variation distance of each (layer, head, query)
     attention row -- rank-sorted -- from the cascade, vs the uniform
     distribution's TV and a shuffled-mass null. Calibration: an explicit
     cascade must read TV = 0; uniform must read its analytic TV.
     Reading: does trained softmax attention approximate the fold's
     selection law -- the 4b question approached from the decode side, in
     the incumbent's own machine, not as a transplant.

  P3 THOUGHT-TRACE FOLD MAP. A thinking model's live output split into
     reasoning span (<think>...</think>) vs answer span; each span's
     counted signature measured in the fold's own terms: branching factor
     (distinct continuations per held context) and repetition mass at the
     structural depth 6. Null: the same statistic on span-shuffled text.
     Reading: reasoning as a measurable structure in the counted domain.

Seed 20260706. Negative results recorded in full.
"""
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run, SEED, battery

WINDOW = 6  # CTX_MAX = GEN_B * GEN_C, the structural depth

PROMPTS = (
    "The derivative of a function measures the rate at which",
    "In 1869 Mendeleev arranged the known elements into",
    "A prime number is a natural number greater than one that",
    "The speed of light in a vacuum is a universal constant equal to",
    "Energy can neither be created nor destroyed, only",
    "The binary system represents every number using only",
    "Water is composed of two atoms of hydrogen and one atom of",
    "To solve a quadratic equation one can apply the",
)

REG = {
    "name": "reasoning-decode",
    "objects": ["GPT-2 (HF cache): MLP pre-activations + attention maps on 8 "
                "registered prompts",
                "qwen3:8b via Ollama: <think> reasoning spans vs answer spans",
                "NAMED DEPENDENCY: R1-Distill-32B vs qwen2.5-coder-32b activation "
                "run requires an activation-exposing GGUF runtime; registered as "
                "this instrument's next object"],
    "statistic": "P1 battery margin on concatenated activation vectors per layer; "
                 "P2 mean TV distance of rank-sorted attention rows from the dyadic "
                 "cascade vs uniform-TV and shuffled-mass null; P3 branching factor "
                 "and repetition mass at depth 6, reasoning vs answer spans, with "
                 "span-shuffled null",
    "verdict_rule": "P1: activations loud (margin > 2x) in layers whose weights are "
                    "loud = the computation carries the law; P2: cascade closer than "
                    "uniform on median row = trained attention leans toward the "
                    "fold's selection law; P3: reasoning-vs-answer signature gap "
                    "beyond the shuffled null = reasoning is counted structure",
    "margin_clause": "P1 wake bar 2x (the 2g bar); P2 claim requires median "
                     "TV(cascade) < TV(uniform) AND < shuffled null; P3 gap must "
                     "exceed the null gap by 1.5x",
}


def cascade(n):
    """The theorem's distribution over n ranked foci: 1/2, 1/4, ...,
    closing remainder -- telescopes to exactly one."""
    c = np.array([2.0 ** -(i + 1) for i in range(n)])
    c[-1] += 1.0 - c.sum()
    return c


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


# ------------------------------------------------------------------ P1 + P2

def gpt2_rows(run, do_p1=True):
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", output_attentions=True,
                                            output_hidden_states=True)
    model.eval()

    acts = defaultdict(list)

    def hook(layer):
        def fn(_m, inp, _out):
            acts[layer].append(inp[0].detach().squeeze(0).numpy())
        return fn

    hooks = ([blk.mlp.c_fc.register_forward_hook(hook(i))
              for i, blk in enumerate(model.transformer.h)] if do_p1 else [])

    attn_all = []
    with torch.no_grad():
        for p in PROMPTS:
            out = model(**tok(p, return_tensors="pt"))
            attn_all.append([a.squeeze(0).numpy() for a in out.attentions])
    for h in hooks:
        h.remove()

    # P1: per-layer battery on the concatenated pre-activation stream
    weight_margin = {0: 12.7, 1: 8.5, 2: 4.1, 3: 3.4, 4: 5.5, 5: 4.8,
                     6: 6.3, 7: 5.9, 8: 8.0, 9: 7.3, 10: 6.7, 11: 8.8}
    print("\n[P1] activation spectrometer -- GPT-2 MLP pre-activations:", flush=True)
    for L in sorted(acts):
        v = np.concatenate([a.ravel() for a in acts[L]])
        rec = battery(v, n_shuffle=3, comparators=False)
        run.record(instrument="activation-battery", model="GPT-2",
                   object=f"h.{L}.mlp pre-activation", layer=L,
                   weight_margin_recorded=weight_margin[L], **rec)
        print(f"  L{L:2d} activation margin {rec['margin']:6.2f}x   "
              f"(weight margin on record {weight_margin[L]:.1f}x)", flush=True)

    # P2: cascade decoder over every (layer, head, query) attention row
    print("\n[P2] attention-cascade decoder -- GPT-2:", flush=True)
    c_probe = cascade(8)
    if tv(c_probe, c_probe) != 0.0:
        from foldprobe import halt
        halt("cascade self-calibration failed -- TV(cascade,cascade) != 0")
    # NULL (corrected at registration of this run): the rank-sorted statistic
    # is permutation-invariant, so a shuffle of the same row is vacuous. The
    # null is a seeded uniform-simplex (Dirichlet-1) row of the same length --
    # what a structureless attention row would read.
    rng = np.random.default_rng(SEED)
    per_layer = []
    for L in range(len(attn_all[0])):
        tvs_c, tvs_u, tvs_null = [], [], []
        for pa in attn_all:
            A = pa[L]                      # (heads, T, T)
            H, T, _ = A.shape
            for h in range(H):
                for qy in range(2, T):     # queries with >= 3 candidates
                    row = A[h, qy, :qy + 1]
                    row = row / row.sum()
                    srt = np.sort(row)[::-1]
                    n = len(srt)
                    tvs_c.append(tv(srt, cascade(n)))
                    tvs_u.append(tv(srt, np.full(n, 1.0 / n)))
                    null_row = rng.dirichlet(np.ones(n))
                    tvs_null.append(tv(np.sort(null_row)[::-1], cascade(n)))
        med_c, med_u = float(np.median(tvs_c)), float(np.median(tvs_u))
        med_n = float(np.median(tvs_null))
        per_layer.append((med_c, med_u))
        run.record(instrument="cascade-decoder", model="GPT-2", layer=L,
                   median_tv_cascade=med_c, median_tv_uniform=med_u,
                   median_tv_shuffled_null=med_n, rows=len(tvs_c),
                   cascade_closer=bool(med_c < med_u))
        print(f"  L{L:2d} median TV: cascade {med_c:.4f}  uniform {med_u:.4f}  "
              f"shuffled-null {med_n:.4f}  {'CASCADE CLOSER' if med_c < med_u else 'uniform closer'}",
              flush=True)
    closer = sum(1 for c, u in per_layer if c < u)
    run.record(instrument="verdict", object="P2-cascade",
               layers_cascade_closer=closer, layers_total=len(per_layer))
    print(f"  cascade closer than uniform in {closer}/{len(per_layer)} layers", flush=True)


# ----------------------------------------------------------------------- P3

def ollama_think(model, prompt, n_predict=3000):
    """Native think-mode call (the teacher_scaffold lesson: reasoning arrives
    in the JSON 'thinking' field, not as inline tags)."""
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "think": True,
                         "options": {"temperature": 0, "num_predict": n_predict}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    return d.get("thinking", ""), d.get("response", "")


def counted_signature(text):
    """Branching factor + repetition mass of a span at the structural
    depth: distinct continuations per held depth-6 char context, and the
    share of contexts seen more than once. Pure counting, no parameters."""
    store = defaultdict(set)
    seen = defaultdict(int)
    t = re.sub(r"\s+", " ", text)
    for i in range(len(t) - WINDOW):
        ctx = t[i:i + WINDOW]
        store[ctx].add(t[i + WINDOW])
        seen[ctx] += 1
    if not store:
        return None
    branching = float(np.mean([len(s) for s in store.values()]))
    rep_mass = sum(c for c in seen.values() if c > 1) / max(sum(seen.values()), 1)
    return branching, rep_mass


THINK_QS = (
    "If a train travels 180 km in 2 hours and then 120 km in 1.5 hours, what is "
    "its average speed for the whole journey?",
    "A rectangle's length is twice its width and its perimeter is 36. What is its area?",
    "If 3 machines make 3 widgets in 3 minutes, how long do 100 machines take to "
    "make 100 widgets?",
    "What is the sum of the first 20 positive odd numbers?",
)


def trace_map(run, model="qwen3:8b"):
    print(f"\n[P3] thought-trace fold map -- {model}:", flush=True)
    rng = np.random.default_rng(SEED)
    gaps, null_gaps = [], []
    for q in THINK_QS:
        try:
            think, answer = ollama_think(model, q)
        except Exception as e:
            run.record(instrument="trace-map", model=model,
                       skipped=f"ollama unavailable: {e}")
            print(f"  skipped: ollama unavailable ({e})", flush=True)
            return
        if len(think) < 200 or len(answer) < 100:
            run.record(instrument="trace-map", model=model, object=q[:48],
                       skipped=f"no usable think/answer split "
                               f"(think {len(think)}, answer {len(answer)} chars)")
            print(f"  skipped {q[:40]}: think {len(think)} / answer {len(answer)} chars",
                  flush=True)
            continue
        s_t = counted_signature(think)
        s_a = counted_signature(answer)
        both = list(think + answer)
        rng.shuffle(both)
        half = len(both) // 2
        n_t = counted_signature("".join(both[:half]))
        n_a = counted_signature("".join(both[half:]))
        if not all((s_t, s_a, n_t, n_a)):
            continue
        gap = abs(s_t[1] - s_a[1])
        ngap = abs(n_t[1] - n_a[1])
        gaps.append(gap)
        null_gaps.append(ngap)
        run.record(instrument="trace-map", model=model, object=q[:48],
                   think_branching=s_t[0], think_rep_mass=s_t[1],
                   answer_branching=s_a[0], answer_rep_mass=s_a[1],
                   rep_mass_gap=gap, shuffled_null_gap=ngap)
        print(f"  rep-mass think {s_t[1]:.3f} vs answer {s_a[1]:.3f} "
              f"(gap {gap:.3f}, null {ngap:.3f})  branching {s_t[0]:.2f}/{s_a[0]:.2f}",
              flush=True)
    if gaps:
        ratio = float(np.mean(gaps) / max(np.mean(null_gaps), 1e-9))
        run.record(instrument="verdict", object="P3-trace-map",
                   mean_gap=float(np.mean(gaps)),
                   mean_null_gap=float(np.mean(null_gaps)), ratio=ratio,
                   counted_structure=bool(ratio > 1.5))
        print(f"  P3: mean gap {np.mean(gaps):.3f} vs null {np.mean(null_gaps):.3f} "
              f"-> ratio {ratio:.1f}x ({'COUNTED STRUCTURE' if ratio > 1.5 else 'within null'})",
              flush=True)


def main():
    which = set(sys.argv[1:]) or {"p1", "p2", "p3"}
    run = Run(REG)
    if which & {"p1", "p2"}:
        gpt2_rows(run, do_p1="p1" in which)
    if "p3" in which:
        trace_map(run)
    print("\nREASONING DECODE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
