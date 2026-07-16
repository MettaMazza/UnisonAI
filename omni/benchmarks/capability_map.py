"""CAPABILITY MAP -- which loud coefficients carry which behavior
(registered). The decode from spectrum to function.

CONSTRUCTION (Rung 3's harness, turned from compression to attribution):
for each law-bearing tensor class of GPT-2 (the presence-proven hot class:
wte + c_fc, margins 3.4-79.3x), flatten to the largest 2^n prefix, WHT,
zero a REGISTERED band, inverse, write back. Arms per class and budget:

    ablate-loud-k    zero the top-k |coefficient| band (the law carriers)
    ablate-random-k  zero k random coefficients, seed 20260706 (the control:
                     same parameter damage, law left standing)

BUDGETS: k = n/64 and n/16 of coefficients (registered before any run).

READOUT (fixed): on the 16-prompt registered set, (a) mean KL of the
next-token distribution vs the unmodified model, (b) top-1 agreement;
per-DOMAIN mean NLL over the mmlu_probe.json question texts grouped by
subject -- the capability axes. ATTRIBUTION = loud-ablation hurting a
domain more than random-ablation at matched budget names the coefficients
that carry it.

CALIBRATION (must pass first): the k=0 arm reproduces the unmodified
model EXACTLY -- KL = 0, agreement = 1, identical domain NLLs.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run, SEED, halt

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = (
    "The capital of France is",
    "Two plus two equals",
    "The chemical symbol for gold is",
    "Once upon a time there was a",
    "The theory of relativity was developed by",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "To be or not to be, that is the",
    "The square root of sixteen is",
    "DNA stands for",
    "The first president of the United States was",
    "Light travels faster than",
    "A triangle has three",
    "The opposite of hot is",
    "Photosynthesis occurs in the",
    "The sum of the angles in a triangle is",
)

CLASSES = {
    "wte": ["wte"],
    "c_fc-early": [f"h.{i}.mlp.c_fc" for i in range(4)],
    "c_fc-mid": [f"h.{i}.mlp.c_fc" for i in range(4, 8)],
    "c_fc-late": [f"h.{i}.mlp.c_fc" for i in range(8, 12)],
}

REG = {
    "name": "capability-map",
    "objects": ["GPT-2 (HF cache), tensor classes: wte, c_fc early/mid/late",
                "16 registered prompts; mmlu_probe.json question texts grouped "
                "by subject as the domain axes"],
    "statistic": "mean next-token KL + top-1 agreement vs unmodified; per-domain "
                 "mean NLL delta; arms ablate-loud vs ablate-random at k = n/64, n/16",
    "verdict_rule": "ATTRIBUTION: a (class, domain) pair where loud-ablation's NLL "
                    "delta exceeds random-ablation's at both budgets = those loud "
                    "coefficients carry that capability",
    "margin_clause": "attribution requires loud-delta > 1.5x random-delta at both "
                     "budgets; calibration (k=0) must read KL=0, agreement=1 exactly",
}


def wht_np(a):
    n = len(a)
    h = 1
    a = a.copy()
    while h < n:
        a = a.reshape(-1, 2 * h)
        x = a[:, :h].copy()
        a[:, :h] += a[:, h:]
        a[:, h:] = x - a[:, h:]
        a = a.reshape(-1)
        h *= 2
    return a


def ablate(W, k, mode, rng):
    """Zero a k-coefficient band of the largest 2^n prefix, in place-shape."""
    flat = W.reshape(-1).astype(np.float64)
    n = 1 << int(np.floor(np.log2(len(flat))))
    v = flat[:n].copy()
    spec = wht_np(v)
    if mode == "loud":
        idx = np.argsort(np.abs(spec))[-k:]
    else:
        idx = rng.choice(n, size=k, replace=False)
    spec[idx] = 0.0
    v2 = wht_np(spec) / n
    out = flat.copy()
    out[:n] = v2
    return out.reshape(W.shape)


def domain_texts():
    """mmlu_probe.json carries no subject field; the registered domain axes
    are its 8 consecutive 16-question blocks (deterministic, fixed before
    any run -- the probe file's own order)."""
    probe = json.load(open(os.path.join(HERE, "mmlu_probe.json")))
    domains = {}
    for b in range(8):
        qs = [item["q"] for item in probe[b * 16:(b + 1) * 16]]
        domains[f"probe-block-{b}"] = " ".join(qs)
    return domains


def main():
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    run = Run(REG)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    name_of = {"wte": "transformer.wte.weight"}
    for i in range(12):
        name_of[f"h.{i}.mlp.c_fc"] = f"transformer.h.{i}.mlp.c_fc.weight"

    domains = domain_texts()
    print(f"[capability-map] domains: {sorted(domains)}", flush=True)

    def readout():
        kls, agree, dom_nll = [], [], {}
        with torch.no_grad():
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt")
                logits = model(**ids).logits[0, -1].double()
                kls.append(F.log_softmax(logits, dim=0).numpy())
            for d, text in domains.items():
                ids = tok(text, return_tensors="pt", truncation=True, max_length=512)
                out = model(**ids, labels=ids["input_ids"])
                dom_nll[d] = float(out.loss)
        return kls, dom_nll

    print("[capability-map] baseline readout...", flush=True)
    base_logp, base_nll = readout()
    base_top1 = [int(np.argmax(lp)) for lp in base_logp]

    def compare(tag):
        logp, nll = readout()
        kl = float(np.mean([np.sum(np.exp(b) * (b - a)) for b, a in zip(base_logp, logp)]))
        ag = float(np.mean([int(np.argmax(a)) == t for a, t in zip(logp, base_top1)]))
        deltas = {d: nll[d] - base_nll[d] for d in nll}
        return kl, ag, deltas

    # ---- calibration: k = 0 must be the identity ----
    kl0, ag0, d0 = compare("k0")
    if not (kl0 == 0.0 and ag0 == 1.0 and all(v == 0.0 for v in d0.values())):
        halt(f"calibration FAIL: k=0 arm not identical (KL {kl0}, agree {ag0})")
    run.record(instrument="verdict", object="calibration-k0", kl=kl0, agreement=ag0,
               passed=True)
    print(f"  calibration k=0: KL {kl0}, agreement {ag0} -- PASS", flush=True)

    rng = np.random.default_rng(SEED)
    attributions = []
    for cls, tensors in CLASSES.items():
        n_min = min(1 << int(np.floor(np.log2(base_state[name_of[t]].numel())))
                    for t in tensors)
        for frac_name, frac in (("n/64", 64), ("n/16", 16)):
            arms = {}
            for mode in ("loud", "random"):
                sd = {k: v.clone() for k, v in base_state.items()}
                for t in tensors:
                    W = base_state[name_of[t]].numpy()
                    k = (1 << int(np.floor(np.log2(W.size)))) // frac
                    ablated = torch.from_numpy(ablate(W, k, mode, rng).astype(np.float32))
                    sd[name_of[t]] = ablated
                    if t == "wte":
                        # GPT-2 ties lm_head.weight to wte: both keys must
                        # carry the ablated tensor or the tie restores the
                        # original on load (the KL=0.0 lesson, recorded)
                        sd["lm_head.weight"] = ablated
                model.load_state_dict(sd)
                kl, ag, deltas = compare(f"{cls}-{mode}-{frac_name}")
                arms[mode] = (kl, ag, deltas)
                run.record(instrument="capability", tensor_class=cls, mode=mode,
                           budget=frac_name, kl=kl, agreement=ag,
                           domain_nll_delta={d: round(v, 5) for d, v in deltas.items()})
                print(f"  {cls:10s} {mode:6s} k={frac_name:5s} KL {kl:8.5f} "
                      f"agree {ag:.2f}", flush=True)
            for d in domains:
                dl, dr = arms["loud"][2][d], arms["random"][2][d]
                if dl > 1.5 * max(dr, 1e-9) and dl > 0.01:
                    attributions.append((cls, frac_name, d, dl, dr))
        model.load_state_dict(base_state)

    # attribution requires the margin at BOTH budgets
    by_pair = defaultdict(set)
    for cls, fr, d, dl, dr in attributions:
        by_pair[(cls, d)].add(fr)
    confirmed = [{"class": c, "domain": d, "budgets": sorted(b)}
                 for (c, d), b in by_pair.items() if len(b) == 2]
    run.record(instrument="verdict", object="capability-map",
               attributions=confirmed)
    print(f"\nCAPABILITY MAP COMPLETE -- {len(confirmed)} attribution(s) at both budgets:",
          flush=True)
    for a in confirmed:
        print(f"  {a['class']} carries {a['domain']}", flush=True)


if __name__ == "__main__":
    main()
