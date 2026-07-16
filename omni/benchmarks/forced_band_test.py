"""THE FORCED-BAND TEST -- the functional band's direct verification
(registered; the check of Step 311, constants/functional_band.ep).

THE FORCED CLAIM UNDER TEST: the band that carries a held field's function
is b^cover(c^3) = 32 coefficients per block -- not an instrument choice.
The natural block of a weight tensor is its row (one unit's field); each
row's spectral window is its largest power-of-two prefix (instrument-class,
stated).

ARMS at matched budget, per row of GPT-2's law-bearing class (wte + all
c_fc, the presence suite's 13 tensors):
    ablate-band    zero the top-32 |Walsh coefficients| of every row
    ablate-random  zero 32 random coefficients of every row (seeded)

READOUT (the capability map's own): mean next-token KL vs the unmodified
model + top-1 agreement, on the registered 16-prompt set. Calibration:
the k=0 arm must reproduce the model exactly.

VERDICT RULE (fixed before any run): the forced band CARRIES THE FUNCTION
iff band-ablation's KL exceeds 10x random-ablation's KL and agreement
under band-ablation falls below 1/2 (the lock) while random stays above.
Recorded whichever way it lands.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import Run, SEED, halt

BAND = 32   # forced: b^cover(c^3), Step 311 -- not tunable here or anywhere

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

REG = {
    "name": "forced-band-test",
    "objects": ["GPT-2 (HF cache): wte + all 12 c_fc, per-row ablation at the "
                "FORCED band k = 32 (Step 311) vs 32 random per row; row window "
                "= largest power-of-two prefix (instrument-class)"],
    "statistic": "mean next-token KL + top-1 agreement vs unmodified, registered "
                 "16-prompt set; k=0 calibration must be exact",
    "verdict_rule": "the forced band carries the function iff band-KL > 10x "
                    "random-KL and band agreement < 1/2 (the lock) with random "
                    "agreement >= 1/2; recorded whichever way",
    "margin_clause": "10x KL ratio and the 1/2 lock, fixed before any run; the "
                     "band size 32 is FORCED (b^cover(c^3)), not registered here",
}


def wht_rows(W):
    n = 1 << int(np.floor(np.log2(W.shape[1])))
    a = W[:, :n].astype(np.float64).copy()
    h = 1
    while h < n:
        a = a.reshape(a.shape[0], -1, 2 * h)
        x = a[:, :, :h].copy()
        a[:, :, :h] += a[:, :, h:]
        a[:, :, h:] = x - a[:, :, h:]
        a = a.reshape(a.shape[0], -1)
        h *= 2
    return a, n


def ablate_rows(W, mode, rng):
    spec, n = wht_rows(W)
    if mode == "band":
        idx = np.argsort(np.abs(spec), axis=1)[:, -BAND:]
    else:
        idx = np.stack([rng.choice(n, size=BAND, replace=False)
                        for _ in range(spec.shape[0])])
    np.put_along_axis(spec, idx, 0.0, axis=1)
    back, _ = wht_rows(spec.reshape(W.shape[0], n))
    out = W.astype(np.float64).copy()
    out[:, :n] = back / n
    return out


def main():
    import torch
    import torch.nn.functional as F
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    run = Run(REG)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    names = ["transformer.wte.weight"] + \
        [f"transformer.h.{i}.mlp.c_fc.weight" for i in range(12)]

    def readout():
        out = []
        with torch.no_grad():
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt")
                out.append(F.log_softmax(model(**ids).logits[0, -1].double(), dim=0).numpy())
        return out

    base_logp = readout()
    base_top1 = [int(np.argmax(lp)) for lp in base_logp]

    def compare():
        logp = readout()
        kl = float(np.mean([np.sum(np.exp(b) * (b - a)) for b, a in zip(base_logp, logp)]))
        ag = float(np.mean([int(np.argmax(a)) == t for a, t in zip(logp, base_top1)]))
        return kl, ag

    # calibration: reload base state exactly
    model.load_state_dict(base_state)
    kl0, ag0 = compare()
    if kl0 != 0.0 or ag0 != 1.0:
        halt(f"calibration failed: KL {kl0}, agreement {ag0}")
    run.record(instrument="verdict", object="calibration", kl=kl0, agreement=ag0)
    print(f"  calibration: KL {kl0}, agreement {ag0} -- exact", flush=True)

    rng = np.random.default_rng(SEED)
    results = {}
    for mode in ("band", "random"):
        sd = {k: v.clone() for k, v in base_state.items()}
        for name in names:
            W = base_state[name].numpy()
            ab = torch.from_numpy(ablate_rows(W, mode, rng).astype(np.float32))
            sd[name] = ab
            if name == "transformer.wte.weight":
                sd["lm_head.weight"] = ab   # the tied-weights lesson, recorded
        model.load_state_dict(sd)
        kl, ag = compare()
        results[mode] = (kl, ag)
        run.record(instrument="forced-band", mode=mode, k_per_row=BAND,
                   kl=round(kl, 5), agreement=round(ag, 4))
        print(f"  {mode:6s} k={BAND}/row: KL {kl:8.4f}  agreement {ag:.2f}", flush=True)

    b_kl, b_ag = results["band"]
    r_kl, r_ag = results["random"]
    holds = (b_kl > 10 * max(r_kl, 1e-9)) and (b_ag < 0.5) and (r_ag >= 0.5)
    run.record(instrument="verdict", object="forced-band-test",
               band_kl=round(b_kl, 5), random_kl=round(r_kl, 5),
               band_agreement=round(b_ag, 4), random_agreement=round(r_ag, 4),
               kl_ratio=round(b_kl / max(r_kl, 1e-9), 1), holds=bool(holds))
    print(f"\nFORCED-BAND TEST: {'THE FORCED BAND CARRIES THE FUNCTION' if holds else 'rule not met as registered -- recorded'} "
          f"(KL ratio {b_kl / max(r_kl, 1e-9):.0f}x; agreement {b_ag:.2f} vs {r_ag:.2f})", flush=True)


if __name__ == "__main__":
    main()
