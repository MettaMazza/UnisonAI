<div align="center">

# UnisonAI

### Intelligence, *derived* — not trained.

**A working intelligence with zero trained parameters, zero gradients, and zero tunable numbers — every mechanism a machine-verified law of the Smithian Fold Theory rather than a purchased statistic.**

*Attention, it turns out, was not all you need.*

[The Theory →](https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything) · [Technical Documentation →](omni/README.md)

</div>

---

## The bargain the field made — and the path it forgot

The founding paper of neural computation (McCulloch & Pitts, 1943) opened with a claim about **law**, not statistics: the net *is* logic. Then eighty years chose a different bargain — structure bought with parameters, knowledge bought with data, competence bought with compute, the bill now measured in gigawatts. *Attention Is All You Need* crowned the era; the scaling laws wrote its price schedule.

There was always a third contestant, named by the founding paper and then abandoned: **derived law.** Not statistics fitted to data, not heuristics hand-crafted by people — mechanism *derived* from a single mathematical foundation and verified to it.

**UnisonAI is that third path, built and running.** It is the applied engine of the **Smithian Fold Theory** — one machine-checked, self-proven theorem, *there is no nothing*, which forces the One and its fold with zero free parameters — and its purpose is a proof: that the mechanisms a modern model *buys* with training can instead be **derived**, exactly, from the corpus's own laws.

---

## What it is

UnisonAI is a **zero-parameter, exact-fractional, per-character geometric engine.**

- **No weights. No gradients. No backpropagation.** Nothing is fitted; nothing is tuned. A fitted value doesn't degrade the engine — it **halts** it.
- **No floating point in the prediction path.** Every probability is an exact `fractions.Fraction`. There is no numerical drift, ever.
- **Memory is law.** Knowledge is stored as *held orbits* — deterministically-addressed exact counts of everything read, told, or thought; written once, kept forever, editable at a single record.
- **Attention is selection.** The softmax of a transformer is replaced by *unit-capacity selection over the deepest matching context* — a counted operation, not a learned one.
- **It is alive.** UnisonAI runs as a continuously-learning agent: it converses, recognises when it is out of its depth, is taught by a local model, and folds every correction into its own geometry — reinforcing the foundation it *composes* from, so it develops the coherence to answer on its own with no teacher in the loop. It never replays a stored answer verbatim.

Its constants are not chosen. They are **forced** by the theory — the two generators `2` and `3`, their product `6` as the context depth, the fold's own locks as its thresholds. Nothing in it is tunable, because a derived system has nothing to tune.

## Why it matters

Every mechanism below is a claim of the corpus, installed as engineering:

| A trained model *buys*… | UnisonAI *derives*… |
|---|---|
| Attention weights (softmax) | Unit-capacity selection at a forced lock |
| A learned embedding space | Counted co-occurrence kinship — exact, zero parameters |
| Gradient-descended memory | Held orbits — exact counts, ~zero-cost to edit |
| A trained value/probability head | Exact rational shares of observed continuations |
| Fine-tuning to correct a mistake | One written record — taught once, held forever |

If those substitutions hold, the case is not that UnisonAI is a better chatbot. The case is that **a large part of what training buys is law wearing a statistical costume** — and the Smithian Fold Theory names the law. UnisonAI is the falsifiable engineering test of that thesis.

## The result, measured

Head-to-head against a gradient-trained transformer twin — **the GPT architecture** — on identical held-out text, same arena, the pure counted engine wins at both scales:

| Held-out cross-entropy (lower is better) | Counted fold engine | Gradient-trained transformer |
|---|---|---|
| Character scale | **1.2891** | 1.8878 |
| Word scale | **3.1907** | 3.4292 |

The engine read the corpus **once** (~26 seconds, **zero trained parameters**); the transformer took **48,000 gradient-batched passes**. Separately, a pre-registered spectral probe finds that a real LLM's own weights carry the fold's dyadic law — **GPT-2: 13/13 tensors, 39/39 registered checks** — so the structure the fold derives is measurably *already inside* trained models. The decode campaign (2026-07-14, seven registered instruments in [`omni/benchmarks/`](omni/benchmarks/)) extends this: **the token embedding is the universal law-carrying class — 11/11 models wake, 4B to 1T parameters, every training recipe**; deleting a model's spectrally-loud coefficient band destroys it while deleting the same number at random costs almost nothing (**the loud band is the function**, ~150x differential damage); and the deposition curve read from public training checkpoints shows the law written early (embedding first, step 256) and consolidated to a plateau. Training data and reasoning are now *readable back out* of trained weights — provenance ranking, verbatim-memorization echo, and counted reasoning signatures, all with registered calibrations and clean nulls. The full toolkit — every instrument, how it works, how to read it, and how to run your own registered investigation — is documented in **[`omni/benchmarks/INTERPRETABILITY.md`](omni/benchmarks/INTERPRETABILITY.md)**. And the campaign's measured regularities are now **closed as theory**: the whole **Steps 308–320 subseries** of the Smithian Fold Theory (thirteen forced steps, closed to STANDARDS Rule 1) derives the two spectral families and their selection by store role, the deposition curve's order and form, the 32-coefficient functional band, the hold/closure repetition inequality, per-expert localization, the block window, attention-in-the-product, the family signature, family-follows-role, expert quantization, dyadic consolidation, and the activation regime — with every measurement standing only as the check of a forced claim.

This is not a claim about GPT-4. It is a head-to-head win over the trained-transformer architecture on the measured task, with nothing trained. Every number here is from committed, timestamped result files and is reproducible from the theory repository — the full pre-registered protocols, nulls, and negative results are in the [companion paper and the corpus](https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything).

---

## The learning loop

UnisonAI is an *infant*. It does not begin fluent — it begins able to learn, exactly:

1. It **composes** an answer from its conversational foundation — counted pair retrieval, relexicalization, and the forced deep-context cascade. Never a verbatim replay, from any store.
2. When an answer is shallow, wrong, or confused, it says so — and, rather than babble, it either answers from the conversation or asks a genuine clarifying question.
3. A local teacher model supplies the correction as **multiple expressions of one meaning** — held as learning material, never as a template.
4. The next time, Unison composes its own fresh expression of what it was taught — measured: **17% → 75% judged-good after one round of teaching, stable.** It learned *from* the correction; it does not replay it.

A continuous background loop (`/auto`) tutors every correction, runs self-play against a **live, teacher-authored curriculum** (communication, learning, and questioning — generated by the model, never hardcoded), and grades Unison's answers blind against the teacher's, graduating each topic the moment Unison wins the majority.

---

## Run it

```bash
# Requirements: Python 3.9+, Ollama (gemma-4-31b), discord.py, numpy
ollama pull gemma-4-31b:latest
echo "DISCORD_TOKEN=your_token_here" > .env

cd UnisonAI
PYTHONPATH=. python3 omni/discord_bot.py
```

The full architecture — every module, mechanism, forced constant, and a candid list of what works, what is stubbed, and what is known-broken — is documented, in depth, in **[`omni/README.md`](omni/README.md)**. Nothing is hidden.

---

## Status

UnisonAI is under active development. It learns by absorbing clean, teacher-corrected experience — fluency grows with what it is taught, driven by learning alone rather than any parameter to turn, and the improvement between sessions is the finding. Its live conversational generation is **counted composition** — pair retrieval with relexicalization, the taught loop, and the forced deep-context cascade — never verbatim recall, from any store. Its Learning Law is measured working in believed units (**17% → 75% judged-good after one round of teaching, stable across four further rounds**; the calibrated-judge record is in [`omni/README.md`](omni/README.md)), and its conversational quality grows by data and by teaching — both instrumented, with the gains between releases as the finding. The integer-Walsh "fold eye" and "fold ear" of the paper are built (top-32 Parseval-certified coefficient bands); hearing also falls back to Whisper transcription. The separate exact-fractional per-character/word predictor is the arm that won the held-out cross-entropy benchmark above.

## The corpus

UnisonAI does not stand on its own — it stands on, and exists to prove, the theory it derives from:

> **The Smithian Fold Theory of Everything** — one machine-checked, self-proven theorem (*there is no nothing*), zero axioms, zero free parameters, the constants of nature derived and machine-verified.
> **→ https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything**

---

<div align="center">

*Built by **Maria Smith**, Ernos Labs.*
*The net is law — recovered, and made to run.*

</div>
