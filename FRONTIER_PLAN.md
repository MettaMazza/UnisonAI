# The Frontier Plan — generalised free generation from exact counts

**The goal (Maria, 2026-07-16):** the proofs live at generalised, LLM-style generation —
fluent, coherent, on-topic free composition of novel surface — produced by the counted,
zero-parameter, forced-constant substrate. Prediction is already won (char 1.289 vs 1.888;
word 3.19 vs 3.43 against the gradient twin); the learning law is measured working
(17%→92% after one teaching round); the instruments are calibrated. What remains is making
the substrate SPEAK the way it already PREDICTS.

**Method discipline:** every ingredient below is an established, tested piece of
mathematical computer science, translated 1-1 into exact counts and forced constants —
never invented novelty ([[TRANSLATION_PLAN.md]]). Every stage is gated by the calibrated
judge AND held-out cross-entropy (generation must improve without prediction regressing).
No proxy ever steers and scores the same thing. Negatives are implementation verdicts:
fix the translation, never lower the goal.

---

## What buys an LLM its coherent generation — decomposed and translated

| Ingredient of LLM coherence | Established mathematics | Counted/forced translation |
|---|---|---|
| Long-context conditioning P(next \| full context) | **kNN-LM** (Khandelwal et al. 2020): interpolate the local LM with a nearest-neighbour memory over contexts — retrieval *becomes* generation | **Kin-context mixing**: retrieve stored contexts similar by counted kinship (content overlap + kin at 1/2, recency 2⁻ᵃᵍᵉ); mix their observed continuations by similarity × the cascade 2^L into the local fluency distribution — the Rung-5e mixing law applied AT GENERATION, which the paper already registers as the next build step. Zero parameters. |
| Long-range topic coherence (attention across sentences) | **Trigger-pair / cache LMs** (Rosenfeld 1996; Kuhn & De Mori): long-distance word pairs and a document cache raise continuation probability | The **PPMI coupling graph as a topic prior**: the conversation's accumulated content set boosts coupled continuations. (Note: as a *generation prior* this is the established trigger table; as a *judge* it stays banned — that was the retracted critic.) |
| Implicit discourse planning | **Plan → realize NLG** (content planning; constrained decoding) | **Skeleton from kin responses**: what humans said in similar contexts supplies the dialogue-act + content-word PLAN only (never the surface); the kin-context generator realizes it, constrained to hit the plan's content words. Non-verbatim by construction — the surface is generated. |
| Local grammatical fluency | n-gram backoff (Katz/KN lineage) | already held: the fluency store L1–4 + exact shares + Laplace floor + 2^L backoff — the CE-winning substrate |
| Whole learned constructions | phrase/chunk memory | unit-capacity PHRASE emission over deep unique contexts (in the design; wire into the new mixer) |
| RLHF-style improvement | **STaR / self-training with earned retention** | generate N candidates → judge/teacher selects → winners reinforce pairs/couplings; unverified output discarded (the paper's generation-closure law) — the mechanism whose taught form already measured 17%→92% |
| Scale | the scaling laws (volume → capability) | corpus volume → deeper suffixes → longer coherent spans; "conversational fluency is a volume phenomenon with its growth rate measured hourly" (the paper) — /scrape + full ultrachat + broad register |

---

## Stages (each gated; parallel-arm rule at the bottom)

**F0 — Instrument the target.**
A free-generation harness with three gates: (a) judged GOOD-rate of the *pure free arm* (no
response-serving allowed — the generator may only consume retrieved material as
distributions/plans, never as surface units); (b) held-out CE unchanged or better than the
committed baselines; (c) a multi-sentence coherence probe (judged, length-graded).
Deliverable: `train_eval/gen_free_harness.py` + committed baseline row.

**F1 — Kin-context mixing (the pivotal bridge; kNN-LM translated).**
Generalize unit-capacity selection from exact-deepest-suffix to kin contexts. Retrieval
stops being the *surface* and becomes the *distribution*: similar held contexts contribute
their continuations, weighted by counted similarity × 2^L, mixed with local fluency.
Expected first honest result: locally fluent multi-clause output that stays near the topic.
Gate: free-arm judged rate > the current n-gram free baseline (≈0), CE non-regressing.

**F2 — Topic cache (long-range coherence).**
The trigger translation: conversation content set (weighted 2⁻ᵃᵍᵉ) multiplies coupled
continuations' shares. Gate: the multi-sentence probe — topic held across 3+ sentences.

**F3 — Plan → realize.**
The pair index (649k) serves PLANS (act + content words from kin responses), the F1+F2
generator realizes them under constrained decoding. This is where free generation should
first *answer like conversation* rather than merely flow. Gate: free-arm judged rate
approaching the retrieval arm's on the opener set.

**F4 — Generation closure (the Learning Law at the generator).**
Best-of-N with judge/teacher selection; earned retention reinforces what won. The taught
curve (17%→92%) predicts this loop's shape. Gate: a climbing judged curve on the free arm.

**F5 — Volume.**
Scale the held corpus (full ultrachat, more vetted dialogue, broad register). Measure the
scaling curve: judged rate and CE vs corpus size — the counted engine's own scaling law,
committed like every other trajectory.

**F6 — The proofs.**
(a) CE: already held at both scales — keep it held. (b) Conversation: the judged
head-to-head vs the 35B, trajectory from the honest 15% floor, opponent-integrity enforced.
(c) The mechanism as theory: the ingredients above are engine constructs on forced
primitives (the kin law, the cascade, the capacity lock, the halving) — whether the
composition is *forced* through the corpus as .ep steps is Maria's derivation work, and the
paper's claims stay scoped to what is forced vs what is engineering until then.
(d) The architecture paper's §8.2 updated to the free generator the day it wins its gate.

## The parallel-arm rule
The retrieval surface (today's 12–25% + taught seats) **keeps serving live** while the free
arm grows behind the same judge. The arms are measured side by side; the free arm takes
over the moment its judged rate crosses the serving arm's — a measured handover, never a
big-bang rewrite. Regressions get recorded, reverted, and kept in the trajectory table like
every honest number so far.
