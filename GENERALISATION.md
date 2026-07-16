# Baseline Generalisation — the counted grokking, predicted, run, and measured

**The claim (from the forced laws, before the data):** gradient "grokking" — memorize first,
generalise later, suddenly — is an observed mystery in trained networks. In the counted
engine it is a **predicted crossing**: a novel prompt is answered from held material **iff
its coupling to something held reaches the lock 1/2** (the binding law, reused). Every
taught meaning covers a kin-NEIGHBORHOOD, not a point; as held meanings accumulate, the
share of novel-prompt-space bound at the lock rises; generalisation onset is the measured
moment that coverage crosses. **How** = kin-neighborhood coverage. **When** = the
lock-crossing, read off a counted predictor *before* the quality curve moves. **Why** = the
lock itself — the same forced 1/2 that gates attention, coherence, graduation, and binding.

## The apparatus, v2 — epoch training, translated (`train_eval/generalisation_epochs.py`)

The fixed-round telescope below was the first instrument; it did its job — one full dyadic
ladder proved the harness end to end and exposed the routing gap (paraphrases never reached
taught meanings because binding counted literal word identity; kin now carries it — see
"kin-carried binding" in the README). But a fixed teach set is an arbitrary total, and
arbitrary totals are not how learning systems are trained. The current apparatus is the
**1-1 translation of the standard epoch training loop**, continuous until the criterion:

| Epoch training | The counted translation |
|---|---|
| training dataset, streamed | the pair corpus's real prompts (counted hygiene filters, deduped, probe-excluded), deterministic hash order — reproducible, no RNG |
| batch | **32 = the band, 2^(b+c)** (forced constant, reused) |
| epoch | **32 batches = band² prompts** (structural, not a knob) |
| per-item loss | **the binding deficit**: kin-carried binding to held meanings vs the lock 1/2 — the same forced quantity that routes serving. Cheap, counted, computed for every item; no judge runs per training item, exactly as real training pays for evaluation only on validation |
| loss-gated update | teach only where binding < the lock; the update is **deposition** (the meaning held with ≥ b = 2 expressions), not a gradient step — the zero-parameter analogue of the weight update |
| held-out validation per epoch | frozen band-sized near/far probe sets under the **judge pool** (two independent calibrated models, concurrent, both must agree GOOD), full transcripts, binding-stratified (GOOD-given-bound vs GOOD-given-unbound — the hypothesis read directly) |
| early stopping | near AND far transfer ≥ the lock for **b = 2 consecutive epochs** — then the run stops and reports; it declares nothing |
| checkpoints / resume | stream cursor (`logs/epoch_state.json`), store backups, ledgers `logs/generalisation_epochs.jsonl` + `logs/epoch_teach.jsonl` (every teaching event: phrasings offered, drift rejected, expressions held) |

Where a transformer's generalisation appears in opaque weight space, here it must appear as
**kin-coverage of held meanings crossing the lock** — which is why onset is predictable in
this engine and only observable-after-the-fact in gradient training.

## The first apparatus (`train_eval/grokking_run.py`)

- **Autonomous teaching**: rounds over a fixed 48-prompt teach set — the engine answers,
  the calibrated judge verdicts (the believed scoreboard; gate passed 10/10|10/10), and on
  a failure the teacher supplies the correction as **three phrasings** (re-expression law:
  ≥ b held expressions; nothing ever served verbatim).
- **Dyadic checkpoints** — rounds 1, 2, 4, 8, 16: the same 2^k telescope convention the
  decode campaign used to read Pythia's training deposition, now reading this engine's
  learning deposition. The two curves are methodological mirrors: gradient deposition read
  from public checkpoints; counted deposition read from teaching rounds.
- **Three frozen curves** (probes never taught, never feedback — read-only):
  1. **Memorization** — judged GOOD on the taught prompts themselves (re-expressed service)
  2. **Near-transfer** — judged GOOD on PARAPHRASES of taught topics (kin must carry the
     meaning across wording: the first generalisation)
  3. **Far-transfer** — judged GOOD on novel topics (coverage must reach them)
- **The predictor**, measured beside the curves: the share of probe prompts whose best
  binding (taught-overlap or pair-prompt similarity) ≥ the lock. The hypothesis is falsifiable
  in one line: **transfer curves move when and only when the binding share crosses.**
- Ledger: `logs/grokking.jsonl`, one row per checkpoint, timestamped.

## The predicted signature

Memorization saturates early (the taught loop is measured at 17% → 75% after one round).
Near-transfer lags it, then climbs as taught kin-neighborhoods overlap the paraphrases.
Far-transfer climbs last, gated by coverage breadth. The binding-share predictor leads each
transfer curve. If the curves move without the predictor, or the predictor crosses without
the curves, the hypothesis is wrong — and that is reportable either way.

## Where the results fold

- **Architecture paper (§9.1 / a new §9.2)**: the three curves + the predictor, the crossing
  round, and the one-line law — generalisation onset as the lock-crossing of counted
  coverage; grokking as a forced threshold rather than an emergent mystery.
- **Decode paper (deposition section)**: the learning-side dyadic deposition curve beside
  Pythia's training-side curve — the same telescope, both paradigms.
- **Corpus (candidate step, drafted after the curves land)**: `generalisation_onset.ep` —
  onset = binding coverage at the lock (a reuse-forcing of the binding law over prompt
  space), with the measured crossing at the Measured wall.

## The gate

Results are local until Maria reviews them; papers fold and mint on her word (the publish
gate). Manual sessions begin from the generalised store the run leaves behind.
