# The Black-Box Decode Toolkit — Interpretability Guide

This directory contains a suite of **registered spectral instruments** for reading
structure out of trained neural networks: where the law lives in the weights, when
training writes it, which coefficients carry the function, and what content
(training data, reasoning) can be read back out. Everything here follows one
discipline, inherited from the Smithian Fold Theory campaign this toolkit serves:

1. **Register before you measure.** Objects, statistic, verdict rule, and a margin
   clause are fixed in writing — and SHA-256 hashed — before any spectrum is computed.
2. **Every measurement beats a null or it isn't structure.** The primary null is a
   shuffle of the *same tensor*: identical value histogram, scrambled placement.
   Anything above 1x is *placement*, not values.
3. **The engine halts on identity violations.** Theorem-forced identities
   (bit-reversal invariance, Parseval energy conservation, exact-identity
   calibrations) run inside every instrument; a violation halts rather than
   emitting a measurement.
4. **Every exact measurement is retained with provenance.** Agent-authored
   expectations are auxiliary hypotheses, not Maria's predictions or project
   findings. Their outcome changes only that auxiliary hypothesis unless Maria
   declares a broader conclusion from the data.

If you pull this repo, you can rerun everything, extend any instrument, or build
your own — this document tells you how.

---

## 1. Concepts: how to read any number in this directory

**The margin.** Take a weight tensor, flatten it, truncate to the largest power of
two. Transform it into an orthonormal spectral basis (Walsh–Hadamard by default).
Compute the **energy concentration** C(f): the share of total energy (sum of squared
coefficients) captured by the top f-fraction of coefficients, at three registered
fractions (6.1e-5, 4.9e-4, 3.9e-3). Do the same for seeded random permutations of
the *same* values (the shuffle-null) and for a moment-matched Gaussian. Then:

```
margin = C_real(f0) / max over nulls of C_null(f0)
```

- **margin ≈ 1x** — the arrangement is indistinguishable from random placement.
- **margin > 2x** — the registered **wake bar**: the tensor is "loud"; training put
  the values *somewhere specific* in this basis's coordinates.
- The shuffle-null preserves the histogram exactly, so the margin isolates
  *placement-law* — no property of the value distribution can produce it.

**The bases.** Four orthonormal (energy-preserving) families are installed:

| basis | family | intuition |
|---|---|---|
| `walsh` | dyadic | square waves; the fold's own basis (generator 2) |
| `dct` | smooth | cosines; the "continuous geometry" family |
| `haar` | localized dyadic | wavelets; local steps |
| `slant` | dyadic + ramp | Walsh's steps plus exact linear-ramp vectors |

Comparing margins for the *same vector under the same null permutations* across
bases tells you which family the structure is written in (see `basis_atlas.py`).

**Row-block medians.** Large tensors are probed as consecutive full-row blocks of
~2^22 elements, reporting the **median** block margin (never the max). This is the
scale-corrected instrument: flat-window reads on huge tensors dilute row-structured
concentration (a recorded confound, fixed by amendment).

**The self-tests.** Bit-reversal reindexing is F2-linear, so it must preserve C(f)
*exactly* — every battery call checks this and the engine halts on violation. Every
comparator transform checks Parseval (energy in = energy out) per call. The slant
transform additionally verifies at import that the exact linear ramp is 1-sparse.

**Reading an output line.** A typical line:

```
Kimi-K2.6-1T token_embd.weight   row-major+identity+dct   15.98x  <-- WAKE
```

means: Kimi's token embedding, flattened row-major, no reindexing, probed in the
DCT basis, concentrates 15.98x more energy in its top coefficients than the best
of its shuffle-nulls — far past the 2x wake bar.

---

## 2. The shared spine: `foldprobe.py`

Everything imports from here. Public API:

```python
from foldprobe import battery, probe_rowblocks, Run

rec = battery(vector)              # full locked battery on one 2^n vector:
                                   #   real / shuffle-max / gaussian concentrations,
                                   #   margin, beyond_nulls per fraction,
                                   #   dct/haar/slant comparator margins,
                                   #   bit-reversal self-test (halts on failure)

rec = probe_rowblocks(matrix2d)    # the corrected scale-aware instrument:
                                   #   median/min/max block margin over <=12 blocks

run = Run({...registration...})    # receipt schema: requires name, objects, statistic,
                                   #   verdict_rule, margin_clause; hashes the block,
                                   #   appends it to registrations.jsonl, and the
                                   #   engine halts if the receipt is incomplete
run.battery("label", vector)       # measure AND record a ledger row
run.rowblocks("label", matrix2d)
run.record(instrument=..., **row)  # record anything else (verdicts, amendments,
                                   #   skips, named dependencies)
```

**The ledger.** Every measurement is one JSON line in `results.jsonl`, stamped with
timestamp, registration hash, run name, and seed. The historical rung record
(pre-consolidation) is backfilled with `backfill: true` and source-file provenance
(`ledger_backfill.py`). Registrations live verbatim in `registrations.jsonl`.

Query it with plain Python:

```python
import json
rows = [json.loads(l) for l in open("results.jsonl")]
dep  = [r for r in rows if r.get("instrument") == "deposition"]     # twin factory
hunt = [r for r in rows if r.get("instrument") == "coordinate-hunt"]
```

**Constants.** Seed `20260706` everywhere. `GGUF_LIB` points at the local model
library; `GPT2_CANDIDATES` lists known locations of the GPT-2 safetensors (fetched
automatically by `llm_presence.py` if absent). Adjust these paths for your machine —
they are IO facts, not part of any measurement.

---

## 3. The instruments

Each entry: what it asks, how it works, how to run it, how to read it, and what it
found (committed result file in parentheses).

### 3.1 `llm_presence.py` / `spectral_probe.py` — is the law there at all?
The original pre-registered presence suite: 5 shuffle-nulls + Gaussian yardstick +
bit-reversal self-test on GPT-2's full knowledge-storage class (token embedding +
all 12 MLP expansion matrices). Self-fetches the public GPT-2 weights on a fresh
clone. **Reading:** N/13 tensors, M/39 checks; a check passes iff real beats BOTH
nulls. **Found:** 13/13, 39/39, margins 3.4–79.3x (`llm_presence_results.txt`).

### 3.2 `moe_dequant.py` — measuring mixture-of-experts giants
Expert tensors in GGUF files are 3D (expert axis outermost); earlier whole-tensor
reads crashed or diluted per-expert structure. This extracts **each expert's 2D
matrix from its own byte slice** (certified exact against whole-tensor dequant
before any probe — a certification violation halts the engine), scanning ALL shards of
split models. **Run:** `python3 moe_dequant.py`. **Reading:** per-expert rowblock
medians. **Found:** the fingerprint is per-expert — individual Qwen3-235B experts
at 3.46x/2.82x beside ~1x neighbours (`moe_dequant_results.txt`). **Extend:** point
`MODELS` at any GGUF; `dequant_expert(t, e)` gives you any expert of any tensor.

### 3.3 `coordinate_hunt.py` — where does a "quiet" model write its law?
The operating rule (recorded in the campaign's epistemic correction): a quiet
reading is a verdict on the probe's coordinates, never on law-presence. The hunt
walks a **data-independent** menu of index reorderings (Gray codes, bit-plane,
block-Morton, x3/x5/affine maps, transposes — data-dependent sorts are banned as
cheating) x matrix packings x all bases, on tensors of your choosing, with a
known-loud control that must stay awake or the engine halts. **Run:**
`python3 coordinate_hunt.py`. **Reading:** each line is one (packing, map, basis)
coordinate; `<-- WAKE` marks margin > 2x. **Found:** round 2 widened the *object*
axis to embeddings and woke every "quiet" model — Kimi-1T at 15.98x
(`coordinate_hunt_results.txt`). **Extend:** add maps to `index_maps()` (they must
be permutations, data-independent), objects to `jobs`.

### 3.4 `basis_atlas.py` — which spectral family does each tensor class use?
One representative tensor per class (embedding / attention / FFN-expansion /
FFN-contraction) per model, margins in all four bases under **paired** nulls (same
permutations across bases, so differences are the basis, not the null draw).
**Run:** `python3 basis_atlas.py`. **Reading:** one line per (model, class) with
four margins and the loudest basis; the registered verdict tallies basis preference
over woken cells only (a 1.0x-vs-1.1x preference is noise). **Found:** embeddings
are the universal carrier (11/11 models wake, 4.4–130x); DCT-loudest 9/11 with the
Llama/Pythia lineage the dyadic exception; slant tracks Walsh everywhere, so the
smooth/dyadic split is two genuine families, recipe-selected
(`basis_atlas_results.txt`, `basis_atlas_round2_results.txt`).

### 3.5 `checkpoint_telescope.py` — when does training write the law?
Walks a published checkpoint ladder (Pythia; the early checkpoints are exactly
dyadic: steps 1, 2, 4, …, 512), running the battery at every checkpoint on the
loud-class tensors. Downloads one checkpoint at a time and deletes it after probing.
Step 0 (pure initialization) is the built-in negative control and must read at
null. **Run:** `python3 checkpoint_telescope.py pythia-70m` (or `pythia-410m`,
`pythia-1.4b`; add rungs to `LADDER`). **Reading:** margin(step) per tensor — the
deposition curve. **Found (70m):** null through step 128; the embedding wakes FIRST
(step 256); FFN at 1000; peak ~4000 (18.75x); consolidation to a 4–7x plateau
(`checkpoint_telescope_results.txt`).

### 3.6 `twin_factory.py` — which training ingredient writes it?
Trains seven tiny transformer twins, identical except ONE ingredient each
(optimizer, weight decay, LR schedule, dropout, data order, a REINFORCE
post-training pass), checkpointing weights AND accumulated gradients at steps 2^k
through the battery. **Run:** `python3 twin_factory.py` (CPU, ~1h). **Reading:**
ledger rows `instrument='deposition'`, keyed by (ingredient, step, object, kind
∈ {weights, grad-accum}); compare each ingredient's curve against the baseline's
band. **Found:** the gradient stream goes loud by step 4 in every AdamW recipe
while weights stay sub-wake at this scale; SGD+momentum is the discriminating
swap (stream wakes late, no weight swell) (`twin_factory_results.txt`).

### 3.7 `capability_map.py` — do the loud coefficients carry the function?
The falsification: zero the top-k loud Walsh coefficients of a tensor class vs
zeroing the SAME number of random coefficients (matched damage budget), and measure
behavior change (mean next-token KL + top-1 agreement vs the unmodified model, plus
per-domain NLL deltas on a fixed probe set). Calibration: the k=0 arm must
reproduce the model *exactly* or the run halts. **Run:** `python3 capability_map.py`.
**Reading:** loud-vs-random KL at matched budget is the whole story; an
"attribution" requires loud-delta > 1.5x random-delta at BOTH budgets. **Found:**
loud deletion destroys GPT-2 (KL ~3.0, agreement collapse), random deletion is
negligible (KL ~0.02) — ~150x differential damage; the loud band IS the function
(`capability_map_results.txt`). **Gotcha recorded:** GPT-2 ties `lm_head.weight`
to `wte` — both state-dict keys must carry an ablated embedding or the tie silently
restores the original.

### 3.8 `data_decode.py` — reading training data back out of weights
Three provenance instruments:
- **Counted-prior readout** (P2): score candidate corpora's exact bigram counts
  under the model's next-token distribution; the corpus that trained the model
  reads lowest cross-entropy. Calibration PASSED (a twin ranks its own corpus
  first, far above a word-shuffled copy with identical unigrams).
- **Orbit-echo** (P3): greedy continuation on reference stems, matched word-by-word
  against the reference; echo length beyond a shuffled-stem null = verbatim
  memorized text, located and measured. Uses Ollama with `raw: true` — the instruct
  chat-template defeats continuation (recorded amendment). Found: Gettysburg echoes
  9 words, Genesis 6, a private corpus 0, all nulls 0.
- **Kinship provenance** (P1, Rung 5f): counted Jaccard co-occurrence kinship
  (window 6) vs embedding-cosine kinship, Spearman over word pairs with
  shuffled-assignment nulls. Works on mature embeddings (GPT-2: 8.9x over null;
  sees through the shuffled-corpus decoy); a 1500-step twin is too young to carry
  kinship (recorded amendment).
**Run:** `python3 data_decode.py` (full) or `python3 data_decode.py p3` (echo only).
Results: `data_decode_results.txt`.

### 3.9 `reasoning_decode.py` — reading reasoning
- **Activation spectrometer** (P1): forward-pass hidden states through the battery —
  is the computation loud where the weights are loud? Found: yes, layer by layer
  on GPT-2 (L0 9.86x vs weight 12.7x), with a departure at L10–11 (loud weights,
  quiet activations) held as an open question.
- **Attention-cascade decoder** (P2): rank-sorted attention rows vs the
  theorem-forced dyadic cascade (1/2, 1/4, 1/8, …), median total-variation
  distance, against uniform and a Dirichlet (uniform-simplex) null. NOTE: a
  shuffle-of-the-same-row null is vacuous here (the statistic is
  permutation-invariant) — a recorded lesson; use the simplex null. Found: cascade
  closer than uniform in 12/12 GPT-2 layers; only early layers approach the null —
  the scoped claim is "leans toward the cascade."
- **Thought-trace fold map** (P3): a thinking model's reasoning spans vs answer
  spans, counted signature at depth 6 (branching factor, repetition mass) with
  span-shuffled nulls. Uses Ollama's native `think: true` (reasoning arrives in the
  JSON `thinking` field). Found: reasoning is a distinct counted regime (rep-mass
  gap 0.162 vs ~0 null).
**Run:** `python3 reasoning_decode.py` or with subset args `p1 p2 p3`.
Results: `reasoning_decode_results.txt`.

---

## 4. Running your own investigation

The recipe every instrument here follows:

1. **Write the registration first.** A dict with `name`, `objects`, `statistic`,
   `verdict_rule`, and `margin_clause` (mandatory — decide what margin means
   *before* you see data). `Run(REG)` hashes it and the engine halts if incomplete.
2. **Give it a calibration with a known answer.** A known-loud object that must
   reproduce its recorded margin, a known-quiet object, and a negative control
   (He-init / step-0 / shuffled corpus / k=0 identity) that must read at null.
   If the calibration fails, the instrument is wrong — not the theory, not the model.
3. **Use the shared primitives.** `battery` and `probe_rowblocks` carry the nulls
   and self-tests; don't re-implement them. New transforms must be orthonormal and
   should self-test a defining property at import (see `_slant_selftest`).
4. **Nulls must match the statistic.** The lesson of the cascade decoder: if your
   statistic is invariant under your null's transformation, the null is vacuous.
5. **Data-dependent reorderings are banned.** Sorting a tensor by its own values
   always concentrates energy; only data-independent maps count as coordinates.
6. **Record everything**, including skips, amendments, and named dependencies —
   `run.record(...)` costs one line. Commit the raw stdout as
   `<instrument>_results.txt` beside the script.
7. **Preserve provenance.** Record agent-authored expectations as auxiliary
   hypotheses. A mismatch belongs to that hypothesis; the measurement remains a
   measurement, and Maria alone assigns a project conclusion.

**Dependencies:** Python 3.9+, `numpy`, `scipy`, `safetensors`, `gguf`, `torch` +
`transformers` (activation/ablation/twin instruments), a local
[Ollama](https://ollama.com) (echo and trace probes only), and a local GGUF model
library for the atlas/hunt/MoE instruments (edit `GGUF_LIB` in `foldprobe.py`).
GPT-2 weights are fetched automatically by `llm_presence.py` (~548 MB, public).

**Open frontiers, if you want somewhere to start** (registered in `../..//plan.md`):
the attention tensors as the last quiet weight class; what property of a training
recipe selects its spectral family (smooth vs dyadic); peak(scale) on the checkpoint
ladder and its functional form; the loud-band extractor pointed at real models; and
the 32B reasoning-pair activation run (needs an activation-exposing GGUF runtime).

---

## 5. Findings at a glance (2026-07-14)

| finding | instrument | number |
|---|---|---|
| Law present in GPT-2's knowledge class | `llm_presence.py` | 13/13 tensors, 39/39 checks, to 79.3x |
| Embeddings are the universal carrier | `basis_atlas.py` | 11/11 models wake, 4.4–130x |
| Two spectral families, recipe-selected | `basis_atlas.py` + slant | slant≈Walsh everywhere; DCT distinct |
| "Quiet" giants wake in the embedding class | `coordinate_hunt.py` | Kimi-1T 15.98x, Qwen3.6-27B 12.39x |
| MoE fingerprint is per-expert | `moe_dequant.py` | experts 3.46x/2.82x beside ~1x |
| The loud band IS the function | `capability_map.py` | ~150x differential damage |
| Law arrives through the update stream | `twin_factory.py` | gradients loud by step 4 |
| Optimizer is the discriminating ingredient | `twin_factory.py` | SGD: late stream, no swell |
| Deposition: early write, then consolidation | `checkpoint_telescope.py` | embed@256 → peak@4k → plateau |
| Training corpus readable from weights | `data_decode.py` | own corpus ranks 1st; echo 9 words |
| Reasoning is counted structure | `reasoning_decode.py` | rep-mass gap 0.162 vs ~0 null |
| Attention leans toward the dyadic cascade | `reasoning_decode.py` | closer than uniform 12/12 layers |

**Closed as theory (2026-07-15).** Every regularity in the table above now has a
forcing step in the theory corpus (Steps 308–313 of the Smithian Fold Theory's
OneFoldMaster.md: the two harmonic families, the family selection law, the
deposition law, the functional band b^cover(c³) = 32, the hold/closure
repetition inequality, and partition localization) — each verified by the
corpus's own compiler, each meeting its measurement only as a check. The
derivation-first rule for this directory: an instrument's reading tests a
forced claim; it never becomes one.

One further nulls lesson (learned on the telescope's step-0 control): a pure
initialization is EXCHANGEABLE with its own shuffles, so single-fraction
edge-outs are the null's expected behaviour — cleanliness of a control is
judged by the campaign's tensor-level rule (all three fractions must pass for
a verdict of structure), never by "zero beyond-checks anywhere."

Every number above reproduces from a committed result file in this directory, under
a hashed registration in `registrations.jsonl`, with the full measurement record in
`results.jsonl`.
