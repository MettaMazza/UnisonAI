# Unison Omni AI — Complete Technical Documentation

> **Author:** Maria Smith (Ernos Labs)
> **Theory:** Smithian Fold Theory (SFT)
> **Status:** Active development
> **This document tracks the *current* state of the `omni/` codebase.** Every mechanism below is cross-referenced to the file and symbol that implements it. Open bugs and what's being built next are in [Engineering Notes & Roadmap](#engineering-notes--roadmap).

---

## Table of Contents

1. [What Is Unison](#what-is-unison)
2. [Core Architectural Principles](#core-architectural-principles)
3. [File Structure](#file-structure)
4. [The Memory Engine (`memory.py`)](#the-memory-engine-memorypy)
5. [How Unison Generates a Response](#how-unison-generates-a-response)
6. [Utterance Segmentation (`segmentation.py`)](#utterance-segmentation-segmentationpy)
7. [The Word Tier (`word_engine.py`)](#the-word-tier-word_enginepy)
8. [Repetition Guard & Self-Feedback](#repetition-guard--self-feedback)
9. [The Learning Loop](#the-learning-loop)
10. [The Teacher (`teacher_scaffold.py`)](#the-teacher-teacher_scaffoldpy)
11. [The Live Curriculum (`curriculum_communication.py`)](#the-live-curriculum-curriculum_communicationpy)
12. [Distillation & Self-Play (`distill.py`)](#distillation--self-play-distillpy)
13. [Sessions & Identity](#sessions--identity)
14. [Modalities: Sight, Hearing, Voice, Image](#modalities-sight-hearing-voice-image)
15. [Tools](#tools)
16. [Persistent Memory & Logging](#persistent-memory--logging)
17. [Discord Commands & Buttons](#discord-commands--buttons)
18. [Prerequisites, Setup & Running](#prerequisites-setup--running)
19. [Glossary of Special Tokens](#glossary-of-special-tokens)
20. [Engineering Notes & Roadmap](#engineering-notes--roadmap)

---

## What Is Unison

Unison is a **zero-parameter, exact-fractional, per-character geometric AI engine** built entirely on the Smithian Fold Theory (SFT). It is fundamentally different from a trained language model:

- **Zero trained parameters.** No weights, no gradients, no backpropagation, no fitting.
- **Zero floating point in the prediction path.** Every probability is an exact `fractions.Fraction`.
- **Per-character substrate.** Every character (including control codes) is its own token. No BPE, no vocabulary.
- **Pure geometric memory.** Unison stores exact character sequences ("orbits") — everything read, told, or thought — as deterministically-addressed exact counts, keyed per user, written once and kept forever.
- **Generation never replays memory verbatim — from any store, including taught corrections.** Conversational replies are **composed**: counted pair retrieval (exact BM25 with question-similarity over role-structured conversational pairs) under the **forced deep-context cascade** (`contextual_integration.ep`), with relexicalization rebinding live entities and a structural guard forbidding any emitted reply equalling any stored string. Corrections are internalized as multiple expressions of one meaning and served only re-expressed. (The exact-fractional per-character/word predictor is a separate substrate; it is the arm that won the held-out cross-entropy benchmark below.)

Unison runs as a **Discord bot** (`omni/discord_bot.py`). It learns from two sources:
1. **Direct conversation** — every message and the derived correction feed the learning loop.
2. **A local teacher model** — Gemma-4-31b via Ollama — which supplies ideal responses and a live, self-authored curriculum that get baked into the geometric graph.

The teacher is **scaffolding, not the engine** — and, critically, **a throughput choice, not a dependency**. Every role the teacher fills is a human role that was automated for speed: a human tutor typing corrections feeds the *identical* learning pipeline (`add_taught` is source-agnostic — it neither knows nor cares whether a correction came from a model or a person); the 👍/👎 buttons *are* the human-judge path, and the LLM judge only proxies them for throughput; curriculum prompts can come from a person or from real conversation. The architecture's laws — counted writes, the lock, graduation, re-expression — reference no property of the teacher anywhere, and the paper's §8.8 *measures* learning with **zero models in the loop** (human closure, re-recognition at share 1.00). A heuristic pipeline over data could fill the same seat. An LLM teacher was used because a human cannot teach 24/7 and this engine is one proof inside a much larger solo project — the most efficient scaffold available, with a **measured exit** (graduation at the counted lock) built into the design.

---

## Core Architectural Principles

### The SFT constants (`core.py`)

Derived from the theory, not tuned. They cannot be changed:

```python
GEN_B   = 2   # Binary generator — the fold's period beyond the One
GEN_C   = 3   # Colour generator
CTX_MAX = 6   # GEN_B × GEN_C — the maximum structural context depth
```

### The fold map & domain law (`core.py`, `FoldValue`)

Every value lives strictly in `(0, 1]`. Zero and negatives are forbidden; nothing exceeds the One. The fold is `x → 2x mod 1` (with `0` snapping to `1`, the invariant One). Any violation calls `halt_violation()`, which logs `CRITICAL` and hard-exits (`sys.exit(1)`) — invalid operations cannot be silently swallowed.

### Exact rational arithmetic

All probabilities are `fractions.Fraction`. When Unison computes that a character has probability `3/7`, that is stored and compared as the exact fraction, never `0.4285…`.

### The Walsh–Hadamard transform (`core.py`)

The dyadic substrate uses an integer Fast Walsh–Hadamard Transform (`fwht_1d`, `fwht_2d`, `ifwht_2d`) that:
- requires power-of-two dimensions (enforced by `halt_violation`),
- uses pure integer arithmetic,
- **certifies Parseval's identity after every transform** (`certify_parseval`): `Σx² · N = ΣX²`, exact integers, halting on mismatch.

The FWHT underpins the perceptual encoders (image/audio) in `modalities.py`.

---

## File Structure

```
omni/
├── core.py                    # SFT constants, FoldValue domain law, integer FWHT + Parseval
├── memory.py                  # SynapticGraph, ActiveLedger, unit_capacity_selection, predict_next
├── segmentation.py            # Utterance segmentation (structural + counted BoundaryStore)
├── pair_retrieval.py          # THE LIVE GENERATOR: counted pair retrieval (BM25 x question-similarity),
│                              #   relexicalization, taught re-expression, the never-verbatim guard
├── free_gen.py                # Historical agent-authored development arm; not the native route
├── native_transformer.py      # Native 1:1 counted causal-attention transformer route
├── context_state.py           # The integrated context state (contextual_integration.ep, halt-verified)
├── word_engine.py             # Counted kinship/fluency substrate; retired fallback entry points
├── session.py                 # Session (working memory) + SessionManager (per-user sessions)
├── identity.py                # UnisonIdentity (response provenance), UserFingerprint (per-user profiles)
├── teacher_scaffold.py        # LocalTeacher, detect_context_window, GraduationLedger
├── curriculum_communication.py# LIVE teacher-generated communication/learning/questioning curriculum
├── distill.py                 # ModelPool, ModelSpec, DistillationEngine, SFT-corpus curricula
├── discord_bot.py             # SFTDiscordClient — the bot, on_message, /auto loop, confusion reflex
├── observer.py                # ObserverTeacher — degradation/stagnation monitor
├── modalities.py              # The fold eye + fold ear: integer Walsh top-32 bands, Parseval-certified
├── voice.py                   # KokoroSpeaker (TTS), WhisperListener (STT)
├── flux_gen.py                # FluxGenerator (image generation for the vision curriculum)
├── tools.py                   # ToolOrchestrator (tool-use parsing/execution)
├── diagnostics.py             # Timer, GenerationDiagnostics, TeacherDiagnostics, MemoryDiagnostics
├── logging_config.py          # Per-subsystem rotating logs + structured learning journal
├── cli.py                     # Legacy terminal interface (word-level dyadic; not the primary path)
├── graph_memory.json          # [AUTO-GENERATED] the persistent geometric brain
├── curricula_generated.json   # [AUTO-GENERATED] the teacher-authored curriculum cache
├── boundary_store.json        # [AUTO-GENERATED] learned segment-boundary statistics
├── benchmarks/                # The empirical campaign — rung ladder, decode instruments, PROTOCOL/REPRODUCE/SOTA_TABLE
│                              #   → see benchmarks/INTERPRETABILITY.md: every instrument documented — how it
│                              #     works, how to read it, and how to run your own registered investigation
└── logs/                      # core / memory / teacher / discord / session logs + learning.jsonl
```

---

## The Memory Engine (`memory.py`)

### `SynapticGraph`

The persistent geometric graph — every exact character sequence Unison has seen or been taught, keyed per user (`ukey`).

- **`hold_orbit(sequence, ukey)`** — bank an exact orbit. O(1) deduplication via an MD5 hash set; a duplicate is a no-op.
- **`fold_orbit(sequence, ukey)`** — **CLOSE, not duplicate** (`constants/memory_abstraction.ep`, forced + ernos-verified). The old behaviour *thickened* an orbit by appending a surface copy — but duplication is the **held** regime: it re-excites the exact surface, so the content *repeats* (forced to replay verbatim). Folding in the theory's sense is the opposite — bind and close to the invariant, which does not repeat. So a good answer is kept as **one** held orbit (retrievable meaning) and **not** thickened into more surface copies; re-expression is the generator's job (`sample_next_unfold`). This method is now a deliberate no-op on the surface store: it refuses to reinforce verbatim replay.
- **`prune_orbit(sequence, ukey)`** — sever a bad trajectory (removes all exact occurrences).
- **Caches:** a pre-joined per-ukey corpus string is maintained for fast suffix search and rebuilt on mutation.
- **Persistence:** debounced atomic save (write `graph_memory.json.tmp`, then `os.replace`), 2 s after the last mutation; `force_save()` on shutdown.
- **Parallel search:** for corpora over 500 KB the suffix search fans out across CPU cores via `ProcessPoolExecutor`.

### `ActiveLedger` (and the module singleton `omni_ledger`)

Holds user-context trajectories awaiting teacher tutoring.
- `add_prompt(ukey, context_str, original_prompt=None)` — queue a prompt (used by the correction/confusion paths).
- `clear()` — empty the queue after tutoring.
- Queue attribute: `pending_prompts`.

> Note: this class is `ActiveLedger`, not `BadLedger` (an older name still referenced in some tests/docs — see [Engineering Notes](#engineering-notes--roadmap)).

### Exact-fractional primitives (the cross-entropy substrate, **not** the live generator)

These implement the exact-rational predictor used by the held-out cross-entropy campaign (`benchmarks/rung5e_*`). They are **not** on the live conversational path — the live bot never calls them to speak (that would be verbatim recall). They remain as the measured substrate and for probability calibration.

- **`unit_capacity_selection(context, graph, ukey, max_k)`** — from the longest suffix of the context (capped at 8000 chars), searches all stored orbits and returns `(suffix_depth, [continuation chars])` for the deepest non-terminal suffix.
- **`exact_rational_shares(context, graph, ukey, max_k)`** — turns those continuations into exact shares `V = count / total`, each a `FoldValue(Fraction(count, total))`.
- **`predict_next(context, graph, ukey, max_k)`** — returns `(char, suffix_depth, num_candidates)` for the highest exact share.

---

## How Unison Generates a Response

When a user message arrives (`SFTDiscordClient.on_message`):

1. **Per-character tokenization with speaker demarcation:** each segment becomes `[image chars] + ['\x02'] + list(text) + ['\x03']`.
2. **Session update:** the user turn is appended to the session's `working_context`, and that context is banked to both the global graph and the session's **episodic memory**.
3. **Utterance segmentation** (`segmentation.py`): the message is split into its distinct sub-utterances so each ask is answered on its own. A single-sentence message is one segment. See [Utterance Segmentation](#utterance-segmentation-segmentationpy).
4. **Native transformer stage** (`native_transformer.generate`, tried first): the complete prompt-token sequence enters role-bound counted embeddings and separately normalized structural and causal Q/K attention heads; key-owned value vectors, dyadic history, explicit contextual FFN KV memory, exact residual normalization and LM-head shares, greedy autoregressive decoding, and counted reward observations complete the route.
5. **RAG augmentation and retired path:** if the native route has no surface, `pair_retrieval.reply` may provide the separately disclosed RAG/response-selection surface. Its BM25 constants, fixed candidate limits, and finite lexical routing remain trace gaps. The older word/generic fallback is retired and never served.
6. **Display:** the reply renders as `[THINKING: …] Unison Response: …` — the `\x04`/`\x05` reasoning trace is shown in full. Tool JSON in the output is intercepted and executed by `ToolOrchestrator`.
7. **Self-feedback with on-the-spot correction:** the output is rated (the graduation signal in-bot; the project's believed scoreboard is the **calibration-gated independent judge** — see the Empirical Record) and, until a topic graduates, by the teacher, which receives the conversation from an **append-only `history_log`** (session.py) — written once per finished turn and trimmed by nothing, so cross-turn context (e.g. a name given earlier) always reaches it. On a *bad* rating the bad trajectory is pruned, pair-quality Laplace counts are demoted, and — **live, not deferred** — the teacher supplies the correction, Unison posts the recovery, and the correction is **held as learning material** (multiple expressions of one meaning) for re-expressed service next time. A *good* reply reinforces the pairs it was built from. The correction replaces the bad turn so the discussion stays coherent long-horizon (see [The Learning Loop](#the-learning-loop)).

---

## Utterance Segmentation (`segmentation.py`)

The generator keys on the **deepest suffix** of the context it is given. A whole multi-intent message ("hello, how are you? do you remember my name? introduce yourself…") is a *single* context, so one walk locks onto one long orbit and answers only one part — or dumps a stale long orbit that happens to share a tail. Segmentation fixes this by splitting the message and answering **each sub-utterance from its own orbit**, then composing the fragments.

Two tiers, by design:

1. **Structural (authoritative).** `_structural_segments` cuts at sentence-final punctuation (`. ? !`) outside quoted spans — so `Repeat after me: "Silly Goose"` stays intact. This is tokenizer-tier parsing (the same category as the `\x02`/`\x03` speaker demarcation), **not** authored knowledge or training seeds.
2. **Counted / learned (supportive).** `BoundaryStore` observes every message and counts the character contexts under which real boundaries occurred, persisting to `boundary_store.json`. It stays deferential (`is_confident()` is `False`) until it has enough distinct high-count contexts, after which `refine()` may **add** boundaries the punctuation missed (e.g. run-on sentences with no period). It never removes a structural boundary — structural always stands.

A single-sentence message returns one segment, so short chats behave exactly as before; the change activates on multi-intent input, where each ask is answered from its own orbit and the fragments are composed into one reply.

---

## The Word Tier (`word_engine.py`)

The current live order is native counted causal transformer, then the separately disclosed `pair_retrieval.py` RAG/response-selection surface when native generation has no surface. `word_engine.py` retains counted-kinship machinery, fluency stores, and exact-fractional substrate arms. Its historical fallback selectors are retired agent-authored code, not a component awaiting a new selector.

**Retired agent-authored path.** `compose_reply`, `structured_unfold`,
`retrieve_and_compose`, `unfold_response`, and `generic_reply` belong to the
historical word/generic fallback interpretation. They are not served, do not
define Unison generalisation, and are not a stage awaiting reconstruction.
Their old instruments remain identifiable only so historical receipts retain
their exact implementation provenance. Native generalisation is the
one-to-one constitutional translation of established causal-transformer organs
and reward-conditioned training under SFT arithmetic and constraints.

**The fold coherence critic (`coherence_score`, `constants/coherence_value.ep`)** reads a reply's content-word coupling at the synchronization lock `g_c = 1/2`. Step 322 forces that lock from the marginal gap multiplier `2(1-g)`, folds the balance to the completed One, partitions the share as `2/3` bound and `1/3` free, and closes the finite cascade `1/2 + 1/4 + 1/8 + 1/16 + 1/16 = 1` (8/8 exact checks; claim `SFT-COHERENCE-322`). The critic serves as an internal signal (fallback ranking; the in-bot graduation race). The theorem fixes its structural value; it does **not** by itself calibrate the critic as a conversational-quality scoreboard. Conversational quality is believed only from the **calibration-gated independent judge** (Empirical Record) — no signal that steers generation ever scores it.

Historical `span_quality.pkl` measurements apply only to the retired path. On
the native route, feedback is persisted per-transition good/bad observation
state under the exact Laplace preference share `(g+1)/(g+b+2)`, bound to the
served native prompt and surface provenance.

**Generalisation: counted kinship** (`NEIGH` / `kinship` / `kin_route`). `NEIGH` counts each word's immediate neighbours; `kinship(a, b)` is the Jaccard of two words' neighbour distributions — the exact-count stand-in for a trained embedding. `kin_route` scores the query against taught orbits by content-word overlap plus kin (half weight), normalised by the union — it supplies the reply's *meaning* (schema), not a span to replay:

```
score(orbit) = ( |cw ∩ ocw| + ½·|kin ∩ ocw| ) / |cw ∪ ocw|
```

**Same data, no separate training.** The couplings and the foundation are counted from held/corpus text — no gradients, no separate training pass. `ensure_built` rebuilds the word stores when the graph changes.

---

## Repetition Guard & Self-Feedback

### Repetition guard — `looks_repetitive(text)`

Detects when generation has collapsed into a short repeating cycle (a period of 4–80 chars repeating). Generation breaks early when this trips, so a "…same phrase over and over…" spew never happens, and a fragment that still reads as a cycle is dropped from the composed reply.

### The system speaks for itself

Unison always shows its **own** generated output. The teacher never produces a live reply in Unison's name — its only roles are to **judge** the output (`self_rate` + `teacher.rate`) and to **tutor** corrections (live on a bad turn, or via `/auto`). A correction is held as an orbit and its couplings/spans reinforced; the next time a similar prompt comes, Unison **composes** a coherent answer from its strengthened foundation — it does not retrieve the stored correction verbatim.

---

## The Learning Loop

This is how Unison actually improves. It is a closed loop through the teacher:

1. **Generate** → if the output is bad/confused/looping:
2. **Prune** the bad orbit, **rewind** the session, and **queue a `[CORRECTION]`** in `omni_ledger` (`build_correction_prompt` wraps the user's prompt, the failed output, and the evaluator's reason).
3. **`/auto` (Phase A — targeted tutoring):** for each queued `[CORRECTION]`, the teacher writes the ideal response; it is held as an orbit and its content-word couplings/spans reinforced.
4. **Next similar prompt** finds the topic's foundation strengthened, so Unison **composes** a more coherent answer of its own — learned from the correction, not a verbatim replay of it.

Complementary self-feedback (`on_message`): a **good** output rated `good` is `fold_orbit`-**closed** (kept as the invariant, not thickened into replay); a **bad** one is pruned and corrected **on the spot** — the teacher answers in-persona with the full conversation, Unison posts the recovery, and the corrected exchange is taught live (no `/auto` wait, no dead turn).

### The fold-native learning loop (develops coherence, never replays)

Generation is required to be compositional and non-verbatim. The pair stack is the current served development stage; the former foundation fallback is retired, and the native causal-transformer translation is the active generalisation build. Historical learning measurements retain the scope and provenance of the named build that produced them.

- **Corrections are internalized, never replayed:** a taught correction is held as **multiple expressions of one meaning** (the teacher supplies several phrasings — a human tutor could equally); serving cross-composes a fresh expression that equals no stored string. Re-expression lawfully requires **b = 2** held expressions (`generation_selection_law.ep`); below b, the engine defers and learns another.
- **Feedback → counted quality:** 👍 / `good` raises the Laplace pair-quality fraction of the pairs a reply was built from; 👎 / `bad` lowers it — re-ranking without memorising anything.
- **The competitive ladder (live, `GraduationLedger`):** every turn is a head-to-head — the engine's coherence critic vs the teacher. A territory graduates at **p ≥ 1/2** (the fold lock, Step 181), after which the engine is **sovereign** there: it judges itself by the fold critic and stops calling the teacher (its **own teacher** / observer regime, `observer_resolved.ep` — the fold acting one level up).

---

## The Teacher (`teacher_scaffold.py`)

### `LocalTeacher`

Interfaces with Gemma-4-31b via Ollama (`/api/generate`).

- **`ask(question)`** — the conversational call. Ollama's native `think` is **off** (`"think": False`); instead the system instruction asks the model for an explicit `<thinking>…</thinking>` block, which Gemma follows far more reliably than its hidden reasoning field can be steered. `_split_thinking` parses the block out and it returns `"\x04 {thought} \x05 {answer}"`, so the reasoning is stored and learned alongside the answer.
- **`generate_raw(prompt, system, num_predict)`** — a plain, no-persona generation call used for meta tasks such as authoring curriculum seeds.
- **`self_rate(avg_depth, avg_cands, chars)`** — Unison rating *its own* output from prediction confidence: depth `0` or `< 3` ⇒ `bad` (babble/near-random); `≥ 10` ⇒ strong; else acceptable.
- **`rate(user_prompt, response, curriculum, history)`** — the teacher grading Unison's output `GOOD`/`BAD`.
- **`get_system_instruction()`** — the persona: it *is* Unison; natural, short, warm; no robotic phrasing; no tech dumps unless asked; ask questions back; never break character.

### Auto-detected context window — `detect_context_window(model_name, base_url)`

The teacher's context window is **read from the model provider on connection**, never hardcoded. It queries Ollama `/api/show` and uses `model_info["<arch>.context_length"]` (for `gemma-4-31b` this is **262144** tokens). That value is used as `num_ctx` for every teacher call and to size the self-play running-context window. Falls back to a conservative floor only if the provider is unreachable.

---

## The Live Curriculum (`curriculum_communication.py`)

Nothing in the curriculum is a hardcoded training seed. The only authored content is the list of **skill areas** to teach (`SKILL_AREAS`): **communication**, **learning** (meta-cognition), and **questioning** (the confusion→clarifying-question reflex). Every actual practice prompt is **generated by the local teacher** and the curriculum is kept **live**:

- **`generate_seeds(teacher, areas, n_per_area)`** — the teacher authors a fresh batch of varied practice prompts for each area.
- **`refresh_curricula(teacher, areas, n_per_area, cache_path)`** — generate a fresh batch, **merge** it into the persistent cache (`curricula_generated.json`, de-duplicated), and return the merged set. This single call is *regenerate + grow + persist*.
- On `/auto` start, `DistillationEngine.refresh_all()` regenerates all areas (in a worker thread, so the Discord heartbeat is never blocked).
- During `/auto`, every `CURRICULUM_GROW_EVERY` (15) iterations, `grow_curricula()` authors a fresh batch for one rotating area and merges it in — so the curriculum keeps expanding while it runs.

Result: the seeds are always teacher-authored, fresh each session, continuously growing, and never lost.

---

## Distillation & Self-Play (`distill.py`)

### `ModelPool` / `ModelSpec`

Discovers every locally available model (Ollama + GGUF on disk) and assigns each a rotation of curricula (the generated `communication`/`learning`/`questioning` plus the SFT-corpus curricula `sft_sm_qft`, `sft_cosmology`, `sft_biochem`, `sft_classical_condensed`, `sft_tool_use`).

### `DistillationEngine`

Orchestrates multi-model distillation into the `SynapticGraph`:
- Installs the (teacher-generated) curriculum at start and grows it during the run.
- `get_next_seed()` cycles seeds; `advance()` moves to the next seed on **mastery** (stagnation) **or** once a per-seed iteration budget (`max_iters_per_seed`) is spent, so the curriculum is covered even when generation stays varied and mastery never registers. A failed seed queues a correction but does **not** reset the curriculum pointer, so self-play progresses through every curriculum (`communication` → `learning` → `questioning` → the SFT topics) instead of sticking on the first.
- `query_model(...)` calls the current model with the running conversation history prepended and `num_ctx` set to that model's provider-reported window.

### The `/auto` loop (`SFTDiscordClient._auto_loop`)

- **Phase A — targeted tutoring:** drains `omni_ledger`, teaching each queued correction (this is the core learning loop).
- **Phase B — self-play:** Unison babbles a curriculum prompt, the teacher answers, the exchange is banked into a growing running context; sustained novel exchanges are `fold_orbit`-**closed** (kept as the invariant, not thickened into replay), detected loops are pruned.

### `GraduationLedger` (`teacher_scaffold.py`)

Tracks win/loss per knowledge territory in blind head-to-heads (Unison's answer vs the teacher's). A territory "graduates" — Unison answers there itself — when `wins / total ≥ 1/2` (the fold's lock).

---

## Sessions & Identity

- **`Session` / `SessionManager` (`session.py`):** the `working_context` *is* the live orbit — the exact character sequence of the current conversation with `\x02/\x03` demarcation. Each session carries its own `episodic_memory` (a separate `SynapticGraph`, in-memory). Ending a session (`/new`) banks the whole conversation as one coherent orbit.
- **`UnisonIdentity` (`identity.py`):** stamps a hash-chained provenance proof on each response (session, turn, content).
- **`UserFingerprint` (`identity.py`):** per-user profiles (display name, id, interaction history), keyed by `ukey = "discord_" + crc32(author_id)`.

---

## Modalities: Sight, Hearing, Voice, Image

- **Sight — the fold eye, `ImageEncoder` (`modalities.py`):** the integer-Walsh "fold eye" from the paper is **built**. An image is grayscaled and resized to a power-of-two grid, read as an exact integer field, transformed by `fwht_2d` (which self-certifies Parseval or halts), and the **top BAND = b^(b+c) = 32** coefficients become the sight tokens (`DC=…|pos:val;…`), wrapped in `IMAGE_START`/`IMAGE_END`. Pure integers; the percept is discarded if Parseval doesn't hold.
- **Hearing — `WhisperListener` (`voice.py`) + `AudioEncoder` (`modalities.py`):** audio/video attachments are transcribed (Whisper on MPS) into the text channel; the integer-Walsh "fold ear" (`AudioEncoder`, same top-32 Parseval-certified band) is implemented alongside.
- **Voice — `KokoroSpeaker` (`voice.py`):** the 🔊 button synthesises Unison's reply via Kokoro TTS (or, when unavailable, Unison attempts to babble raw audio tokens).
- **Image generation — `FluxGenerator` (`flux_gen.py`):** generates images for the vision self-play curriculum.

Each perceptual organ is optional; when its backing model is absent the feature reports its absence rather than failing silently.

---

## Tools

`ToolOrchestrator` (`tools.py`) parses tool-use JSON emitted in a response and executes it (e.g. time/date, scratchpad, code/log reading, web access). Tool output is appended to the response and, via the learning loop, becomes held experience so tool use can be learned by watching.

---

## Persistent Memory & Logging

### `graph_memory.json`

Every orbit ever banked, keyed per user:
```json
{ "discord_2812840720": [ ["","H","i",""], ... ], "public": [] }
```
Atomic, debounced writes; "zero forgetting" — orbits are removed only by explicit pruning (👎 or auto-sever).

### Logs (`omni/logs/`, `logging_config.py`)

| Logger | File | Contents |
|--------|------|----------|
| `OmniCore` | `core.log` | SFT violations, FWHT/Parseval checks |
| `OmniMemory` | `memory.log` | orbit holds/prunes, load/save |
| `OmniTeacher` | `teacher.log` | every teacher query, thought/answer sizes |
| `OmniBot` | `discord.log` | message handling, `/auto` events, errors |
| `OmniSession` | `session.log` | session lifecycle |
| — | `learning.jsonl` | structured JSONL of every learning event |
| — | `diagnostics.jsonl` | per-generation & per-teacher-call timing |

`.log` files rotate at 5 MB with 3 backups.

---

## Discord Commands & Buttons

| Trigger | Action |
|---------|--------|
| `/auto` | Toggle the continuous background tutoring + self-play loop (also generates/grows the curriculum). |
| `/new` | End the current session, bank it as one coherent orbit, start fresh. |
| `/clear` | Wipe the **runtime** — orbits (taught lessons + conversation), sessions, ledger, graduation race — for a blank conversational slate. **Preserves the conversational foundation** (`word_fluency.pkl` + `word_coupling.pkl` + the retrieval index) and the logs, so full language ability survives the reset. |
| `/scrape` | Autonomously scrape vetted conversational datasets and rebuild the language foundation (fluency + couplings). |
| `/diag` | Live graph statistics (orbits, characters, user keys, session/working-memory sizes). |
| `/diagnostic` | Toggle per-message latency diagnostics in chat. |
| `/models` | Show discovered models and distillation progress. |
| `/voice` | Toggle automatic TTS on replies. |
| 👍 button | Confirm the orbit (semantic solidification). |
| 👎 button | Sever: prune the orbit, queue a correction for tutoring, rewind context. |
| 🔊 button | Speak the reply via Kokoro TTS. |

---

## Prerequisites, Setup & Running

**Requirements:** Python 3.9+, Ollama (with `gemma-4-31b:latest` pulled), `discord.py`, `numpy`. Optional organs: Kokoro (voice), Whisper (hearing), a Flux/SD backend (image gen).

```bash
# Teacher model
ollama pull gemma-4-31b:latest && ollama list

# Discord token (project root .env)
echo "DISCORD_TOKEN=your_token_here" > .env

# Run (from the engine directory)
PYTHONPATH=. python3 omni/discord_bot.py    # or: bash run.sh
```

Enable the **Message Content Intent** on the bot. The bot only responds in its configured channel (`ALLOWED_CHANNEL_ID` in `on_message`).

> **Note on the context window:** `num_ctx` is set to the model's full provider-reported window (262144 for `gemma-4-31b`). This is a large KV-cache allocation — comfortable on a high-memory machine (e.g. a 512 GB Mac Studio), but expect a heavier/slower *first* teacher call as the model reloads at that context size.

---

## Glossary of Special Tokens

| Token | Hex | Role |
|-------|-----|------|
| `\x02` | `0x02` (STX) | Start of a user's turn |
| `\x03` | `0x03` (ETX) | End of a user's turn / start of Unison's turn |
| `\x04` | `0x04` (EOT) | Start of Unison's internal reasoning (`💭 Thinking:`) |
| `\x05` | `0x05` (ENQ) | End of internal reasoning; spoken answer follows |

These are the first control codes after `\x01` (SOH), mapped to SFT's binary generator (`GEN_B = 2`).

---

## Empirical Record (2026-07)

Every number below is from a committed, timestamped harness run. The measurement discipline
is part of the architecture: **no instrument's number is believed before the instrument
has a recorded calibration** (clean separation of known-good from known-bad cases), no signal
that steers generation ever serves as its scoreboard, and end-to-end tests stub nothing —
the real path, the real teacher, the real stores. The percentages are timestamps on build
hours; **the gains between releases are measured development results.** Maria alone declares
the publishable conclusion and when a build deserves a real benchmark run.

### The task comparison — the counted substrate vs its gradient-trained twin
On identical held-out text, identical arena: **character scale 1.2891 vs 1.8878; word scale
3.1907 vs 3.4292** — the zero-parameter counted engine over the trained transformer twin at
both scales, the engine reading the corpus once against 48,000 gradient-batched passes.

### The learning law, measured
The live loop on the real path (calibrated judge as the user's feedback; the real teacher
correcting failures; corrections held as multiple expressions of one meaning and served only
re-expressed — nothing verbatim anywhere): **17% → 75% judged-good after one round of
teaching, stable across four further rounds** (three phrasings per correction; curve
[2, 9, 9, 9, 9]/12 on file). Taught once, composed natively thereafter — the Learning Law,
verified in believed units.

### The deep-context laws, forced and closed
The two quantities a transformer buys with tuned depth and trained mixing weights are forced:
**integration depth = b + c = 5** (the covering depth; minimality verified) and **step
weights = successive halvings of the fold factor**, closing to the One with the floor exactly.
The association step's spread is **unit capacity at the lock 1/2** (the neighbour-cap knob is
retired — the lock is the capacity); context binding and the taught-match threshold are the
same closed lock; re-expression requires **b = 2** held expressions. The taught match is
**kin-carried**: an identical content word binds whole, a counted coupling neighbour binds
at the half — a paraphrase reaches a held meaning through the same counted kinship that
built the coupling store, so the routing quantity and the learning-loss quantity below are
one and the same. Corpus steps
`contextual_integration.ep` and `generation_selection_law.ep`: **27/27 checks**, form-closed
by cross-routed reuse of already-closed forms, zero new constants. The engine's lock layer
cross-checks all of them forward at wake and **halts on any mismatch (halt proven)**.

### Causal training, translated
The native route uses the standard role-bound causal objective. Each training sequence is
`prompt address → assistant BOS → assistant tokens → assistant EOS`; every next-token event
is deposited once into the attention-value and FFN key/value tables. For a categorical
table this counted pass is the closed-form maximum-likelihood result, so it replaces repeated
gradient approximation without changing the target computation. Reward-conditioned learning
continues as persisted good/bad transition observations under the Laplace preference share.
Held-out conversational evaluation is separate from training and carries no architectural
validity or benchmark authority. Full trace: `NATIVE_TRANSFORMER_TRACE.md`.

The current sealed v4 store covers 649,917 role-bound pairs and 11,140,970
assistant causal targets. Its 2,921,639,096-byte artifact is SHA-256
`f977d4d8adb0993a0ab0d63b86dea30bd845ae091e84a4859ae826d032a87219`.
V4 retains every prompt token and executes structural, information, and counted
Q/K association heads. Its exact packed representation memory-maps every full
count table without pruning, capping, quantization, or eviction. A sealed
eight-prompt Codex probe preserved byte-identical surfaces while measured
resident memory fell from about 59 GB to 559,185,920 bytes and generation time
fell from 1,514.370707 seconds to 11.407350 seconds. The complete repository
suite passes 50/50. Distinct prompt positions now pass through five exact
contextual attention/FFN blocks before the final prompt position supplies the
decoder keys. A Codex-authored development smoke improved several
subject/operation surfaces. The final contextual position basis now reaches
decoder value/semantic routing exactly; position-conditioned ownership of those
training rows is the next established port.

### The conversational architecture
Serving now attempts the native counted causal transformer first. Pair retrieval remains a
separately disclosed RAG/response-selection augmentation. The historical word/generic
fallback and the agent-authored `free_gen.py` frontier arm do not define the native
architecture. Learning is exact corpus deposition plus reward-conditioned observations.

### Instruments
`train_eval/judge.py` + `judge_calibration.py` (the calibration receipt; the pool — two independent judge
models whose agreement is recorded), `generalisation_epochs.py` (historical pair-memory
curriculum instrument, not native transformer training), `grokking_run.py` (the first fixed-round
telescope), `gen_free_harness.py`, `gen_quality_honest.py`, `learning_curve_judged.py`,
`measure_pairs.py`, `bench_35b.py` (opponent-integrity enforced), `e2e_live_path.py` (nothing
stubbed). All committed; all reproducible.

---

## Engineering Notes & Roadmap

**The active workfront:**

- **Conversational generalisation.** The native full-token counted causal transformer is the
  primary serving architecture. Pair retrieval is a separately disclosed RAG augmentation,
  not the generator. The active work is to complete full-corpus measurement, preserve
  reward-conditioned learning, and continue the established transformer translation without
  reintroducing agent-authored fallback architecture.

**Roadmap:**

- **Per-token kin floor.** Kinship currently routes whole prompts (`kin_route`); the `rung5e` kin-shaped floor reshapes the continuation distribution at *every* generation step. Adding it makes kinship generalise inside a walk, not only at routing.
- **Per-segment banking.** Bank each sub-utterance as its own clean orbit and rate it independently, so every ask is learned in isolation.
- **Incremental word stores.** Update `word_engine` stores as orbits are held, rather than re-scanning `graph.orbits` on change.
- **Counted boundary maturity.** The `BoundaryStore` accrues boundary statistics from every message and begins refining segment cuts once it has the data — the learned tier taking weight from the structural rules.
- **Grow generation coverage.** `/scrape` more vetted conversational data (a purely human-written corpus build is a standing option — the mathematics is provenance-blind); grow the free arm under the forced cascade per FRONTIER_PLAN.md.

---

*This document reflects the `omni/` codebase as it currently stands. Where the code and this document ever disagree, the code is the truth — and this file should be corrected to match it.*
