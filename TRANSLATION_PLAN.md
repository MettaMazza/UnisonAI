# The 1-1 Translation Plan — the full AI/ML architecture, expressed in the SFT mathematical model

**Premise (Maria, 2026-07-16):** math is math; the framework derives from first principles what
standard mathematical models do; AI/ML is mathematical computer science — established, tried,
tested; the task is **correct translation** of that science into the framework's exact, counted,
forced expression. Never improvised novelty on top of an already-novel system. A failed result
is a verdict on the translation's implementation, not on the established method.

**Scope:** the *entire* architecture — every organ of a modern AI stack mapped 1-1 to its
established mathematics and to its SFT expression, with the honest build-state of each.
State legend: **✅ built+measured** · **🔧 built, needs honest measurement** · **📐 to port/build** · **⛔ retired**.

---

## Part 0 — The constraint layer itself is established computer science

The SFT engine's ground rules are not exotic; each is a known discipline:

| SFT constraint | Established counterpart |
|---|---|
| Zero trained parameters; counts only | For categorical models, **counting is the closed-form maximum-likelihood estimator**. SGD is an iterative approximation needed only where no closed form exists (deep nets). One pass = exact MLE. |
| No floating point in any served probability; exact `Fraction` | **Exact rational arithmetic** — standard numerical-correctness practice; eliminates drift by construction. |
| Forced locks + `halt_violation` (halt on any fitted value / broken identity) | **Design by contract** (Hoare logic, Meyer): assertions as executable specification; violation = hard stop, never silent corruption. |
| Forced constants vs marked engineering bounds | Standard **derived-vs-configured constant** discipline; every number is either derived (cited to its `.ep` step) or declared an engineering bound in place. |
| Never-verbatim generation | **Substitution-based NLG** (delexicalize→relexicalize): emitted surface differs from any stored string by construction. |

---

## Part I — The full-stack 1-1 map

### I.1 Language substrate

| Standard component | Established mathematics | SFT expression | State |
|---|---|---|---|
| **Tokenizer** (BPE/WordPiece) | deterministic finite-state segmentation | per-character exact tokens + word regex `\w+\|[^\w\s]`; control codes `\x02..\x05` as speaker/thought demarcation | ✅ |
| **Token embeddings** (learned vectors) | **Levy & Goldberg 2014: SGNS embeddings factorize the shifted PMI matrix** — embeddings *are* smoothed counted co-occurrence | counted **kinship** (Jaccard over `NEIGH` neighbour distributions, `word_engine.kinship`) + the **PPMI coupling graph** (`word_coupling.pkl`) — the exact-count form of the same object. The decode campaign independently measured the embedding class as the universal law-carrier. | ✅ |
| **LM head** (softmax over vocab) | categorical distribution from counts | **exact rational shares** `count/total` (`memory.py:415`) | ✅ |
| **Smoothing** (unseen mass) | **Laplace/add-one smoothing** — the classical answer | the **No-Zero floor** `1/(total+1)` — identical formula, forced by the domain law rather than chosen | ✅ |
| **Backoff/interpolation** (Katz, Jelinek-Mercer, Kneser-Ney) | interpolated n-gram mixing with tuned λ per level | **fold-factor level mix** `2^L` across all holding depths (`word_engine.py:281,407`) — interpolated backoff with **forced** weights (Rung 5e). Beat the trained twin at word scale (3.1907 vs 3.4292). | ✅ |
| **Attention** (softmax QK selection) | soft selection over context; hard/top-1 attention is an established variant; her decode campaign measured trained attention leaning to the dyadic cascade 1/2,1/4,… in 12/12 layers | **unit-capacity selection** at the forced lock 1/2 (deepest-held-suffix selection); cascade weights `2^-k` where ranked mixing is needed | ✅ (substrate) |
| **Positional/context decay** (ALiBi-style recency bias) | exponential recency weighting | attention **halves with age**, `2^-age` — the forced factor b (Step 315), replacing a tuned decay (`session.py` design; wired into retrieval in Stage 2) | 🔧 |
| **Pretraining** (SGD over corpus) | MLE fitting | **one exact counted pass** (closed-form MLE — 26 s vs 48,000 gradient batches at the task gate) | ✅ |
| **Knowledge store** (FFN key-value memories — Geva et al.) | weights as implicit KV memory | **held orbits** — explicit, deterministically-addressed exact KV store; inspectable, editable at one record. The decode campaign measured the weight-store correspondence spectrally (Steps 308–320). | ✅ |

### I.2 Conversation (the active workfront — detail in Part II)

| Standard component | Established mathematics | SFT expression | State |
|---|---|---|---|
| **RAG / response selection** (retrieval chatbots) | context→response pair retrieval; inverted index + collection statistics | prompt→response **pair index** (395,787 pairs, `build_pairs.py`) | 🔧 |
| **Ranking** | **BM25** (Robertson–Spärck Jones) | exact formula; k1=6/5, b=3/4 as the standard's canonical values marked engineering (b=3/4 coincides with the forced 3/4 — noted, not claimed); `log` confined to rank *order*, never a served probability | 📐 Stage 2 |
| **Query expansion** | pseudo-relevance/thesaurus expansion over distributional similarity (counted, pre-neural) | **kin expansion** at half weight (`KIN_VOTE = 1/2` — the fold factor, already the engine's law) | 🔧 wire-in |
| **NLG surface** (task-oriented NLG) | **delexicalize → relexicalize** (ELIZA→AIML→slot templates) | slot classes by counted structural rules (proper nouns, numbers, source-prompt-only content words) rebound to the live conversation; **never-verbatim guard** (≥1 rebound slot or next candidate); generalizes the corpus's own tool-trace template law | 📐 Stage 3 |
| **Dialogue state tracking** (rule-based DST) | entity/slot memory, structural parsing (tokenizer-tier) | **relation-facts channel** — in the paper's design, **NOT yet in `omni/`**; port it (name recall depends on it) | 📐 Stage 4 |
| **Feedback learning** (click-through re-ranking; the counted core of RLHF's preference step) | **Laplace rule of succession** | per-pair quality `(good+1)/(good+bad+2)` — exact fraction multiplier on rank | 📐 Stage 4 (replaces span ±6 clamp 🔧) |
| **Taught QA memory** (FAQ exact-match) | exact/near prompt match precedence | teacher corrections append as pairs; exact-match outranks BM25; served through the same relexicalization (non-verbatim by rebinding) | 📐 Stage 4 |
| **Utterance segmentation** | finite-state sentence splitting + learned boundary statistics | `_structural_segments` (authoritative) + counted `BoundaryStore` (supportive, confidence-gated) | ✅ |
| ~~Loose-sentence splice generation~~ | *(no established counterpart — this was the improvisation)* | — | ⛔ retired (judged ceiling 0%) |

### I.3 Perception & speech

| Standard component | Established mathematics | SFT expression | State |
|---|---|---|---|
| **Vision encoder** (ViT patch embedding / CNN features) | **orthogonal-transform coding + top-k energy compaction** — the mathematics of JPEG and classical vision; Walsh-Hadamard transform coding is established image coding | the **fold eye**: integer `fwht_2d`, top-**32** coefficients (BAND = b^(b+c), forced), integer **Parseval self-certified per act** or the percept is discarded | ✅ built; 🔧 recognition quality unmeasured by an honest gate |
| **Audio front-end** (STFT/mel filterbank, MFCC) | orthogonal filterbank features | the **fold ear**: integer Walsh top-32 band, Parseval-certified (`AudioEncoder`) | 🔧 |
| **ASR** | trained recognizer as scaffold | Whisper as **teacher scaffold**; new sounds closed once, then native-recognized (removal-proof ladder) | 🔧 |
| **TTS** | trained synthesizer as scaffold | Kokoro as scaffold; replay from held record after one teaching | 🔧 |
| **Video** | frame+audio composition | composition of eye/ear — no new organ | 🔧 |

### I.4 Learning system

| Standard component | Established mathematics | SFT expression | State |
|---|---|---|---|
| **Distillation** (Hinton teacher→student) | supervised transfer from a stronger model | LocalTeacher (Gemma) corrections absorbed as held pairs/orbits; teacher is scaffolding with a **measured exit** | ✅ mechanism; 🔧 efficacy needs judged measurement |
| **Student-surpasses-teacher gate** | **the sign test at the 50% null** — the classical paired-comparison criterion | **GraduationLedger p ≥ 1/2** (the forced lock, Step 181): blind head-to-heads per territory; sovereign on graduation | ✅ built |
| **Curriculum learning** (Bengio) | staged practice distribution, teacher-authored | live teacher-generated curriculum (`curriculum_communication.py`), grows on completion | ✅ |
| **Self-play** (TD-Gammon lineage) | generate → verify → retain only verified | `/auto` loop with **earned retention** (unverified output discarded) | 🔧 verification gate must be the calibrated judge, not the fold critic |
| **RLHF preference step** | preference-based re-ranking (bandit feedback) | 👍/👎 + judge verdicts → counted pair-quality fractions (I.2) — no reward model, the counts are the preferences | 📐 Stage 4 |

### I.5 Agency, sessions, identity, evaluation

| Standard component | Established mathematics | SFT expression | State |
|---|---|---|---|
| **Tool use / function calling** (ReAct) | reason→act→observe; tool traces as training data | `ToolOrchestrator` (path-jailed); every call held as a trace; **tool graduation** (taught once → engine runs the act natively, re-reads the world instead of reprinting stale values) | ✅ built; 🔧 breadth |
| **Context window** | fixed window + cache | **context = youngest memory** — one object at different ages, halving attention, no window cliff | ✅ design; 🔧 measure |
| **Conversation log for the judge/teacher** | append-only event log | `session.history_log` (write-once per turn, never trimmed) | ✅ verified on e2e |
| **Provenance/watermark** | hash-chained attestation | `UnisonIdentity` response proofs | ✅ |
| **Eval: LLM-as-judge** | judge + **calibration against known labels** (measurement theory) | Gemma temp-0 GOOD/BAD as sole conversational scoreboard, **calibration gate first** (known-good vs known-bad separation ≥ 9/10 each side); no steering signal is ever a scoreboard | 📐 Stage 0 |
| **Eval: benchmarks** | held-out CE; fixed public probes; head-to-heads | rung ladder (CE — real, committed); MMLU probe (committed baseline); bench_35b (judge must pass the gate before any win-rate) | ✅ CE / 🔧 rest |
| **E2E testing** | integration tests over the real path | `e2e_live_path.py` — nothing stubbed, real teacher, real `on_message` | ✅ |
| **Data pipeline safety** | curated sources + content filtering | vetted scrape allowlist; register/injection/URL filters at index build | ✅ |
| **The fold critic** (`coherence_value.ep` implementation) | *(as surface-quality scorer: none — it measured common-word co-occurrence)* | retired from all ranking/scoreboard roles until it passes the Stage-0 calibration gate; the forced law stands, the implementation was the fault | ⛔ as scoreboard |

---

## Part II — The conversational refactor (workfront detail)

The pipeline, all established, end to end:

```
user turn ──┐
history_log ─┤→ query build (turns at 2^-age) → kin expansion (half weight)
             │→ BM25 over pair prompts → × pair-quality fraction (Laplace)
             │→ top-N candidate responses (standalone-filtered at index build)
             │→ delexicalize source-context slots → relexicalize to live entities
             │   (relation-facts: name, topic; fewest-unbound-slots wins)
             │→ never-verbatim guard (≥1 rebound slot or next candidate)
             └→ reply  →  judge/teacher verdict  →  pair-quality counts, taught pairs
```

One mechanism serves greetings, topical chat, corrections, and identity questions — previously
four broken special cases. Greetings stop being a hole by construction (the corpus holds
thousands of real responses to "hello, how are you").

## Part III — Stages (each gated by the calibrated judge; measured values only, no self-imposed targets)

- **Stage 0 — Calibrate the ruler.** `judge_calibration.py`: known-good (real corpus responses
  on their own prompts) vs known-bad (word-salad, off-topic, fragments); ≥9/10 separation each
  side or the judge is fixed until it passes. Nothing else counts until this does. Same gate for
  the bench_35b judge and for the fold critic if it is ever to return.
- **Stage 1 — Index v2.** Standalone filter; tf, lengths, avgdl, N, df stored; full-corpus rebuild.
- **Stage 2 — Ranking.** BM25 + kin expansion + `2^-age` context + quality fraction. **Measure
  the judged top-1 ceiling** (diagnostic serve, raw) — the architecture's honest ceiling.
- **Stage 3 — Realization.** Delexicalize/relexicalize; verbatim guard; wire `compose_reply` and
  `generic_reply`; retire the splice path. **Measure judged end-to-end.**
- **Stage 4 — Learning + state.** Pair-quality counts; taught-pair precedence; **port the
  relation-facts channel** (name recall); verify multi-turn recall on the no-stubs e2e.
- **Stage 5 — Honest replacements.** Re-measure everything retracted (gen quality n≥32,
  learning curve in judged units, bench_35b with calibrated judge); publish measured values.
- **Stage 6 — Organ audit + cleanup.** Run the same honesty pass over the non-conversational
  organs marked 🔧 (fold eye/ear recognition, self-play retention gate, tool breadth): each gets
  a calibrated instrument before its next claim. Delete dead paths; sync dev→repo; one accurate push.

---

*Discipline: instruments calibrate before they count ([[honest-evaluation-first]]); e2e over the
real path, nothing stubbed ([[e2e-no-stubs]]); never verbatim, never canned
([[never-generate-verbatim]]); engineering negatives are implementation verdicts, not approach
verdicts; forced vs engineering constants marked in place ([[sft-derived-vs-engineering-constants]]).*
