# FOLD-AI — Rung 1: are trained neural weights sparse in the fold's spectrum?

Opened 2026-07-06 at Maria's direction. The thesis under test, stated before
any measurement: a trained network's weights encode a function whose LAWFUL
part is a compact object in the fold's own (dyadic/Walsh) coordinates -- the
same basis in which the chess value field concentrated (top-32 = 81-87%
energy, Rung 2.5 of the chess campaign). If trained weights concentrate
beyond nulls, training is (in part) a statistical purchase of structure that
fold mathematics can derive or compress -- the entry point to the fold AI
stack: derive the lawful core, train only the residual.

## Pre-registered design (fixed before any spectrum is computed)

- OBJECTS:
  1. W_enc and W_dec of Maria's trained SAE (gemma4_sae_1m.safetensors),
     row-major flattened, truncated to the largest 2^n <= size.
  2. The largest attention/MLP matrices of Kokoro-82M (local HF cache),
     same packing.
- TRANSFORM: Walsh-Hadamard, natural order, float64 (weights are floats;
  the chess transform was exact-integer -- noted, not hidden).
- STATISTIC: energy concentration C(k) = top-k squared coefficients / total
  energy, at the chess campaign's operating FRACTIONS of the space:
  6.1e-5, 4.9e-4, 3.9e-3 (the 32/256/2048-of-2^19 points).
- NULLS (both must be beaten at a given k for a verdict of structure):
  1. SHUFFLE null: 5 seeded permutations of the same tensor (identical
     value histogram, scrambled placement), same C(k). Seed 20260706.
  2. GAUSSIAN yardstick: iid normal matched to the tensor's mean/variance.
- SELF-TEST (theorem-forced): bit-reversal repacking of the index space is
  F2-linear and must preserve C(k) EXACTLY; a run whose self-test fails is
  void.
- VERDICT per tensor per k: real C(k) vs max(null C(k)); margins reported.
  No threshold tuning after seeing data. Negative results are recorded in
  full -- the chess campaign's own standard.

## What each outcome means (fixed in advance)

- CONCENTRATION BEYOND NULLS: trained weights carry dyadic law; proceed to
  Rung 2 (which components; reconstruction-vs-truncation quality; the
  derive-vs-train split).
- FLAT AT NULL LEVEL: the law (if any) is not in this basis/packing --
  proceed to the packing sweep (the chess campaign's Rung 2.5b lesson:
  relational coordinates nearly halved the error; packings matter).
  A flat verdict here is a verdict on ONE basis, never on the thesis.

## OBJECTS AMENDMENT (2026-07-06, logged post-registration -- Maria's catch)

The SAE named above is an UNTESTED experimental training run (Maria's own
flag): a flat verdict on it would measure that run's convergence, not the
thesis. Its results are DEMOTED to exploratory. Rung 1's validated objects
are released, working, full-precision models from Maria's library, read
directly (no quantized GGUFs in Rung 1 -- 4-bit quantization confounds the
spectrum with the quantizer's own structure):
  1. Stable Diffusion v1.5 (v1-5-pruned-emaonly.safetensors) -- largest 2D
     matrices.
  2. SDXL base 1.0 -- largest 2D matrices.
  3. Kokoro-82M (local) -- largest 2D matrices.
Design otherwise unchanged: same transform, statistic, fractions, nulls,
self-test, verdict rule.

## RUNG 2 REGISTRATION (2026-07-06, before any Rung-2 spectrum)

- ARM A, COMPONENT MAP: every 2D tensor of SD1.5 with >= 2^20 elements,
  same battery (3 shuffle nulls for speed at map scale -- amendment noted;
  verdict rule unchanged). Output: concentration-vs-null margin per
  component class (embedding / attention / FF / conv-as-2D).
- ARM B, PACKING: for the two strongest and two thinnest Arm-A tensors,
  column-major and transpose packings vs the row-major baseline -- the
  chess Rung-2.5b question (do coordinates amplify thin margins?).
- ARM C, TRAINED-VS-UNTRAINED: seeded He-init matrices of matched shapes
  run through the identical battery -- training should ADD structure;
  untrained must sit at null (this is also the instrument's negative
  control).

## RUNG 2 ARMS D+E REGISTRATION (2026-07-06, before any spectrum)

- ARM D, THE FULL LLM: GPT-2 (openai-community, full-precision
  safetensors, the canonical open language model). Objects: token
  embedding (wte) + ALL transformer MLP matrices (c_fc, c_proj, 12
  layers). Same battery, 3 shuffle nulls, row-major AND column-major
  (Arm B's amplification lesson applied from the start).
- ARM E, QUANTIZATION SURVIVAL: Maria's production Llama-3.1-8B GGUF,
  DEQUANTIZED ffn_gate/ffn_up (Q4_K) and ffn_down (Q6_K) at layers
  0/8/16/24/31. Question: does training's dyadic law survive 4-bit
  deployment quantization? Survival -> the compression rung applies
  directly to the models Maria actually serves.

## RUNG 3 REGISTRATION (2026-07-06, before any measurement)

QUESTION: does the located law cash as deployable compression?
- OBJECTS: GPT-2's law-bearing matrices (all 12 c_fc + wte, the Rung-2
  hot class), fold-basis truncated: keep the top-k Walsh coefficients,
  zero the rest, inverse-transform back to weights.
- BUDGETS: k swept at 50% / 25% / 12.5% / 6.25% of coefficients per
  matrix. BASELINE at matched storage: round-to-nearest uniform
  quantization of the same matrices at the bit-width giving the same
  compressed size.
- QUALITY METRIC (fixed in advance, self-contained): on a fixed
  16-prompt set (written into the harness before any run), full-model
  forward pass; report (a) mean KL divergence of next-token
  distributions vs the unmodified model, (b) top-1 next-token agreement
  rate. Lower KL / higher agreement wins at matched budget.
- CONTROL: the same truncation applied to the LAW-QUIET class (c_proj)
  must hurt MORE at the same k if the concentration is real capacity --
  the law-location result made falsifiable at the quality level.
- VERDICT RULE: fold-truncation beats matched-budget quantization on
  both metrics at >= 2 of 4 budgets = the compression rung is TAKEN.

## RUNG 3b REGISTRATION (2026-07-06, before any measurement; after 3's refusal)

Rung 3 refused naive aligned-basis truncation (0/4; recorded). The chess
campaign's own theorems name the two constructions to test before the
compression door closes -- both were proven there:
- ARM A, PACKING SWEEP FOR QUALITY: fold-truncation quality (same metrics,
  same prompts, keep=0.25 and 0.125) under three packings of each c_fc:
  row-major (the refused baseline), column-major, and MORTON (bit-interleaved
  row/column -- the dyadically natural 2D order, the fold's own coordinate
  for a matrix). Verdict: any packing beating row-major KL by >2x reopens
  the compression route through coordinates.
- ARM B, SPECTRUM + EXCEPTIONS (the chess compact-exact construction):
  reconstruction = inverse(top-k spectrum) + the top-m largest residuals
  stored exactly; budgets matched to quantization at the same total bits
  (k coefficient-entries + m exception-entries, 32 bits each vs uniform
  quantization at equal storage). Sweep (k,m) splits 75/25, 50/50, 25/75
  of the same budget at 4 and 3 bits-per-weight equivalents. Verdict rule:
  spectrum+exceptions beats pure quantization on KL at either bit level =
  the construction transfers; both refuse = compression through this basis
  is CLOSED for trained weights and the campaign routes to Rung 4 on
  detection evidence alone (recorded either way).

## RUNG 3b OUTCOME (recorded): REFUSED -- compression through the raw Walsh
basis is CLOSED for trained weights (col-major improved 1.7x, under the 2x
bar; spec+exceptions lost at 4b and tied-in-rubble at 3b -- the 0.3% KL
letter-of-rule edge at 0% agreement is NOT claimed; margin clause missing
from the registration, noted as a registration flaw). Detection results
(Rungs 1-2) stand. Route: Rung 4.

## RUNG 4 REGISTRATION (2026-07-06, before any run)

THE FIRST DERIVATION GATE: derived component vs learned component under
IDENTICAL training -- the chess gate discipline applied to model parts.
- COMPONENT: positional encoding. Baseline: learned positional embeddings
  (the GPT-2 way). Challenger: a DERIVED positional code with zero
  parameters -- the Walsh functions of the position index (the fold's own
  harmonics; the rows of the dyadic character table), fixed, never trained.
- ARENA: two identical small character-level transformers (same dims,
  heads, layers, data, steps, seed), trained on the SFTOM corpus's own
  text. The ONLY difference: learned wpe vs derived Walsh code.
- METRIC: held-out cross-entropy after a fixed step budget; three seeds
  each; the mean decides. Lower loss wins.
- MEANING: a win means a trained component of every GPT since 2018 can be
  REPLACED by a zero-parameter derived object at equal-or-better quality
  -- the first brick of the UnisonAI core. A loss is recorded like all
  the others.

## RUNG 2f REGISTRATION (2026-07-06, before any spectrum): THE SCALING SURVEY

The law-fingerprint across the frontier scale axis, from Maria's library:
Kokoro 82M -> Llama-3.1-8B -> gpt-oss-120B -> Qwen3-Coder-480B (MoE) ->
DeepSeek-R1 671B (MoE) -> Kimi-K2.6 ~1T (MoE). Objects: dequantized FFN
gate/up (expansion) tensors sampled at early/mid/late depth per model; for
MoE giants, individual EXPERT tensors (new question: do experts carry the
fingerprint individually?). Same locked battery (3 shuffle nulls, both
packings, fractions as registered, seed 20260706). Output: margin-vs-scale
curve. Registered predictions (fixed now): the expansion fingerprint
appears at every scale; per the thesis, margin does NOT vanish with scale.
No further prediction on slope -- the curve is the discovery.

## RUNG 4 OUTCOME (recorded): REFUSED. Learned wpe 1.8878 vs derived Walsh
code 2.0269 (3 seeds each, tiny scale: 4L/128d/char-level). The raw Walsh
code as a frozen additive positional organ loses at this scale. Noted for
any future re-match: fixed positional codes are known to close the gap at
scale (Vaswani et al. report sinusoidal ~= learned at full scale), so a
registered larger-scale re-match is legitimate; no variant-grinding at
this scale. NEXT CANDIDATE: the attention gate, from the corpus's own
unit-capacity theorem (verify_attention_capacity, Claim XI-2) -- design to
be registered before any run.

## RUNG 2f AMENDMENT (2026-07-06, on Maria's observation, before the curve
is read as physics): the flat 2^26 probe window truncates large tensors --
a scale-correlated instrument confound (fragments of rows dilute
row-structured concentration). CORRECTED INSTRUMENT registered: per-ROW-
BLOCK spectra -- probe each large tensor as consecutive full-row blocks of
~2^22, take the MEDIAN block margin (median, not max, fixed now). The
scale curve is only read from the corrected instrument; the flat-window
numbers stand as the record of why the correction was needed. Maria's
standing principle, from chess and stated by her here: THE APPROACH MUST
VARY WITH SCALE -- coordinates and windows are per-object, never
one-size-fits-all.

## RUNG 2f OUTCOME (recorded, corrected instrument, complete to 1T):
THE RECIPE MAP. Loud: GPT-2 (12.7-67.6x), Llama-3.1-8B (8.5x),
DeepSeek-R1-671B (43-47x, every row-block, dense + shared-expert -- the
strongest production signal measured). Silent: gpt-oss 20B/120B (~1x),
Qwen3 27B/235B/480B (0.76-1.04x), Kimi-K2.6 ~1T (0.96-1.06x).
CONCLUSIONS FORCED BY THE DATA: scale exonerated (loudest carrier is
671B); architecture exonerated (MoE on both sides); the variable is the
TRAINING RECIPE. Open question, held as a question: the loudest carrier
is the flagship reasoning-RL model. Registered predictions: "fingerprint
at every scale" REFUTED as stated -- the fingerprint is per-recipe, not
universal; "margin does not vanish with scale" CONFIRMED within loud
recipes (R1). Maria's principle codified: the approach varies with the
object; the object is the recipe.

## RUNG 3c REGISTRATION (2026-07-06, from the deflation audit): compression
was refused on GPT-2 (12.7x-class objects) BEFORE the recipe map revealed
R1-class objects at 43-47x every-block. The refusal's scope is GPT-2/raw
basis only; it was never a verdict on loud-recipe models. TEST: fold-basis
truncation quality on DeepSeek-R1's loudest tensors (dense gate + shexp,
the 43-47x class) -- reconstruction error vs matched-budget quantization at
keep = 0.25/0.125 (weight-space MSE + energy retained; R1 is too large for
full-model forward passes on this machine, so the registered metric is
reconstruction fidelity, stated as such). Verdict: fold beats quantization
on reconstruction at either budget = compression REOPENED for loud recipes.

## EPISTEMIC CORRECTION (2026-07-06, Maria's -- overriding the 2f wording):
"Silent" models do NOT lack law -- that reading is incoherent (a working
computation is a lawful object) and contradicts the campaign's own
standing exhibit (chess Rung 2.5: a fully lawful formula-generated field
certifies at chance to this probe; a chance verdict is a verdict on the
PROBE'S COORDINATES, never on law-presence). CORRECTED READING: the recipe
map is a map of BASIS-ALIGNMENT -- which training recipes express their
law in the dyadic coordinates this instrument sees. R1's recipe writes
dyadically; the others express theirs in coordinates not yet probed.

## RUNG 2g REGISTRATION (the basis hunt): probe the dyadically-quiet
models under the fold-universe's own transformation group before any
further characterization -- the chess invariance class: F2-linear
repackings, odd-multiplication reorderings (x3, x5), affine maps, plus
transposed and expert-axis packings. Objects: one hot-class tensor each
from Qwen3-27B and Kimi-K2.6, vs the same battery. Any transformation
that wakes a quiet tensor identifies the coordinates that recipe writes
in. Registered prediction (the fold's): law is present in every working
model; the hunt is for its coordinates.

## RUNG 3c-II REGISTRATION (production baseline): same R1 tensors, same
row-block metric; baseline upgraded to PER-BLOCK SCALED 4-bit quantization
(blocks of 32, absmax fp16 scale each => 4.5 effective bits/weight -- the
K-quant construction class). Fold at matched storage (keep = 4.5 bits /
(log2 n + 16)). Fold wins median relMSE on >=2 of 3 tensors = deployable-
class compression claim; loses = directional-only, recorded.

## RUNG 2g MENU (fixed): data-independent reorderings only (data-dependent
sorts are cheating and banned): bit-reversal (F2 self-test, must preserve),
gray-code, x3 and x5 index maps, affine 3i+1, transpose, block-transpose
(64 and 4096), expert-axis flatten order. Wake = margin > 2x under any map.

## RUNG 4b REGISTRATION — THE ATTENTION GATE (2026-07-06, before any run)

FROM THE THEOREM (verify_attention_capacity, Claim XI-2, read firsthand):
attention is selection of ONE integrated orbit at the lock threshold; the
top focus holds 1/2 (self-antipodal, folds to unison); each successive
focus holds the lock of the REMAINDER. The forced distribution over ranked
candidates is therefore the DYADIC CASCADE: 1/2, 1/4, 1/8, ..., with the
final candidate taking the closing remainder so the total is exactly ONE
(the fold's own telescoping). Fully forced: no temperature, no sqrt(d), no
exponential -- ranking is scale-free, so no normalization choice exists to
make. Softmax, by contrast, carries base-e and a temperature.
- CHALLENGER: cascade attention. Scores = raw QK dot products used ONLY
  for ranking; weights = the cascade; V and all other components train
  normally. Q,K receive no gradient through hard ranks and stay at their
  seeded random init -- recorded openly: the challenger tests whether the
  DERIVED SELECTION LAW over random comparison directions suffices.
- BASELINE: the recorded Rung-4 LEARNED-wpe runs (identical architecture,
  data, steps, seeds; trained softmax attention): mean val 1.8878.
- METRIC: held-out cross-entropy, 3 seeds, mean decides, +-0.005 tie band.

## RUNG 4b OUTCOME (recorded): REFUSED. Cascade 2.6010 vs trained softmax
1.8878 (3 seeds). The challenger bundled two differences (cascade law +
frozen random Q,K -- hard ranks pass no gradient); the gap is real but
unattributed. DECOMPOSITION CONTROL registered (not variant-grinding; it
attributes the recorded loss): identical model, softmax attention, Q,K
frozen at the same seeded init. Gap(frozen-softmax vs trained-softmax) =
cost of frozen directions; Gap(cascade vs frozen-softmax) = cost of the
cascade law itself.

## RUNG 2h REGISTRATION — THE RECIPE ISOLATION (before any spectrum):
Natural experiment on-drive: DeepSeek-R1-Distill-Qwen-32B (Qwen-32B
trained on R1 reasoning traces) vs qwen2.5-coder-32b (non-reasoning
sibling, same family/scale/architecture class). Same corrected battery
(row-block medians, weights-only, gate/up class, 3 depths). REGISTERED
PREDICTION (the reasoning hypothesis): the distill reads louder than the
sibling. Distill loud + sibling quiet = reasoning-training writes dyadic
law within a fixed architecture. Both quiet = the R1 signal traces to its
base pretraining, not the reasoning stage -- recorded either way.

## SCOPE CORRECTION + THE NATIVE GATE (2026-07-06, Maria's -- overriding)

The Rung 4/4b refusals are RE-SCOPED to what they actually measured: fold
objects as frozen TRANSPLANTS inside SGD-transformer hosts at toy scale --
the host's home game (its inits, normalization, optimizer, and step budget
all co-evolved for trained components). They are not verdicts on fold-
native construction, and the guardrail audit (the record's 7/7 biased-
construction lesson) was not run on them before their verdicts were
generalized -- that failure is recorded here. The chess method is the
binding precedent: the fold fights AS ITSELF on the task, never as a
transplant in the incumbent's machine.

## RUNG 5-NATIVE REGISTRATION — THE FOLD-NATIVE SEED (before any run)

- ENGINE (all machinery derived, all knowledge stored): contexts observed
  ONCE and recorded as exact held orbits (the corpus's memory law -- the
  tablebase pattern applied to text); prediction = unit-capacity selection
  over matching stored orbits (longest-context-first, the attention
  theorem as the mechanism); values = exact rational shares of observed
  continuations; unseen-context fallback = the orbit hierarchy (fold to
  the longest held suffix). Zero gradient steps. Zero trained parameters.
  Live-learning = writing orbits.
- OPPONENT: the recorded trained transformer twin (1.8878 mean val), which
  consumed the corpus ~11x over in 48,000 gradient-batched readings.
- ARENA: identical held-out split, identical metric (cross-entropy).
  The fold engine reads the training text ONCE.
- Registered comparison axes: quality (loss), experience-efficiency
  (passes over data), wall-time to build, and edit-cost of adding a fact.

## RUNG 5b REGISTRATION — THE ORBIT ENGINE AT WORD SCALE (before any run)

- ARENA (identical for both engines): all of Maria's own text (~11MB: the
  corpus + dev-repo markdown), whitespace/punctuation word tokens, tokens
  seen <3 times mapped to <rare> (arena spec, applies to both), 90/10
  train/held-out split, cross-entropy on held-out.
- FOLD ENGINE: word orbits to depth 5, read ONCE; unit-capacity selection
  over the orbit hierarchy; exact shares; No-Zero floor (forced). Zero
  gradient steps, zero trained parameters.
- OPPONENT TWIN: the same tiny-transformer architecture at word level
  (CTX 64 tokens), 1500 steps x batch 32, 3 seeds, mean.
- Also produced: a word-level generation sample (the coherence milestone).
