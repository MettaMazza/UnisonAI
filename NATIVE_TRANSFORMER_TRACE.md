# Native causal-transformer translation trace

## Scope and authority

This is Unison's native conversational-generalisation architecture. It is a
one-to-one constitutional re-derivation of established causal-transformer and
training computation under SFT arithmetic and constraints. It is not a repair,
rename, or interpretation of the retired word/generic fallback.

The engine owns forcing and halt validity. Maria Smith owns publishable
conclusions, benchmark timing, and project direction. Implementation tests
below establish code behaviour only; they do not declare a benchmark result.

## Organ-by-organ trace

| Established transformer/training organ | SFT translation | Executable anchor |
|---|---|---|
| Role-bound training sequence | The complete observed prompt token sequence supplies key/address observations; only assistant response transitions occupy output/value state; explicit BOS/EOS boundaries prevent cross-response leakage | `build_counted_transformer` |
| Tokenizer | Deterministic word/punctuation segmentation and deterministic contraction merge; function words, punctuation, and repetition are retained before attention | `_tokens`, `_prompt_tokens` |
| Token embedding | Each content token is represented by its exact prompt co-occurrence row; this is the sparse counted relation that trained embeddings compress into a learned vector | `profiles`, `profile_index` |
| Positional/context representation | Every token occurrence has a distinct `(turn age, within-turn position, sequence position)` address. Relative-position products are ranked onto the complete dyadic cascade; older turns carry successive exact dyadic age shares | `PositionAddress`, `positional_head`, `_position_addresses` |
| Causal query | The last two generated token identities form the current prefix query | `prev_id`, `last_id` |
| Attention keys | Every addressed token occurrence from the current prompt and aged user history enters five contextual blocks; the final prompt position supplies the decoder readout, as in causal next-token prediction. No stop-word, punctuation, uniqueness, or authored semantic filter precedes attention | `prompt_context.contextualize`, `aggregate_keys`, `_contextual_keys` |
| Multi-head Q/K compatibility | Prompt self-attention uses the exact counted-embedding Gram product plus the relative-position product; product order selects the complete forced dyadic cascade with its closing identity floor. Decoder attention then applies its structural, inverse-exposure information, and conditional counted Q/K heads. Head combination is exact identity addition with no fitted blend | `prompt_context._bilinear`, `_cascade_distribution`, `_attention_key_weights`, `_attention` |
| Attention values | Final hidden mass remains split over every distinct prompt position through decoder value routing. Existing v4 response-token vectors remain token-owned. The active v5 build counts the complete `(relative position, prompt token, last, previous, next)` observation relation; exact marginals supply position-owned values and both semantic FFN depths without triplicating observations | `DecoderContext`, `_decoder_value_sources`, `values`, `build_position_conditioned_relation.py` |
| Causal mask | Only the already-generated prefix and preceding prompt/history are addressable; future response tokens are absent by construction | sequential build and `generate_tokens` |
| Feed-forward KV memory | Each prompt block applies a counted embedding-relation FFN over a response-local position basis. Decoder semantic branches now retain that position basis before reading `(previous, last, attended-key)→next-token`, falling to `(last, key)` only when the deeper address is absent; a separate assistant-prefix table retains syntax | `prompt_context._feed_forward`, `_decoder_value_sources`, `semantic_ffn3`, `semantic_ffn`, `ffn2`, `ffn3` |
| Residual addition | Add the unit attention value, semantic-FFN, and prefix-FFN distributions with the standard identity coefficient | `next_distribution` |
| Normalisation | Divide exact positive mass by its total and assert closure to the One | `_normalize` |
| LM head | Exact categorical next-token shares | `next_distribution` |
| Autoregressive decoding | Standard greedy decode from BOS until the learned EOS; `BAND` is the explicit execution budget. The live argmax groups exact row denominators, closes them to their least common multiple, and compares reward-conditioned integers by cross multiplication; tests compare it token-for-token with the materialized LM head. Prompt-key value rows are prepared once per response in a bounded local cache that is discarded after generation | `next_token_id`, `_integer_residual_scores`, `generate_tokens`, `generate` |
| Pretraining | One deterministic counted pass over every role-bound pair, the closed-form categorical MLE | `train_eval/build_native_transformer.py` |
| Reward-conditioned learning | Every observed good/bad native transition updates a persisted Laplace preference share `(good+1)/(good+bad+2)`; prompt and surface hashes bind each event, the live trace binds feedback to the exact native segment, and RAG/native ledgers cannot cross-update | `mark_feedback`, `_rewards`, `discord_bot._last_native_feedback`, `discord_bot._last_rag_feedback` |
| RAG augmentation | Pair retrieval remains a separately disclosed response-selection/RAG instrument; it is not the native generator or a fallback. The served native method now returns native output or defers without calling retrieval | `discord_bot._generate_fragment_multiscale`, `pair_retrieval.py` |

## Explicit exclusions

- No learned floating-point weights.
- No beam width, top-k, top-p, temperature, random tie noise, reward scale, or
  candidate-count cap in the native route.
- No sub-half fallback lock.
- No loose-sentence splice, canned generic response, or verbatim response walk.
- No agent-authored quality verdict is used as architectural validity.

## Verification

`tests/test_native_transformer.py` checks the role boundary, exact closure of
attention/FFN/LM-head distributions, prompt-dependent causal generation,
complete prompt-token retention, active counted Q/K weighting, dyadic history
shares, and persistent counted reward learning. The full-corpus
artifact build records its source-pair hash and reports its exact sequence,
target, vocabulary, attention-key, and FFN-key counts.

The current v4 receipt seals 649,917 role-bound responses, 11,140,970
assistant causal targets, 22,415,744 Q/K relations, 96,721 value vectors, and
81,111,826 deep contextual FFN addresses. Artifact SHA-256 is
`f977d4d8adb0993a0ab0d63b86dea30bd845ae091e84a4859ae826d032a87219`;
the packed receipt verifies all seven serving tables with zero failures.

The original pickle is translated into an exact memory-mapped serving
representation by `train_eval/pack_native_transformer.py`. No row is sampled,
pruned, capped, quantized, or evicted: the seven tables retain their complete
keys and integer counts, and lookup verifies each stored key. The packed store
contains 6,832,650,424 bytes and is bound to the original artifact SHA above.
`train_eval/seal_packed_native_transformer.py` hashes every index and data file;
the runtime additionally halts if the packed manifest or either serving source
file drifts from `native_transformer_v4_packed_receipt.json`.

Exact-equivalence tests compare every fixture row, full distributions, greedy
argmax, and generated surfaces between the pickle and packed forms. The fixed
eight-prompt development probe was byte-identical across the representation
change. Measured on the same machine, resident memory fell from approximately
59 GB for the unpacked Python object to 559,185,920 bytes for the bounded packed
route. Cold activation took 0.032017 seconds and the eight generations took
11.407350 seconds total, compared with 1,514.370707 seconds for the earlier
unpacked projected-head probe. These are Codex implementation measurements,
not a benchmark conclusion.

The focused native transformer and position-relation suite passes 23/23, and
the complete current repository suite passes 55/55.

The five-layer prompt-context route is now the production native key path.
Hidden values remain factorised over the complete response-local position basis,
so support cannot exceed the observed prompt positions and no cap is introduced.
Shared-denominator integer state is proven equivalent to the direct Fraction
mixture on fixtures. The Q/K product selects the ranked cascade forced by
`attention_in_the_product.ep` and `partition_localization.ep`; the closing
remainder stays on the identity address. The final prompt position is read into
the decoder, preventing a bag-of-token aggregation from erasing order.

The source-bound eight-prompt Codex probe closed every contextual key state to
the One, used 701,612,032 bytes maximum resident memory on the promoted route,
and completed the eight generations in 19.627877 seconds. Seven surfaces were
unchanged from the packed v4 probe; public speaking changed to
`I don't know. i just want to do for fun?`. This is implementation evidence,
not Maria's benchmark conclusion. Decoder value and semantic routing now
consume the complete contextual final-position basis; an exact equivalence
test proves the branches recombine to the existing v4 token-owned rows. The
active next training port is to count position-conditioned value/semantic rows
from the role-bound corpus so those preserved branches can carry distinct
learned observations.

The source-bound contextual-cost receipt counts 14,510,060 prompt positions
across the 649,917 training pairs. Dense within-prompt attention would produce
475,093,198 position pairs per layer and 2,375,465,990 pair touches through the
five-step depth forced by `contextual_integration.ep`—a 19,003,727,920-byte
lower bound even if each touch held only one uint64 value and no key. The next
port therefore keeps the complete five-layer computation but factorises shared
counted relations and streams response-local position state; it does not
materialize duplicated layer-pair rows or introduce a cap.

The role-conditioned induction/copy head was exercised on a matched native
multi-turn panel: continuity changed from 0/4 to 4/4, and a separate transfer
panel returned 8/8 exact continuations. The served native path returned the
same four surfaces and no longer called pair retrieval after native deferral.
A free four-conversation probe then exposed self-activation after a generic
language-head opening; requiring the prompt itself to admit the induction
relation preserved 4/4 continuity and 8/8 transfer while removing every full
statement copy. All four free follow-ups became non-empty, but their surfaces
remained generic. A full-head probe located the remaining first-token issue in
the native heads themselves: the combined head assigned `I` the largest share
in all four cases (0.0857, 0.0866, 0.1070, and 0.0942). These are source-bound
Codex implementation measurements, not Maria's benchmark findings.

The next documented training port is now active rather than merely proposed.
Its full-corpus sizing receipt counts 277,583,049 position/target observations.
Naively storing value, semantic-FFN2, and semantic-FFN3 copies could require up
to 832,749,147 row/value entries. The implemented canonical relation stores
each `(relative position, prompt token, last, previous, next)` observation once;
exact marginalisation supplies all three established organs. The factorisation
passes an external sort-and-aggregate reconstruction test, and the complete
uncapped corpus build is resumable and currently in progress. No position
limit, candidate cap, sampled corpus, pruning rule, or fitted capacity enters.

## Superseded development interpretation

The Codex-authored v3 implementation reduced prompts to unique content words,
then used an inverse-frequency specificity head while leaving its stored Q/K
compatibility counts inactive. Its retained smoke records remain useful
implementation provenance, but that route is not the Unison architecture and
does not define a project result. V4 replaces those deviations with complete
token-level attention and an executable counted Q/K head before any full-corpus
result is considered.
