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
| Positional/context representation | Current turn has unit share; preceding user turns carry successive exact dyadic age shares; repeated token occurrences deposit repeated structural mass. Distinct within-turn token-position addresses are not yet represented and remain an explicit production gap | `_positional_keys`, `_context_keys` |
| Causal query | The last two generated token identities form the current prefix query | `prev_id`, `last_id` |
| Attention keys | Every token identity from the current prompt and aged user history; no stop-word, punctuation, uniqueness, or authored semantic filter precedes attention | `_context_keys` |
| Multi-head Q/K compatibility | Three separately normalized exact projections of the complete token state: structural identity, inverse-exposure information, and conditional counted Q/K association. Their identity combination is exact addition with no fitted blend | `_attention_key_weights`, `_attention` |
| Attention values | One exact response-token count vector owned by each prompt key, kept factorised from Q/K and FFN | `values`, `_attention` |
| Causal mask | Only the already-generated prefix and preceding prompt/history are addressable; future response tokens are absent by construction | sequential build and `generate_tokens` |
| Feed-forward KV memory | Attention state enters an explicit `(previous, last, attended-key)→next-token` semantic FFN, falling to `(last, key)` only when the deeper address is absent; a separate assistant-prefix table retains syntax | `semantic_ffn3`, `semantic_ffn`, `_semantic_ffn`, `ffn2`, `ffn3`, `_ffn` |
| Residual addition | Add the unit attention value, semantic-FFN, and prefix-FFN distributions with the standard identity coefficient | `next_distribution` |
| Normalisation | Divide exact positive mass by its total and assert closure to the One | `_normalize` |
| LM head | Exact categorical next-token shares | `next_distribution` |
| Autoregressive decoding | Standard greedy decode from BOS until the learned EOS; `BAND` is the explicit execution budget. The live argmax groups exact row denominators, closes them to their least common multiple, and compares reward-conditioned integers by cross multiplication; tests compare it token-for-token with the materialized LM head. Prompt-key value rows are prepared once per response in a bounded local cache that is discarded after generation | `next_token_id`, `_integer_residual_scores`, `generate_tokens`, `generate` |
| Pretraining | One deterministic counted pass over every role-bound pair, the closed-form categorical MLE | `train_eval/build_native_transformer.py` |
| Reward-conditioned learning | Every observed good/bad native transition updates a persisted Laplace preference share `(good+1)/(good+bad+2)`; prompt and surface hashes bind each event, the live trace binds feedback to the exact native segment, and RAG/native ledgers cannot cross-update | `mark_feedback`, `_rewards`, `discord_bot._last_native_feedback`, `discord_bot._last_rag_feedback` |
| RAG augmentation | Existing pair retrieval remains a separately disclosed response-selection/RAG surface; it is not the native generator and not the retired fallback | `discord_bot._generate_fragment_multiscale` |

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

The complete repository suite passes 42/42.

The fixed Codex development probe does not establish full conversational
generalisation. Its projected-head route improved several subject/operation
surfaces, while greeting, gardening, meal, and music remained generic. The
next established transformer organs are stacked prompt self-attention and
distinct within-turn positional addresses, making keys context-dependent
before decoder attention. The exact memory-mapped representation and bounded
decode kernel now remove the prior runtime-memory obstruction without changing
the architectural computation.

## Superseded development interpretation

The Codex-authored v3 implementation reduced prompts to unique content words,
then used an inverse-frequency specificity head while leaving its stored Q/K
compatibility counts inactive. Its retained smoke records remain useful
implementation provenance, but that route is not the Unison architecture and
does not define a project result. V4 replaces those deviations with complete
token-level attention and an executable counted Q/K head before any full-corpus
result is considered.
