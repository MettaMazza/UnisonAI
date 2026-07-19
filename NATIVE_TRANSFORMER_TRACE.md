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
| Autoregressive decoding | Standard greedy decode from BOS until the learned EOS; `BAND` is the explicit execution budget. The live argmax accumulates the same exact normalized organ rows lazily and omits only the final common normalization; tests compare it token-for-token with the materialized LM head | `next_token_id`, `_unnormalized_residual`, `generate_tokens`, `generate` |
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
the complete repository suite passes 37/37.

The fixed Codex development probe does not establish full conversational
generalisation. Its projected-head route improved several subject/operation
surfaces, while greeting, gardening, meal, and music remained generic. The
next established transformer organs are stacked prompt self-attention and
distinct within-turn positional addresses, making keys context-dependent
before decoder attention. Runtime representation must also become streamed or
memory-mapped while preserving exact counts and argmax identity.

## Superseded development interpretation

The Codex-authored v3 implementation reduced prompts to unique content words,
then used an inverse-frequency specificity head while leaving its stored Q/K
compatibility counts inactive. Its retained smoke records remain useful
implementation provenance, but that route is not the Unison architecture and
does not define a project result. V4 replaces those deviations with complete
token-level attention and an executable counted Q/K head before any full-corpus
result is considered.
