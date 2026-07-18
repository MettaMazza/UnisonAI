# Codex-authored auxiliary generation-quality benchmark

This benchmark was introduced by Codex during development. It is not Maria's
prediction, not an engine-derived definition of parity, and not authority to
declare a project finding, loss, or endpoint. Its measurements belong to the
exact protocol below while Unison's benchmark-victory programme continues.

`generation_quality_campaign_v1.json` fixes the prompts, seed, arms, judge
pool, calibration rule, source hashes, and runtime-store inventory before a
quality result is allowed to exist.

The protocol has two irreversible stages:

1. `registered_generation_quality.py register ...` binds the exact runtime
   stores and local judge model digests in a new registration file.
2. `registered_generation_quality.py run ...` rechecks every bound identity,
   recalibrates both judges, and writes a new sealed result directory. Existing
   registrations and result directories are never overwritten.

The runtime stores were reconstructed on 2026-07-18 and sealed by
`runtime_store_build_receipt.json`. The builders now stage output, refuse empty
corpora or pair stores, and replace the destination only after a successful
build. The receipt binds 649,917 role-correct prompt/reply pairs, a 100,303-word
fluency vocabulary, 11,685 coupling words, 11,755 kin words, 4,618,246
conditioned keys, and 8,000,727 trigram contexts. It records resolved dataset
cache revisions while disclosing that the builder calls do not yet pass pinned
revision arguments; the exact artifacts are hash-bound, but byte-identical
future reconstruction is not claimed from source names alone.

`generation_quality_registration_20260718.json` bound the exact stores, source,
campaign, and local judge model digests before generation. In the sealed run,
both judges passed calibration at 10/10 known-good and 10/10 known-bad examples.
Across the 12 preregistered prompts, the unanimous-good measurement was 0/12
for the baseline, 0/12 for F1, and 0/12 for F3. This is the output of the
Codex-selected benchmark on those builds. It does not redefine Unison's
architectural parity, invalidate the engine's verified receipts, or declare
Maria's finding or loss. The forward-forcing and benchmark-victory campaign
continues under Maria's direction.

`verify_registered_generation_quality.py` independently checks the campaign,
registration, calibration, row-level unanimity, summaries, source bindings, and
seal hashes. `--verify-runtime` additionally rehashes every present registered
runtime artifact.

The legacy `gen_free_harness.py` remains development history; its mutable JSONL
log is not publication evidence.

## Response-only fluency development boundary

The sealed outputs exposed a common surface-realisation defect: the legacy
fluency builder treated the mixed conversational corpus as one continuous token
stream, allowing contexts to cross unrelated response and document boundaries.
`build_response_fluency.py` constructs a separate development
artifact from assistant responses in the sealed pair store only, resetting all
context at every response boundary. `response_fluency_v1_receipt.json` binds
649,917 responses, 10,491,053 tokens, 93,098 words, and exact order-1 through
order-4 context counts. The legacy artifact and sealed campaign are unchanged.
`response_fluency_runtime_arm_v1.json` now makes that sealed artifact explicitly
selectable by the native runtime while leaving the default unchanged. Maria
Smith alone decides real comparison timing and the conclusion supported by its
data.

The historical sealed campaign remains source-strict by default: current
development changes correctly produce a source-drift halt. The verifier's
explicit `--allow-source-drift` archival mode checks the immutable campaign,
registration, calibration, result, unanimity arithmetic, and seal without
pretending the historical source is the current checkout; it never rewrites the
receipt.
