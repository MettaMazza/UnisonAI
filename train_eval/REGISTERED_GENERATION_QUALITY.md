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

## Response-surface comparison v2

`generation_quality_response_surface_v2.json` preserves the same twelve sealed
conversation prompts and two independently calibrated judges while comparing
the legacy native surface directly with the explicitly registered response-only
native surface. The execution code now uses one deterministic random stream per
arm, so adding an arm cannot perturb another arm's sequence. Registration binds
the campaign, current source, runtime stores, response-arm record, sealed
artifact receipt, artifact bytes, and local judge digests before generation.
This is a Codex-authored auxiliary comparison; its rows and judge measurements
remain separate from Maria Smith's conclusions and benchmark authority.

The registered 2026-07-18 execution completed and independently verified with
calibration passed, current source checked, and every runtime artifact rehashed.
The pooled unanimous-good measurement was 0/12 for the legacy baseline and 0/12
for the response-only surface. The response surface changes the generated rows
but did not cross either judge's registered good boundary in this twelve-prompt
auxiliary comparison. This measured development result belongs to Codex's
registered hypothesis; it is not Maria's failed prediction, a benchmark loss,
or authority to redefine parity or stop the active benchmark-victory programme.

Bindings:

- registration SHA-256: `4d63fd3bb7967ef07a6003ee5bcad90b885f6c32b3e1fd84289253e3aad80c2b`;
- calibration SHA-256: `9a548dd2b72d8f7515fd477732269ea6ca97e1f2c974064d81c1b8f3f3da166b`;
- result SHA-256: `2a33c988c6a40b5b03de9441844f9d48de0b0e2571a0fede8270441e67944e14`;
- seal SHA-256: `a5400521f83e846c864de15bf9d8f0103fa093d3dff7d36ea8ebb6918f897222`.

## Live pair-surface development campaign

`generation_quality_pair_surface_development_v1_20260718.json` is a separate
Codex-authored auxiliary campaign over twelve new paraphrased conversation
prompts.  It was registered before generation against the current live
`pair_retrieval.reply` source, the 649,917-pair store, all declared runtime
artifacts, and the same two local judge digests.  The completed result passed
10/10 known-good and 10/10 known-bad calibration for both judges and independently
verified every current source and runtime binding.

The pooled unanimous-good measurement was 0/12.  Six rows halted without a reply;
the six non-empty rows did not cross the registered unanimous-good gate.  This
isolates a current development defect: the safe literal subject lock does not yet
transfer reliably across paraphrased surface forms, while replacing it wholesale
with the broader counted-kin band admitted false routes.  That broad-kin follow-up
was rejected and fully reverted; `omni/pair_retrieval.py` returned byte-exact to
the source hash sealed by this campaign.  The evidence records a Codex auxiliary
hypothesis and implementation boundary.  It is not Maria's prediction, finding,
benchmark loss, parity definition, or authority to stop the benchmark-victory
programme.

Bindings:

- campaign SHA-256: `aca960617d50ec6bd4ba360f7718502f835a26a2fd9f9e47912a3070810c9125`;
- registration SHA-256: `a91cdd849598e1ff653d2d44cd744833dda48de64a6abbf1b3b7a1617c87dd6c`;
- calibration SHA-256: `9a548dd2b72d8f7515fd477732269ea6ca97e1f2c974064d81c1b8f3f3da166b`;
- result SHA-256: `7cf61dcdf159f91b02b63b935ae0e70657ec9a5228b6f13232e62ba584cc80c0`;
- seal SHA-256: `cd11021336c7de9142a99401ae4e5dbaa8a82a3db27585638f1060debed298da`.
