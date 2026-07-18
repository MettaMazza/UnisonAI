# Generation boundary provenance

This ledger separates the engine's Step 322–324 identities from constitutional
implementation and current engineering.  The engine owns validity: duplicated
literals or prose do not create a second authority.

## Forced identities

- Step 322 coherence lock: `FOCUS_LOCK = 1/2`.
- Step 323 context depth: `INTEGRATION_DEPTH = b + c = 5`.
- Step 323 cascade factor: `cascade_share(1) = 1/2`, with the deepest floor
  repeated so the five-level cascade closes exactly to the One.
- Step 324 admission and binding locks: `SPREAD_LOCK = FOCUS_LOCK = 1/2`.
- Step 324 fresh cross-variant support: `REEXPRESS_MIN = b = 2`.

The canonical quantities remain in `omni/core.py`.  Active generation code uses
the executable operations in `omni/generation_boundaries.py`; it must not repeat
these boundaries as local numeric literals.

## Constitutional mechanisms

- Experience is held as exact counts and addressed deterministically.
- Retrieval produces counted continuation distributions rather than replaying a
  stored response surface.
- Context is integrated through the five-stage dyadic cascade.
- Generation is constrained by the admitted half-One context and by counted plan
  coverage.
- Fresh taught composition requires two independently held variants.

## Engineering quantities

These remain disclosed implementation status until separately forced or
constitutionally re-derived: `TOPK_POS`, maximum/minimum word limits, beam width,
beam expansion, `n_best`, candidate caps, closure multiplier, tie noise,
paraphrase bounds, and the legacy conversational fallback threshold `0.30`.

This provenance patch makes the generation boundary executable. It does not
declare a conversational-quality result or replace the independent judge. The
sealed 2026-07-18 campaign is a Codex-authored auxiliary benchmark and records
baseline 0/12, F1 0/12, and F3 0/12 under two-judge unanimity after perfect
same-run calibration. It is not Maria's prediction, finding, loss, parity
definition, or authority over a real run. The response-only fluency artifact is
a selectable development surface. Maria Smith alone decides when its evidence
supports a publishable conclusion or when it enters a real benchmark run.

## Explicit response-only surface

`WordEngine(fluency_path=...)` and `configure_fluency_store(...)` now provide an
explicit native route to a sealed response-only fluency artifact without
renaming or replacing the live legacy store. An explicitly selected artifact
must declare schema `unison-response-fluency/v1`, role `assistant-response`, and
boundary policy `reset-before-and-after-every-response`; any mismatch halts its
activation and leaves an empty surface. `fluency_identity()` reports the exact
path and SHA-256 together with the declared provenance. Selection does not make
the artifact the default and carries no agent-authored benchmark conclusion or
parity definition.

The live Discord runtime now accepts `UNISON_FLUENCY_ARM` as an explicit arm
selection. `train_eval/response_fluency_runtime_arm_v1.json` binds the sealed
receipt and 318,595,101-byte artifact; the loader rechecks both SHA-256 values,
the response-only schema, role, and boundary policy before generation can use
it. With no explicit arm, the live default is unchanged. Runtime selection is
an implementation route, not agent authority over benchmark timing or results.
The committed arm was exercised directly: it loaded 93,098 words at exact
orders 1–4 and returned artifact SHA-256
`01bf496745d109a04d1983c78ed298c180b970aaf64ed30da32600a956b1d8b2` with
the registered response role and boundary policy.
