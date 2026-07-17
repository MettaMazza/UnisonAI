# Reproduce the current UnisonAI release

This guide distinguishes current executable receipts from historical campaign artifacts. Reference machine: Apple M3 Ultra with 512 GB unified memory.

## 1. Current standalone structural receipt — 21/21

From the UnisonAI repository root:

```sh
python3 omni/benchmarks/verify_omni.py
```

Expected final line:

```text
VERIFY OMNI: 21/21 checks pass
```

The suite covers forced locks, halt-on-mismatch, integer-Walsh sight and hearing, held memory, rational shares, diagnostics, removal-proof voice, and persistence across process death. Optional local voice assets are machine IO dependencies, not SFT quantities.

## 2. Main SFT corpus — 326 suites / 2,002 checks

From the main Smithian Fold Theory repository:

```sh
./verify/prove_current_source_isolated.sh
```

Expected receipt:

```text
CURRENT_SOURCE_COMPLETE suites=326 checks=2002 failures=0
CERTIFICATE_COMPARE identical=326 drifted=0 absent=0 total=326
```

This isolated gate does not modify the committed certificate tree.

## 3. GPT-2 spectral presence — historical registered campaign

```sh
python3 omni/benchmarks/llm_presence.py
```

Expected committed campaign result: 13/13 tensors and 39/39 registered real-versus-null checks. The instrument may fetch public GPT-2 weights if absent; model weights are runtime dependencies and must not be committed to this repository.

## 4. Decode campaign

Each instrument, dependency, registration rule, null, and raw receipt is documented in `INTERPRETABILITY.md`. Results are committed as `*_results.txt`, with registrations in `registrations.jsonl` and structured rows in `results.jsonl`.

Do not turn an absent optional model, local corpus path, or stopped long run into a silent pass. Preserve the exact output and scope the result to the instrument that ran.

## Historical receipt correction

`omni/benchmarks/verify_unison_results.txt` records **43/47**, not 47/47. It is a historical suite and is not substituted for the current 21/21 verifier. The two receipts test different surfaces.
