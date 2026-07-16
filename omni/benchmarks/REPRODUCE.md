# REPRODUCE — the headline results, four commands

Every headline claim of the UnisonAI campaign is reproducible on a fresh
clone with the commands below. Each command prints its own verdict; the
expected output is stated beside it. Reference machine: Apple Silicon
Mac Studio, 512 GB unified memory, macOS — runtimes are from that
machine and scale with hardware.

## 1. The theory prover — 307 suites, 1,844 forced checks, 0 failures

```
cd "Smithian Fold Theory"
sh RUN_EVERYTHING.sh
```

Runs the ENTIRE verification surface of the Smithian Fold Theory: the
full proof corpus from the axiom to every measured comparison (C
compiler only), the tamper test (mutate one forced constant and PROVE
the engine halts), the hardened fine-structure uniqueness scan, and the
live CODATA/NIST comparison over the network. Expected: every section
that runs reports PASS; the final manifest lists exactly what ran.
Needs: a C compiler; python3 for sections 3–6; network for section 4.

## 2. LLM presence — 13/13 tensors, 39/39 registered checks

```
cd "Smithian Fold Theory/fold_ai"
python3 llm_presence.py
```

The pre-registered presence protocol (5 seeded shuffle-nulls +
moment-matched Gaussian + theorem-forced bit-reversal self-test per
tensor, seed 20260706) applied to public GPT-2: token embedding + all
12 MLP expansion matrices. **If `gpt2_model.safetensors` is absent the
script fetches it itself** (~548 MB, official Hugging Face repository)
— one command on a fresh clone. Expected: `13/13 tensors, 39/39
registered checks`, margins 3.4–79.3×. Needs: `numpy`, `safetensors`;
network for the one-time fetch. Runtime: ~2 minutes + the fetch.

## 3. Rung 5c, the loud-subspace transfer test — verdict SUPPORTED

```
cd "Smithian Fold Theory/fold_ai"
python3 rung5c_loud_subspace.py
```

Pre-registered (protocol fixed in the file header before data): the
trained twin's advantage must live in its dyadically-loud Walsh
subspace and nowhere else. Two theorem-forced self-tests must pass
in-run or the run is void. Expected: `VERDICT (pre-registered rule):
SUPPORTED`, loud beating random at every budget k ∈ {16, 32, 64} and
the attention control weaker. Needs: `torch`, `numpy`. NOTE: the arena
reads the theory corpus at its committed absolute paths
(`~/Desktop/Smithian Fold Theory`, `~/Desktop/SFTOM`); clone both repos
to those locations or the corpus (and the exact numbers) will differ —
the committed result file `rung5c_results.txt` is the reference.
Runtime: ~10–20 minutes (twin training dominates).

## 4. The engine end-to-end — 36/36

```
cd "Smithian Fold Theory/fold_ai"
python3 verify_unison.py
```

The full engine verification: forced-lock enforcement (a fitted value
HALTS the engine), learning-law closure, rebirth persistence across
process death, the fold eye/ear/voice with integer Parseval
self-certification, the removal-proof ladders, the agentic toolkit, and
the benchmark instruments. Expected: `36/36`. Needs: `numpy`; several
checks exercise the live observer ladder and speech — `ollama` with
`gemma4:26b` pulled, and the kokoro voice at its configured path, for
the full 36; absent organs report their checks honestly rather than
silently passing.

---

Results files committed beside the scripts: `llm_presence_results.txt`,
`rung5c_results.txt`, `rung5b_results.txt`, `rung5b_rematch_results.txt`,
`rung5d_results.txt`, `verify_unison_results.txt`, `benchmarks.tsv`,
`benchmarks_sota.tsv`, `SOTA_TABLE.md`.
