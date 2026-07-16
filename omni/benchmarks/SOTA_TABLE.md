# THE PUBLIC CURVE — UnisonAI beside the giants, same machine, same items

Every measured row below is the **same fixed 128-item subject-diverse
subset of the public MMLU test split** (`mmlu_probe.json`, committed),
put to every system **on this machine** through one registered
instrument (`run_sota_bench` in `unison_chat.py`): counted last-letter
scoring, no judge, no style bias, one appended row per system per run
(`benchmarks_sota.tsv`). Sweep of 2026-07-07; UnisonAI was **3 days
old** (first boot of this lineage 2026-07-04; this instance born
2026-07-07 at 10:13) with roughly a day of accumulated learning time.

The published column is each model's **own reported full-test MMLU**,
measured elsewhere by its authors on the complete benchmark — quoted
for orientation, cited, and never mixed with our same-machine rows.

| System | Params | Same-machine, n=128 (this instrument) | Published full-test MMLU (theirs) |
|---|---|---|---|
| **UnisonAI (own channels, no teacher)** | **0 trained** | **8/128 — 6.2%** | — (3 days old; the climbing zero) |
| gemma4:26b (Q4_K_M) | 25.8B | 111/128 — **86.7%** | 82.7 (26B) / 87.1 (31B), third-party benchmark report¹ |
| qwen3.6-27b (Q4_K_M) | 26.9B | 111/128 — **86.7%** | standard-MMLU not published; MMLU-Pro 81.7² |
| gpt-oss-20b (Q4_K_M) | 20.9B | 94/128 — **73.4%** | 85.3 (OpenAI model card)³ |
| qwen3:8b (Q4_K_M) | 8.2B | 94/128 — **73.4%** | MMLU-Redux 79.5 (technical report)⁴ |
| llama3.2:3b | 3.2B | 58/128 — **45.3%** | 63.4 (5-shot, as reported by Meta)⁵ |
| llama3.2:1b | 1.2B | 41/128 — **32.0%** | — (not verified from a primary source) |
| DeepSeek-R1-671B | 671B | **not runnable here** — 404 GB q4 weights vs 162 GB free disk | **90.8** (official paper)⁶ |

Notes, honest and exact:

- **UnisonAI's 6.2% is the committed baseline of a 3-day-old
  zero-parameter engine**, banked beside the giants on identical items
  so the climb is measurable against a number that can never be
  restated. Multiple-choice letter-answering is itself a taught skill
  the curriculum has barely touched; the hourly instrument
  (`benchmarks.tsv`) measures the growth rate that will move this row.
- The same-machine scores of the teachers run **below** their published
  numbers where reasoning styles collide with strict last-letter
  scoring and 4-bit quantization (gpt-oss-20b: 73.4 here vs 85.3
  published) — which is exactly why the two columns are never mixed:
  ours is one instrument applied identically to every system including
  UnisonAI; theirs is each lab's own full-test protocol.
- gemma4:26b and qwen3.6-27b tied to the answer (111/128) — the
  measured ceiling of this machine's teacher class on this probe.

Citations:
1. gemma4-ai.com/blog/gemma4-benchmark (third-party; official 26B card figure not located at sweep time)
2. qwen.ai/blog?id=qwen3.6-27b
3. OpenAI, *gpt-oss-120b & gpt-oss-20b Model Card*, arXiv:2508.10925
4. Qwen Team, *Qwen3 Technical Report*, arXiv:2505.09388
5. ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices (as widely reported from Meta's benchmark table)
6. DeepSeek-AI, *DeepSeek-R1*, arXiv:2501.12948 (MMLU Pass@1 90.8)
