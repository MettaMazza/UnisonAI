"""Split papers/Unison_AI_Architecture.md (v4.7) into two cross-linked papers, preserving
Maria's prose verbatim by line-slicing. v4.7 remains committed as the dated combined record.

A: papers/UnisonAI_Architecture.md          — the pure architecture paper
B: papers/Fold_Decode_Interpretability.md   — the interpretability / fold-decode paper
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "papers", "Unison_AI_Architecture.md")
A_OUT = os.path.join(HERE, "..", "papers", "UnisonAI_Architecture.md")
B_OUT = os.path.join(HERE, "..", "papers", "Fold_Decode_Interpretability.md")

lines = open(SRC, encoding="utf-8").read().splitlines()

def sl(a, b):
    """1-indexed inclusive slice."""
    return "\n".join(lines[a - 1:b])

HEAD_A = """# UnisonAI: A Forced, Derived Omni-Model Architecture with Zero Parameters

## *Attention, it turns out, was not all you need*

**Maria Smith (Ernos Labs)** — the architecture paper, v5.0, 2026-07-16
Split from the combined record *UnisonAI: A Forced, Derived Omni-Model Architecture* (v4.x
lineage, DOI 10.5281/zenodo.21217279), which remains committed as the dated record. The
spectral science this engine stands beside — the decode of trained neural weights — is the
companion paper **[The Law Inside Trained Weights: The Fold Decode Campaign](Fold_Decode_Interpretability.md)**.

> **THIS PAPER IS A PROOF, NOT THE MAIN EVENT.** Everything here derives from **The Smithian
> Fold Theory of Everything**: one machine-checked, self-proven theorem — *there is no nothing* —
> zero axioms, zero free parameters, machine-verified by 326 proof suites and 2,002 forced checks.
> **Theory (Zenodo):** https://doi.org/10.5281/zenodo.21182469 · **Theory (GitHub):** https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything
> **This engine (GitHub):** https://github.com/MettaMazza/UnisonAI · **Companion decode paper:** [Fold_Decode_Interpretability.md](Fold_Decode_Interpretability.md)

---

## Abstract
"""

ABSTRACT_A_POINTER = """
**The spectral science** — the pre-registered decode of trained neural weights (presence,
location, recipe map to one trillion parameters, the loud-band function attribution, the
transfer verdicts of Rungs 5c/5d/5e, and the forced two-family law) — is the companion
paper, [The Law Inside Trained Weights](Fold_Decode_Interpretability.md); its results are
cited here where the architecture stands on them.
"""

UPDATE_82 = """
#### 8.2.1 The conversational generator — architecture update (2026-07-16)

The live conversational surface described above (the per-token walk over held distributions)
was superseded in the engine after this record's measurements: an unfiltered token walk and,
separately, verbatim orbit replay were both retired — the first for incoherence, the second
because **verbatim recall is a violation of the architecture's law**. The current live
generator is the corpus's established-mathematics translation, all counted and exact:
**prompt→response pair retrieval** over role-structured conversational corpora (BM25 with
question-question Jaccard, dialogue-act matching, counted-kinship expansion at the fold
factor 1/2, conversation context weighted 2^-age — the forced halving), **Laplace
(g+1)/(g+b+2) feedback re-ranking**, taught corrections as pairs (the FAQ law), and
**delexicalize→relexicalize realization** (the tool-trace template law generalized) under a
never-verbatim guard. Its quality is scored ONLY by a calibration-gated independent judge —
see §9's 2026-07-16 addendum for the honest measured record, including the retraction that
preceded it. The exact-fractional counted substrate of this section remains the
cross-entropy-measured foundation and the memory law is unchanged.
"""

UPDATE_89 = """
> **[Update 2026-07-16]** The current teacher relay captures reasoning via an explicit
> `<thinking>…</thinking>` output block with native think disabled — the formatted block
> proved far more steerable than the hidden reasoning field. The thought channel, STaR
> gating, and full-transparency streaming are unchanged.
"""

ADDENDUM_9 = """
### 9.1 The honest conversational record — addendum of 2026-07-16

This addendum replaces every conversational-quality number previously reported from the
engine's own fold critic. That critic was found to reward common-word co-occurrence (rating
word-salad 1.00 and specific coherent prose 0.17) while also steering generation — a proxy
measuring itself. The numbers it produced were **retracted** (engine repo, commit `5cc3786`),
and an instrument standard was adopted: **no signal that steers generation may serve as its
scoreboard, and every judge must pass a calibration gate** (clean separation of known-good
from known-bad replies) **before its numbers are believed.**

Under the calibrated independent judge (gate: 10/10 known-good, 10/10 known-bad):

| conversational generator | judged GOOD (16 real openers) |
|---|---|
| retired splicer over loose sentences | 0/16 |
| true role-structured pairs, first ranking | 1/16 |
| + Jaccard/act/serve-gate/relexicalization | 2/16 (one recorded regression to 1/16 was caught and reverted) |
| + role-correct casual-dialogue sources (544k pairs) | **3/16 (19%)** |

The trajectory — including its regression — is the finding: a calibrated ruler, honest
engineering against it, and mechanisms improved only where the judged rows dictated. The
identity-question class is unanswerable from an anonymous corpus by construction and
converts through the live taught-pair loop; a head-to-head against a 35B-parameter model
produced a **void** win-rate (the opponent harness truncated its reasoning) and is not
reported — the engine's absolute judged score on that wider set was 17%, consistent. All
harnesses, trajectories, and the full Empirical Record are committed in the engine
repository (`omni/README.md`, `train_eval/`).
"""

HEAD_B = """# The Law Inside Trained Weights: The Fold Decode Campaign

## *A registered, null-controlled spectral decode of gradient-trained neural networks*

**Maria Smith (Ernos Labs)** — the interpretability paper, v1.0, 2026-07-16
Split from the combined record *UnisonAI: A Forced, Derived Omni-Model Architecture* (v4.x
lineage, DOI 10.5281/zenodo.21217279), which remains committed as the dated record. The
zero-parameter engine these results feed — and the architecture that stands on them — is the
companion paper **[UnisonAI: A Forced, Derived Omni-Model Architecture](UnisonAI_Architecture.md)**.

> Everything here derives from and is verified against **The Smithian Fold Theory of
> Everything**: one machine-checked, self-proven theorem — *there is no nothing* — zero axioms,
> zero free parameters, 326 proof suites, 2,002 forced checks.
> **Theory (Zenodo):** https://doi.org/10.5281/zenodo.21182469 · **Theory (GitHub):** https://github.com/MettaMazza/Smithian-Fold-Theory-Of-Everything
> **Instruments & ledgers (GitHub):** https://github.com/MettaMazza/UnisonAI (`omni/benchmarks/`) · **Companion architecture paper:** [UnisonAI_Architecture.md](UnisonAI_Architecture.md)

---

## Abstract
"""

# ---- Paper A ----
a = [HEAD_A,
     sl(18, 18), "", ABSTRACT_A_POINTER, sl(24, 28), "",
     "---", "", sl(32, 40), "",                         # §1 the field began with law
     sl(179, 195), UPDATE_82, "",                        # §8 intro + 8.1-8.2 + update
     sl(197, 220), "",                                   # 8.3-8.8
     sl(221, 223), UPDATE_89, "",                        # 8.9 + update
     sl(225, 235), "",                                   # 8.10-8.12
     sl(237, 257), ADDENDUM_9, "",                       # §9 + honest addendum
     sl(259, 273), "",                                   # §10-§12
     "## 13. Discussion", "", sl(279, 281), "",          # the architecture-facing discussion
     sl(283, 289), "",                                   # §14 + references
     "*Architecture paper v5.0 (2026-07-16), split from the v4.7 combined record with its "
     "prose preserved; §8.2.1, the §8.9 update note, and §9.1 are the additions — the honest "
     "conversational record and the mechanisms as they run today. Every other number is from "
     "the committed, timestamped campaign records of the v4.x lineage.*"]

# ---- Paper B ----
b = [HEAD_B,
     sl(20, 22), "", sl(28, 28), "",
     "---", "", sl(32, 44), "",                          # §1 + §2 the instrument
     sl(46, 177), "",                                    # §3-§7.12 the whole decode
     "## Discussion", "", sl(277, 277), "", sl(281, 281), "",
     sl(283, 289), "",
     "*Decode paper v1.0 (2026-07-16), split from the v4.7 combined record with its prose "
     "preserved verbatim. Every number is from committed, timestamped campaign records; "
     "nothing is projected.*"]

open(A_OUT, "w", encoding="utf-8").write("\n".join(a) + "\n")
open(B_OUT, "w", encoding="utf-8").write("\n".join(b) + "\n")
print(f"A: {A_OUT} ({os.path.getsize(A_OUT)} bytes)")
print(f"B: {B_OUT} ({os.path.getsize(B_OUT)} bytes)")
