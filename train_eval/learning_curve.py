"""Learning-loop verification — does feedback -> reinforcement actually MOVE coherence?

The claim under test (Stage 3): when a reply is coherent the foundation spans + content
couplings that made it are reinforced, so retrieval increasingly prefers what works and
generation gets MORE coherent with use. This measures that curve on the REAL engine.

Method (no stubs, no teacher, no GPU): for each topical prompt, sample K replies per round.
The fold critic (coherence_value.ep — the engine's own value, exactly the sovereign signal)
labels each good/bad; good replies reinforce their spans+couplings, bad ones weaken them.
Track mean coherence per round. If the loop works, the curve rises (or holds high) and the
coherent-fraction increases — the engine is learning which foundation language works.

SAFETY: reinforces the singleton's IN-MEMORY stores only and NEVER calls save_*, so the
live bot's word_coupling.pkl / span_quality.pkl on disk are untouched. Separate process,
read-only on disk.

Run:  PYTHONPATH=. python3 train_eval/learning_curve.py
"""
import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import SynapticGraph
from omni.word_engine import word_engine, tokenize, _content_words

PROMPTS = [
    "what do you think about the ocean",
    "tell me about space",
    "what should I cook for dinner",
    "do you like music",
    "what are your hobbies",
    "I'm feeling a bit sad today",
    "recommend a good book",
    "what makes a good friend",
]
ROUNDS = 8
SAMPLES = 6          # replies sampled per prompt per round
GOOD = 0.40          # fold-coherence threshold for a "good" reply (reinforce)


def schema_for(msg):
    c = _content_words(tokenize(msg))
    return c + c


def main():
    # compose_reply needs only the retrieval / coupling / span stores (loaded lazily) — NOT
    # the word-orbit build, whose multiprocessing pool stalls here. Warm the stores directly.
    word_engine._load_retrieval(); word_engine._load_coupling(); word_engine._load_span_quality()
    assert hasattr(word_engine, "reinforce_spans")
    rng = random.Random(11)

    print("round |  mean_fold  coherent_frac  (reinforcing good spans/couplings, NO save)")
    print("-" * 68)
    trajectory = []
    for r in range(ROUNDS):
        scores = []
        for p in PROMPTS:
            for _ in range(SAMPLES):
                reply = (word_engine.compose_reply(schema_for(p), rng) or "").strip()
                cw = _content_words(tokenize(reply))
                sc, _ = word_engine.coherence_score(cw) if reply else (0.0, 0.0)
                scores.append(sc)
                # THE LEARNING LOOP (in-memory only): good -> reinforce, bad -> weaken.
                if reply:
                    if sc >= GOOD:
                        word_engine.reinforce_spans(True)
                        word_engine.reinforce_couplings(cw)
                    else:
                        word_engine.reinforce_spans(False)
                        word_engine.weaken_couplings(cw)
        mean = sum(scores) / len(scores)
        frac = sum(1 for s in scores if s >= GOOD) / len(scores)
        trajectory.append(mean)
        print(f"  {r:2d}  |   {mean:.3f}      {frac:.0%}")

    # verdict: coherence should not degrade, and should trend up over the session
    first, last = trajectory[0], trajectory[-1]
    peak = max(trajectory)
    rose = last >= first - 0.02 and peak >= first     # non-degrading + reaches >= start
    improved = last > first + 0.01
    print("-" * 68)
    print(f"start={first:.3f}  end={last:.3f}  peak={peak:.3f}  delta={last-first:+.3f}")
    print("VERDICT:", "IMPROVES" if improved else ("HOLDS (non-degrading)" if rose else "DEGRADES — loop not helping"))
    # confirm no accidental persistence flag left dirty-and-saved (we never call save_*)
    print("(disk stores untouched: save_* never called)")
    return 0 if rose else 1


if __name__ == "__main__":
    sys.exit(main())
