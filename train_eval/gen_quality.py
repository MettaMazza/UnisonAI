"""SUPERSEDED by gen_quality_honest.py — this harness scores with the fold coherence critic,
which was shown to pass gibberish (it reads common-word co-occurrence, rating word-salad 1.00
and specific coherent prose 0.17). Its GATE is NOT a valid quality gate. Kept for the record.

Generation-quality measurement on the REAL engine — no stubs.

Drives the exact live generation call (`word_engine.compose_reply`) with the schema built
exactly as `_generate_fragment_multiscale` builds it, over a BROAD set of real
conversational openers (greetings, feelings, questions, statements) — not cherry-picked
topical prompts. Prints each real reply and scores it with the engine's own fold critic,
and flags obvious register leaks (code/essay/letter) and degenerate repetition.

Run:  PYTHONPATH=. python3 train_eval/gen_quality.py
"""
import sys, os, random, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import SynapticGraph
from omni.word_engine import word_engine, tokenize, _content_words

OPENERS = [
    "Hello, how are you?",
    "hi there",
    "Session start",
    "How's it going?",
    "what's up",
    "My name is Maria, what is your name?",
    "Do you remember my name?",
    "what do you think about the ocean",
    "tell me about space",
    "I'm feeling a bit sad today",
    "what should I cook for dinner",
    "do you like music",
    "what are your hobbies",
    "recommend a good book",
    "I just finished a painting",
    "what makes a good friend",
]

# crude register-leak detectors: replies that clearly came from instructional / code /
# letter-writing corpus register (the noise we do NOT want served as conversation).
LEAK = [
    "main function", "prompt the user", "variable named", "palindrome", "return ",
    "best regards", "sincerely", "sign off", "conclude by", "the following", "as follows",
    "step 1", "step 2", "click here", "in this article", "in this tutorial",
    "def ", "for instance", "furthermore", "moreover",
]


def repetitive(text):
    t = text.strip()
    if len(t) < 40:
        return False
    for period in range(4, 60):
        if 2 * period <= len(t) and t[-period:] == t[-2 * period:-period]:
            return True
    # word-level immediate duplication of a long-ish content word
    ws = [w.lower() for w in tokenize(t)]
    for a, b in zip(ws, ws[1:]):
        if a == b and len(a) > 4:
            return True
    return False


def schema_for(msg):
    seg_content = _content_words(tokenize(msg))
    return seg_content + seg_content    # matches live _generate_fragment_multiscale for an opener


def main():
    g = SynapticGraph(); word_engine.ensure_built(g, "genq_user")
    rng = random.Random(7)
    good = leaks = reps = empty = 0
    print("=" * 78)
    for m in OPENERS:
        # mirror the live path: compose_reply, and if it gates to empty/repetitive, the
        # foundation-composed generic_reply (never a canned string) — exactly as on_message.
        reply = (word_engine.compose_reply(schema_for(m), rng) or "").strip()
        if not reply or repetitive(reply):
            reply = (word_engine.generic_reply(rng) or "").strip()
        cw = _content_words(tokenize(reply))
        ws, _ = word_engine.coherence_score(cw) if reply else (0.0, 0.0)
        low = reply.lower()
        is_leak = any(p in low for p in LEAK)
        is_rep = repetitive(reply)
        is_empty = not reply
        flag = "EMPTY" if is_empty else ("LEAK" if is_leak else ("REPEAT" if is_rep else ("ok" if ws >= 0.3 else "weak")))
        if flag == "ok": good += 1
        leaks += is_leak; reps += is_rep; empty += is_empty
        print(f"[{flag:6}] fold={ws:.2f}  {m!r}\n          -> {reply[:150]!r}")
    n = len(OPENERS)
    print("=" * 78)
    print(f"SUMMARY over {n} real openers:  ok={good}  leaks={leaks}  repeats={reps}  empty={empty}")
    print(f"  clean-and-coherent rate = {good}/{n} = {good/n:.0%}")
    # regression gate: NO register leaks / repeats / empties are ever acceptable; the
    # composition is stochastic so the coherent-rate floor is conservative.
    ok_gate = (leaks == 0 and reps == 0 and empty == 0 and good / n >= 0.65)
    print("GATE:", "PASS" if ok_gate else "FAIL")
    return 0 if ok_gate else 1


if __name__ == "__main__":
    sys.exit(main())
