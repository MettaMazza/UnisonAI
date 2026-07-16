"""HONEST generation-quality measurement — judged by a real model, NOT the fold critic.

The fold `coherence_score` is a hackable proxy (it rewards common-word co-occurrence, so it
rates gibberish 1.00 and specific coherent prose 0.17) and it ALSO steers generation, so
measuring with it is a ruler measuring itself. This harness judges each reply with a real
model (Gemma via Ollama) on actual coherence + relevance — an independent signal that reads
semantics — and reports the real good-rate. It also prints the fold critic's score beside the
honest verdict so the gap between the proxy and reality is visible.

Run:  PYTHONPATH=. python3 train_eval/gen_quality_honest.py   (needs Ollama + gemma-4-31b)
"""
import sys, os, json, re, urllib.request, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import word_engine, tokenize, _content_words
from omni.pair_retrieval import pair_retrieval
from train_eval.judge import judge as _judge

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma-4-31b:latest"

OPENERS = [
    "Hello, how are you?", "hi there", "how's it going?", "what's up",
    "My name is Maria, what is your name?", "Nice to meet you, do you remember my name?",
    "what do you think about the ocean", "tell me about space",
    "I'm feeling a bit sad today", "what should I cook for dinner",
    "do you like music", "what are your hobbies", "recommend a good book",
    "I just finished a painting", "what makes a good friend",
    "Nothing much I'm just chilling, what's on your mind?",
]


def judge(user_msg, reply):
    """Real-model verdict: is `reply` a coherent, relevant, grammatical conversational reply
    to `user_msg`? Returns (good_bool, one_word_reason)."""
    prompt = (
        "You are grading a chatbot reply for basic quality. Be strict.\n"
        f"User said: {user_msg!r}\n"
        f"Chatbot replied: {reply!r}\n\n"
        "Is the reply GOOD (grammatical, coherent English AND a relevant response to the user) "
        "or BAD (gibberish, word-salad, a non-sequitur, off-topic, or a broken fragment)?\n"
        "Answer with exactly one word: GOOD or BAD.")
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "think": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read().decode()).get("response", "")
        out = re.sub(r"(?is)<think>.*?</think>", "", out).upper()
        m = re.findall(r"\b(GOOD|BAD)\b", out)
        return (m[-1] == "GOOD" if m else False), out.strip()[:20]
    except Exception as e:
        return False, f"err:{e}"


def schema_for(msg):
    c = _content_words(tokenize(msg)); return c + c


def main():
    word_engine._load_retrieval(); word_engine._load_coupling(); word_engine._load_span_quality()
    rng = random.Random(3)
    good = 0
    rows = []
    print("verdict | fold  | reply")
    print("-" * 78)
    for m in OPENERS:
        reply = (word_engine.compose_reply(schema_for(m), rng) or "").strip()
        if not reply:
            reply = (word_engine.generic_reply(rng) or "").strip()
        cw = _content_words(tokenize(reply))
        fold, _ = word_engine.coherence_score(cw) if reply else (0.0, 0.0)
        ok, reason = judge(m, reply)
        good += ok
        rows.append({"prompt": m, "reply": reply, "fold": round(fold, 2), "judge_good": ok})
        print(f"{'GOOD' if ok else 'BAD ':4} | {fold:.2f}  | {reply[:64]!r}")
    n = len(OPENERS)
    print("-" * 78)
    print(f"HONEST good-rate (real-model judge): {good}/{n} = {good/n:.0%}")
    fold_ok = sum(1 for r in rows if r['fold'] >= 0.3)
    print(f"fold-critic would have said 'ok' (>=0.3): {fold_ok}/{n} = {fold_ok/n:.0%}  <- the proxy's inflation")
    with open(os.path.join(os.path.dirname(__file__), "..", "logs", "gen_quality_honest.jsonl"), "a") as f:
        f.write(json.dumps({"good_rate": good / n, "fold_ok_rate": fold_ok / n, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
