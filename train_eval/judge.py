"""The shared honest judge — the project's ONLY conversational scoreboard (TRANSLATION_PLAN
Stage 0). Independent local model, temperature 0, strict GOOD/BAD on grammar + coherence +
relevance. It takes no part in steering generation, and its numbers count only after
`judge_calibration.py` passes ([[honest-evaluation-first]])."""
import json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma-4-31b:latest"
MODEL2 = "qwen3.6-27b:latest"   # judge #2 — the pool (no verdict depends on one model)


def judge2(user_msg, reply, timeout=180):
    """Judge #2 (independent family). Same strict GOOD/BAD contract."""
    return _judge_with(MODEL2, user_msg, reply, timeout)


def judge_pool(user_msg, reply, timeout=180):
    """The POOL verdict (the paper's own design): GOOD only when BOTH independent judges
    agree GOOD; disagreement counts as not-good and is recorded. The two judges are
    independent models — they run CONCURRENTLY."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        fa = ex.submit(judge, user_msg, reply, timeout)
        fb = ex.submit(judge2, user_msg, reply, timeout)
        a, _ = fa.result()
        b, _ = fb.result()
    return (a and b), ("agree" if a == b else "split")


def judge(user_msg, reply, timeout=180):
    return _judge_with(MODEL, user_msg, reply, timeout)


def _judge_with(model, user_msg, reply, timeout=180):
    """Return (good: bool, raw_tail: str). Empty replies are BAD by definition."""
    if not reply or not reply.strip():
        return False, "empty"
    prompt = (
        "You are grading a chatbot reply for basic conversational quality. Be strict.\n"
        f"User said: {user_msg!r}\n"
        f"Chatbot replied: {reply!r}\n\n"
        "GOOD = grammatical, coherent English AND a relevant, sensible response to what the user said.\n"
        "BAD = word-salad, gibberish, a broken fragment, wrong register (code, movie-script lines, "
        "letter closings), or a non-sequitur / off-topic reply.\n"
        "Answer with exactly one word: GOOD or BAD.")
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "think": False, "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode()).get("response", "")
        out = re.sub(r"(?is)<think>.*?</think>", "", out).upper()
        m = re.findall(r"\b(GOOD|BAD)\b", out)
        return (m[-1] == "GOOD" if m else False), out.strip()[:40]
    except Exception as e:
        return False, f"err:{e}"
