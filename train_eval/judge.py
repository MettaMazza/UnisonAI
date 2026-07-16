"""The shared honest judge — the project's ONLY conversational scoreboard (TRANSLATION_PLAN
Stage 0). Independent local model, temperature 0, strict GOOD/BAD on grammar + coherence +
relevance. It takes no part in steering generation, and its numbers count only after
`judge_calibration.py` passes ([[honest-evaluation-first]])."""
import json, re, urllib.request

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "gemma-4-31b:latest"


def judge(user_msg, reply, timeout=180):
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
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
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
