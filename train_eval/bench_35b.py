"""PHASE A/Stage 5 — head-to-head: UnisonAI (pair retrieval + relexicalization) vs Qwen3.5-35B.

Both systems answer the same held-out conversational prompts; BOTH replies are judged by the
CALIBRATED independent judge (train_eval/judge.py — Stage-0 gate PASSED 10/10|10/10). No
uncalibrated pairwise judging (the earlier Qwen-as-judge produced a void 6-6 on gibberish and
was retracted). Win = engine GOOD & 35B BAD; tie = same verdict; win-rate = wins + ties/2.

Standalone/offline: does NOT touch the live bot. Usage: PYTHONPATH=. python3 train_eval/bench_35b.py
"""
import os, sys, re, json, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.pair_retrieval import pair_retrieval
from train_eval.judge import judge

QWEN = os.path.expanduser("~/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q8_0.gguf")
OUT = os.path.join(os.path.dirname(__file__), "..", "logs", "bench_35b.jsonl")

PROMPTS = [
    "what do you think about the ocean",
    "how are you feeling today",
    "what should I cook for dinner",
    "do you like music",
    "tell me about space",
    "what are your hobbies",
    "explain photosynthesis simply",
    "recommend a good book",
    "what's your favorite movie",
    "how do I stay motivated",
    "tell me something interesting",
    "what makes a good friend",
]


def engine_reply(p):
    return (pair_retrieval.reply(p) or "").strip()


def _strip_think(t):
    if "</think>" in t:
        t = t.split("</think>")[-1]
    return re.sub(r"(?is)</?think>", "", t).strip()


def qwen_reply(llm, p):
    try:
        out = llm.create_chat_completion(messages=[
            {"role": "system", "content": "You are a warm, friendly conversational assistant. "
             "Reply directly in 1-3 short natural sentences."},
            {"role": "user", "content": p}], max_tokens=2600, temperature=0.7)
        return _strip_think(out["choices"][0]["message"]["content"])
    except Exception as e:
        return f"(qwen error: {e})"


def opponent_valid(r):
    """OPPONENT-INTEGRITY: a row counts only if the 35B's reply is a real reply, not leaked
    or truncated reasoning (the earlier run's win-rate was voided for exactly this)."""
    if not r or r.startswith("(qwen error"):
        return False
    low = r.lower()
    return not any(m in low for m in ("thinking process", "**analyze", "drafting the response",
                                      "1.  **", "internal monologue"))


def main():
    from llama_cpp import Llama
    print("loading Qwen3.5-35B …", flush=True)
    llm = Llama(model_path=QWEN, n_ctx=2048, n_gpu_layers=-1, verbose=False)

    rows, wins, ties, e_good, q_good, n_valid = [], 0, 0, 0, 0, 0
    for p in PROMPTS:
        er, qr = engine_reply(p), qwen_reply(llm, p)
        valid = opponent_valid(qr)
        eg, _ = judge(p, er)
        qg, _ = judge(p, qr) if valid else (False, "invalid")
        e_good += eg
        if valid:
            n_valid += 1; q_good += qg
            if eg and not qg:
                wins += 1
            elif eg == qg:
                ties += 1
        rows.append({"prompt": p, "engine": er[:200], "qwen": qr[:200],
                     "engine_good": eg, "qwen_good": qg, "opponent_valid": valid})
        print(f"\nPROMPT: {p}\n  ENGINE [{'G' if eg else 'B'}]: {er[:100]}\n"
              f"  QWEN   [{('G' if qg else 'B') if valid else 'VOID'}]: {qr[:100]}", flush=True)

    n = len(PROMPTS)
    winrate = ((wins + 0.5 * ties) / n_valid) if n_valid else None
    summary = {"ts": datetime.datetime.utcnow().isoformat() + "Z", "n": n, "n_valid": n_valid,
               "engine_good": e_good, "qwen_good": q_good,
               "engine_winrate_vs_35b": (round(winrate, 3) if winrate is not None else None),
               "wins": wins, "ties": ties,
               "judge": "calibrated gemma GOOD/BAD (Stage-0 gate passed); opponent-integrity checked"}
    print("\n" + "=" * 60)
    print(f"ENGINE vs Qwen3.5-35B over {n} prompts (calibrated judge; {n_valid} opponent-valid):")
    print(f"  judged GOOD:  engine {e_good}/{n}  |  35B {q_good}/{n_valid} (valid rows)")
    if winrate is not None and n_valid >= 8:
        print(f"  ENGINE WIN-RATE vs 35B: {winrate:.0%} over {n_valid} valid rows  (parity = 50%)")
    else:
        print(f"  WIN-RATE: VOID — only {n_valid} opponent-valid rows (< 8); not reported")
    with open(OUT, "a") as f:
        f.write(json.dumps({"summary": summary, "rows": rows}) + "\n")
    print(f"  -> logged to {OUT}")


if __name__ == "__main__":
    main()
