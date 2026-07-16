"""The learning curve in JUDGED units — supersedes the retracted fold-critic version.

Simulates the live learning law on the REAL path, nothing stubbed:
  round: reply = pair_retrieval.reply(p)  ->  verdict = calibrated judge (the user's 👍/👎)
         -> mark_feedback(verdict) (Laplace counts)
         -> if BAD: the REAL teacher (Gemma) writes the in-persona answer -> add_taught(p, ans)
            (the live correction path — the FAQ law)
Later rounds serve taught pairs for the prompts the engine failed, so the judged rate should
climb — that climb IS the learning law, measured in believed units.

STORE ISOLATION: taught/quality stores are redirected to scratch copies; the live
omni/taught_pairs.pkl and omni/pair_quality.pkl are never written by this harness.

Run:  PYTHONPATH=. python3 train_eval/learning_curve_judged.py
"""
import sys, os, json, re, urllib.request
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import omni.pair_retrieval as PR
from train_eval.judge import judge

# ---- isolate the mutable stores (never touch the live files) ----
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
PR.TAUGHT_PATH = os.path.join(SCRATCH, "curve_taught_pairs.pkl")
PR.QUALITY_PATH = os.path.join(SCRATCH, "curve_pair_quality.pkl")
for p in (PR.TAUGHT_PATH, PR.QUALITY_PATH):
    if os.path.exists(p):
        os.remove(p)
pr = PR.PairRetrieval()          # fresh instance bound to the scratch paths

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "gemma-4-31b:latest"
ROUNDS = 5
PROMPTS = [
    "Hello, how are you?", "how's it going?",
    "My name is Maria, what is your name?",
    "what do you think about the ocean", "tell me about space",
    "I'm feeling a bit sad today", "what should I cook for dinner",
    "do you like music", "what are your hobbies", "recommend a good book",
    "I just finished a painting", "what makes a good friend",
]


def teacher_answer(p):
    """The real teacher's in-persona correction (the live path's ask)."""
    prompt = (f"The user says: {p!r}. Respond as Unison — natural, warm, in your own voice, "
              f"in 1-2 short sentences. No preamble, just the reply.")
    body = json.dumps({"model": TEACHER, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0.7}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=180) as r:
            out = json.loads(r.read().decode()).get("response", "")
        out = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", out).strip()
        return out.split("\n")[0][:300]
    except Exception:
        return ""


def main():
    print(f"round | judged GOOD  (learning law: judge=feedback, teacher corrects BADs)")
    print("-" * 70)
    curve = []
    for rnd in range(1, ROUNDS + 1):
        good = 0
        for p in PROMPTS:
            reply = (pr.reply(p) or "").strip()
            ok, _ = judge(p, reply)
            good += ok
            pr.mark_feedback(ok)
            if not ok:
                # the teacher supplies TWO phrasings of the correction — one meaning,
                # multiple expressions; serving re-expresses by cross-variant composition
                ans = teacher_answer(p)
                ans2 = teacher_answer(p)
                ans3 = teacher_answer(p)
                if ans:
                    pr.add_taught(p, ans, variants=[a for a in (ans2, ans3) if a and a != ans])
        curve.append(good)
        print(f"  {rnd}   |  {good}/{len(PROMPTS)} = {good/len(PROMPTS):.0%}")
    print("-" * 70)
    verdict = ("LEARNS" if curve[-1] > curve[0]
               else ("HOLDS" if curve[-1] == curve[0] else "DEGRADES"))
    print(f"curve: {curve}  VERDICT: {verdict}")
    with open(os.path.join(SCRATCH, "learning_curve_judged.jsonl"), "a") as f:
        f.write(json.dumps({"curve": curve, "n": len(PROMPTS), "verdict": verdict}) + "\n")


if __name__ == "__main__":
    main()
