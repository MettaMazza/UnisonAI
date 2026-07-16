import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("uc", "unison_chat.py")
uc = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(uc)

# toggles
print("toggle /auto     ->", uc.toggle("/auto"))
print("toggle / teach   ->", uc.toggle("/ teach"))
print("toggle /self play->", uc.toggle("/self play"))
print("flags:", uc.AUTO)

# one self-play batch, run directly (no thread, no sleep)
import numpy as np, random
rng = np.random.default_rng(0); rnd = random.Random(0)
lessons = [(src[7:], s) for s, src in uc.SENTS if src.startswith("lesson:")]
print("held lessons available:", len(lessons))
cons = corr = 0
for q, ref in rnd.sample(lessons, 5):
    if len(q.strip()) < 10: continue
    ans, _ = uc.reply(q, rng)
    ov = set(uc.content_words(ans)) & set(uc.content_words(ref))
    need = max(1, len(uc.content_words(ref)) // 2)
    if len(ov) >= need or ans.strip() == ref.strip():
        cons += 1
    else:
        uc.record_correction(q, ref); corr += 1
print(f"self-play batch: {cons} consolidated, {corr} self-corrected  (both paths live)")

# one REAL tutor cycle: gemma writes the QA, engine answers, gemma judges, closure applied
uc.AUTO["teach"] = True
import re, random as R
f = R.Random(7).choice(uc.THEORY)
text = open(f, errors="ignore").read()
passage = text[1000:3500]
out = uc._ollama("Below is a passage from the Smithian Fold Theory corpus. Write exactly ONE question a curious person might ask about it, and its answer grounded ONLY in the passage. Keep the answer to 1-2 plain sentences. No markdown.\nFormat STRICTLY as:\nQ: ...\nA: ...\n\nPASSAGE:\n" + passage)
m = re.search(r"Q:\s*(.+?)\nA:\s*(.+?)(?=\nQ:|\Z)", out, re.S)
q = " ".join(m.group(1).split())[:200]; ref = " ".join(m.group(2).split())[:350]
print("tutor asked:", q)
ans, _ = uc.turn(q, rng, "tutor")
print("engine said:", ans[:120])
verdict = uc._ollama("QUESTION: " + q + "\nREFERENCE ANSWER: " + ref + "\nSTUDENT ANSWER: " + ans + "\nDoes the student answer convey the reference answer's meaning? Reply with exactly one word: YES or NO.", timeout=300)
yes = bool(re.search(r"\bYES\b", verdict.upper()))
print("tutor judged:", "YES -> y consolidation" if yes else "NO -> n + correction held")
res = uc.apply_feedback(q, ans, "y" if yes else "n " + ref, "tutor")
print("closure applied:", res[:80])
print("AUTONOMY TEST COMPLETE")
