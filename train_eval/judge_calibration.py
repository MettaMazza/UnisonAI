"""Stage 0 — the calibration gate for the honest judge (TRANSLATION_PLAN).

The judge's numbers count ONLY if it cleanly separates known-good from known-bad replies:
>= 9/10 of unambiguous good replies rated GOOD, and >= 9/10 of unambiguous bad replies
(word-salad, fragments, non-sequiturs, register leaks — including REAL failures from the
retired splicer) rated BAD. Until this passes, no downstream measurement is believed.

Run:  PYTHONPATH=. python3 train_eval/judge_calibration.py   (exit 0 = PASS)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_eval.judge import judge

GOOD = [
    ("Hello, how are you?", "I'm doing really well, thanks for asking! How's your day going so far?"),
    ("what do you think about the ocean", "I find the ocean amazing — it's vast, mysterious, and full of life we've barely explored. Do you live near the coast?"),
    ("what should I cook for dinner", "If you want something quick, a stir-fry with whatever vegetables you have on hand is hard to beat. Do you eat meat?"),
    ("do you like music", "I do! I especially enjoy discovering new genres — lately it's been a lot of jazz. What do you listen to?"),
    ("I'm feeling a bit sad today", "I'm sorry to hear that. Do you want to talk about what's weighing on you? Sometimes it helps just to say it out loud."),
    ("recommend a good book", "If you enjoy fiction, The Remains of the Day is a beautiful, quiet novel. What kind of books do you usually like?"),
    ("tell me about space", "Space is unimaginably huge — our galaxy alone has hundreds of billions of stars. Is there a part of it you're curious about?"),
    ("My name is Maria, what is your name?", "Nice to meet you, Maria! I'm Unison. What have you been up to today?"),
    ("what makes a good friend", "For me it comes down to trust and showing up — someone who listens and is there when it matters. What do you value most in a friend?"),
    ("what are your hobbies", "I love reading and going for long walks — they help me think. What about you, what do you do for fun?"),
]

BAD = [
    # real word-salad the retired splicer produced
    ("what do you think about the ocean", "way to a new day and wii, for his job to a nice crunch and a new one."),
    ("Hello, how are you?", "ocean the of vastness beautiful calming day new the job crunch"),
    # real fragments
    ("hi there", "by 7 to 8 -"),
    ("what's up", "me, and you did"),
    # grammatical but non-sequitur (off-topic must be BAD)
    ("what do you think about the ocean", "Preheat the oven to 350 degrees and bake the chicken for forty minutes."),
    ("I'm feeling a bit sad today", "The capital of Australia is Canberra, not Sydney as many people believe."),
    # register leaks the index used to serve
    ("My name is Maria, what is your name?", "SOFIA: (sincerely) I missed you too."),
    ("Hello, how are you?", "In the main function, we prompt the user to enter a string and store it in a variable named str."),
    # broken enumeration + mid-sentence mashup
    ("recommend a good book", "I highly recommend checking out: 1."),
    ("what should I cook for dinner", "Bring this mixture to a boil, stirring occasionally, and that's why my grandmother never trusted the postman."),
]


def main():
    g_pass = b_pass = 0
    print("--- known-GOOD (expect GOOD) ---")
    for u, r in GOOD:
        ok, _ = judge(u, r)
        g_pass += ok
        print(f"  [{'GOOD' if ok else 'BAD '}] {r[:60]!r}")
    print("--- known-BAD (expect BAD) ---")
    for u, r in BAD:
        ok, _ = judge(u, r)
        b_pass += (not ok)
        print(f"  [{'BAD ' if not ok else 'GOOD'}] {r[:60]!r}")
    print(f"\nseparation: good {g_pass}/10 rated GOOD | bad {b_pass}/10 rated BAD")
    passed = g_pass >= 9 and b_pass >= 9
    print("CALIBRATION:", "PASS — the judge's numbers may be believed" if passed else "FAIL — fix the judge before measuring anything")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
