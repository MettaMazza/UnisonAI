"""F0 — the free-generation harness (FRONTIER_PLAN gates a and c).

Two arms, both judged by the CALIBRATED judge:
  baseline : the existing n-gram free arm (unfold_response / structured_unfold — the
             substrate free-running; the committed pre-frontier baseline)
  f1       : kin-context mixing (free_gen) — retrieval as DISTRIBUTION, never surface
Plus the multi-sentence probe: the f1 arm asked to continue to 3 sentences, judged.

CE gate note (gate b): F1 is a new arm and does not modify predict_next / the fluency
stores, so the committed CE baselines are unchanged by construction this stage.

Run:  PYTHONPATH=. python3 train_eval/gen_free_harness.py
"""
import sys, os, json, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import word_engine, tokenize, _content_words
from omni.free_gen import free_gen
from train_eval.judge import judge

OPENERS = [
    "Hello, how are you?", "how's it going?",
    "My name is Maria, what is your name?",
    "what do you think about the ocean", "tell me about space",
    "I'm feeling a bit sad today", "what should I cook for dinner",
    "do you like music", "what are your hobbies", "recommend a good book",
    "I just finished a painting", "what makes a good friend",
]


def baseline_arm(p, rng):
    schema = _content_words(tokenize(p)) * 2
    s4 = word_engine.structured_unfold(schema, rng)
    uf = word_engine.unfold_response(schema, rng)
    return (s4 or uf or "").strip()


def main():
    rng = random.Random(7)
    word_engine._load_coupling()
    b_good = f_good = p_good = 0
    rows = []
    print("arm-B[base] arm-F[f1kin] arm-P[f3plan] | prompt")
    print("-" * 72)
    for p in OPENERS:
        br = baseline_arm(p, rng)
        fr = (free_gen.generate(p, rng=rng) or "").strip()
        pr = (free_gen.generate_planned(p, rng=rng) or "").strip()
        bg, _ = judge(p, br)
        fg, _ = judge(p, fr)
        pg, _ = judge(p, pr)
        b_good += bg; f_good += fg; p_good += pg
        rows.append({"prompt": p, "baseline": br, "f1": fr, "f3": pr,
                     "b_good": bg, "f_good": fg, "p_good": pg})
        print(f"   [{'G' if bg else 'B'}] [{'G' if fg else 'B'}] [{'G' if pg else 'B'}] | {p!r}")
        print(f"      f1: {fr[:84]!r}")
        print(f"      f3: {pr[:84]!r}")
    n = len(OPENERS)
    print("-" * 72)
    print(f"FREE-ARM judged GOOD:  baseline {b_good}/{n}  |  F1 {f_good}/{n}  |  F3-planned {p_good}/{n} = {p_good/n:.0%}")

    # multi-sentence probe (gate c): three-sentence continuation, judged as a whole
    ms_good = 0
    MS = ["tell me about space", "what makes a good friend", "I'm feeling a bit sad today"]
    for p in MS:
        parts = []
        for _ in range(3):
            parts.append(free_gen.generate(p + " " + " ".join(parts), rng=rng))
        whole = " ".join(x for x in parts if x)
        ok, _ = judge(p, whole)
        ms_good += ok
        print(f"MS[{'G' if ok else 'B'}] {p!r}: {whole[:100]!r}")
    print(f"MULTI-SENTENCE probe: {ms_good}/{len(MS)}")
    with open(os.path.join(os.path.dirname(__file__), "..", "logs", "gen_free.jsonl"), "a") as f:
        f.write(json.dumps({"baseline": b_good / n, "f1": f_good / n,
                            "multi_sentence": ms_good / len(MS), "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
