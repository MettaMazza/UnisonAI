"""Stage 2+3 measurement — judged by the CALIBRATED judge (Stage 0 passed 10/10|10/10).

Per opener: (a) CEILING — the top-1 retrieved response served raw (diagnostic only, never
production: verbatim is forbidden live); (b) END-TO-END — pair_retrieval.reply(), the
composed non-verbatim reply. Both judged GOOD/BAD. The ceiling bounds what ranking finds;
end-to-end is what the user would get.

Run:  PYTHONPATH=. python3 train_eval/measure_pairs.py
"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_eval.judge import judge
from omni.pair_retrieval import pair_retrieval

OPENERS = [
    "Hello, how are you?", "hi there", "how's it going?", "what's up",
    "My name is Maria, what is your name?", "Nice to meet you, do you remember my name?",
    "what do you think about the ocean", "tell me about space",
    "I'm feeling a bit sad today", "what should I cook for dinner",
    "do you like music", "what are your hobbies", "recommend a good book",
    "I just finished a painting", "what makes a good friend",
    "Nothing much I'm just chilling, what's on your mind?",
]


def main():
    P = pair_retrieval._pairs()
    ceil_g = e2e_g = 0
    rows = []

    # MULTI-TURN name recall (the live-session failure case): the name arrives two turns
    # earlier; the reply must use it (relexicalization + 2^-age context) and be judged GOOD.
    hist = [("user", "My name is Maria, what is your name?"),
            ("unison", "Nice to meet you! I'm Unison."),
            ("user", "It's going well thanks."),
            ("unison", "Glad to hear it!")]
    nq = "Do you remember my name?"
    nr = pair_retrieval.reply(nq, history=hist)
    ng, _ = judge(nq, nr)
    has_name = "Maria" in nr
    print(f"NAME-RECALL e2e[{'G' if ng else 'B'}] name-used[{'Y' if has_name else 'N'}]: {nr[:95]!r}")
    rows.append({"prompt": nq, "composed": nr, "e2e_good": ng, "name_used": has_name})
    for m in OPENERS:
        cands = pair_retrieval.retrieve(m, topn=3)
        top1 = P["responses"][cands[0][1]] if cands else ""
        composed = pair_retrieval.reply(m)
        cg, _ = judge(m, top1)
        eg, _ = judge(m, composed)
        ceil_g += cg; e2e_g += eg
        rows.append({"prompt": m, "top1": top1, "composed": composed,
                     "ceiling_good": cg, "e2e_good": eg})
        print(f"ceil[{'G' if cg else 'B'}] e2e[{'G' if eg else 'B'}] {m!r}")
        print(f"   top1    : {top1[:95]!r}")
        print(f"   composed: {composed[:95]!r}")
    n = len(OPENERS)
    print(f"\nJUDGED CEILING (top-1 raw, diagnostic): {ceil_g}/{n} = {ceil_g/n:.0%}")
    print(f"JUDGED END-TO-END (composed, non-verbatim): {e2e_g}/{n} = {e2e_g/n:.0%}")
    with open(os.path.join(os.path.dirname(__file__), "..", "logs", "measure_pairs.jsonl"), "a") as f:
        f.write(json.dumps({"ceiling": ceil_g / n, "e2e": e2e_g / n, "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
