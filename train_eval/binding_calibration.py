"""The calibration gate for the BINDING measure — the training loss and taught route.

The standing rule ([[honest-evaluation-first]]) applies to EVERY instrument, not only LLM
judges: no measure's numbers are believed before it cleanly separates known cases. The
binding measure is the epoch loop's loss AND the serving router (one quantity by design,
pair_retrieval.taught_binding) — so it gates like a judge:

  known-SHOULD-BIND : true paraphrase pairs (same meaning, different words) must reach
                      the lock 1/2
  known-should-NOT  : the ACTUAL false binds measured in the 2026-07-16 epoch-1 run
                      (generic-verb collisions, act mismatches) must stay below it

PASS = >= 9/10 each side. The epoch run does not launch on a measure that fails this.

Run: PYTHONPATH=. python3 train_eval/binding_calibration.py   (exit 0 = PASS)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.pair_retrieval import pair_retrieval, TAUGHT_LOCK

SHOULD_BIND = [
    # (query, taught prompt) — same meaning, reworded; kin must carry
    ("I'm feeling down today", "I'm feeling a bit sad today"),
    ("hey, how are things going?", "how's it going today?"),
    ("any ideas for tonight's meal?", "what should I cook for dinner?"),
    ("got any book suggestions?", "can you recommend a good book?"),
    ("thoughts on the sea?", "what do you think about the ocean?"),
    ("what kind of music do you enjoy?", "do you like music?"),
    ("how can I keep my motivation up?", "how do I stay motivated?"),
    ("tips for winding down in the evening?", "any tips for relaxing at night?"),
    ("I'm exhausted after work today", "I had a rough day at work"),
    ("what's a good way to relax?", "how do you deal with stress?"),
]

SHOULD_NOT_BIND = [
    # the measured false binds, epoch 1 (logs/generalisation_epochs.jsonl, 2026-07-16)
    ("what's your take on minimalism?", "Thanks, I needed that. I think I'm going to go take a relaxing bath before I get ready."),
    ("what do you think about gardening?", "Thanks, I needed that. I think I'm going to go take a relaxing bath before I get ready."),
    ("hey, how are things?", "Thanks. I'm just trying to take things one day at a time right now."),
    ("what do you get up to for fun?", "It is! It's a lot of hard work, but it's also a lot of fun."),
    ("how can I keep my motivation up?", "Tell me about it! I'm just trying to keep my head above water, you know?"),
    ("I could use some cheering up", "Sure, we could always use another player. What position do you play?"),
    ("tell me about somewhere beautiful", "Can you suggest a beautiful font style that would make my letter look more heartfelt?"),
    ("I'm nervous about public speaking", "Sorry doesn't cut it. You know better than to pick your nose in public."),
    ("do you know anything about beekeeping suits?", "I know, but still. She didn't have to take it out on me. I didn't do anything to her."),
    ("what's it like to live on a houseboat?", "Can you provide more information on how theater companies and producers are using livestreams?"),
]


def main():
    ok_bind = ok_not = 0
    print(f"--- SHOULD BIND (expect >= {TAUGHT_LOCK}) ---")
    for q, t in SHOULD_BIND:
        b = pair_retrieval.taught_binding(q, t)
        hit = b >= TAUGHT_LOCK
        ok_bind += hit
        print(f"  [{'BIND' if hit else 'MISS'}] {b:.2f}  {q!r} ~ {t!r}")
    print(f"--- should NOT bind (expect < {TAUGHT_LOCK}) ---")
    for q, t in SHOULD_NOT_BIND:
        b = pair_retrieval.taught_binding(q, t)
        hit = b < TAUGHT_LOCK
        ok_not += hit
        print(f"  [{'ok  ' if hit else 'FALSE'}] {b:.2f}  {q!r} ~ {t[:60]!r}")
    passed = ok_bind >= 9 and ok_not >= 9
    print(f"BINDING CALIBRATION: bind {ok_bind}/10 | reject {ok_not}/10 -> "
          f"{'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
