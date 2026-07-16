"""Verify UNFOLD stops verbatim replay: generate replies with the capped unfold and
measure the longest contiguous word-run each shares with ANY stored orbit. A high
overlap = still replaying; a low one = re-composed. Also prints replies to eyeball."""
import os, sys, random, re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import SynapticGraph, predict_next
from omni.word_engine import word_engine, generate_multiscale, tokenize

graph = SynapticGraph()
ukey = "discord_2812840720"
word_engine.ensure_built(graph, ukey)

# stored orbit word-sequences (the memorized lessons) for overlap checking
orbit_words = []
for seq in graph.orbits.get(ukey, []):
    s = "".join(seq).replace("\x02", " ").replace("\x03", " ").replace("\x04", " ").replace("\x05", " ")
    orbit_words.append([w.lower() for w in tokenize(s)])


def longest_verbatim_run(out_words):
    """longest run of out_words that appears contiguously in some orbit."""
    best = 0
    ow = [w.lower() for w in out_words]
    for i in range(len(ow)):
        for orb in orbit_words:
            # match starting at each orbit position
            for j in range(len(orb)):
                k = 0
                while i + k < len(ow) and j + k < len(orb) and ow[i + k] == orb[j + k]:
                    k += 1
                if k > best:
                    best = k
    return best


def gen(prompt, seed=0):
    routed = word_engine.kin_route(list(prompt))
    ctx = routed if routed is not None else list("\x02" + prompt + "\x03\x04")
    # topical bias = content words of the routed meaning + the prompt
    bias = tokenize("".join(ctx).replace("\x02", " ").replace("\x03", " ").replace("\x04", " ")) + tokenize(prompt)
    word_engine.set_generation_bias(bias)
    rng = random.Random(seed)
    def word_sample(words, r): return word_engine.sample_next_unfold(words, r)
    def char_predict(chars): return (None,0,0)
    surface, _ = generate_multiscale(ctx, word_sample, char_predict, rng, max_chars=400)
    return surface.replace("\x04", "").replace("\x05", "").strip()


TESTS = ["Hello", "what do you think about the ocean", "how are you feeling today",
         "tell me a story about a dog", "what is your favorite food", "do you remember me"]
print(f"loaded {len(orbit_words)} stored orbits | UNFOLD cap = 4 words\n")
for p in TESTS:
    out = gen(p)
    ow = tokenize(out)
    run = longest_verbatim_run(ow)
    flag = "REPLAY" if run >= 8 else ("partial" if run >= 5 else "re-composed")
    print(f"[{flag}] longest verbatim run = {run} words | PROMPT: {p}")
    print(f"    {out[:200]!r}\n")
