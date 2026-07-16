"""Offline generation comparison: OLD law (sample_next_unit = longest-suffix replay)
vs CORRECTED law (sample_next = the 2^L cascade). Loads the live persisted memory,
generates a response to verbatim AND novel prompts, so we can see replay-vs-compose
before restarting the live bot."""
import os, sys, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.memory import SynapticGraph, predict_next
from omni.word_engine import word_engine, generate_multiscale, STOP, END_USER, THINK_OPEN

graph = SynapticGraph()                      # loads graph_memory.json
ukey = "discord_2812840720"
word_engine.ensure_built(graph, ukey)


def gen(prompt, sampler, seed=0):
    # live path: kin_route gives relevance, then generate from that context
    routed = word_engine.kin_route(list(prompt))
    ctx = routed if routed is not None else list("\x02" + prompt + "\x03\x04")
    rng = random.Random(seed)
    def word_sample(words, r): return sampler(words, r)
    def char_predict(chars): return predict_next(chars, graph, ukey, None)
    surface, _ = generate_multiscale(ctx, word_sample, char_predict, rng, max_chars=400)
    return surface.replace("\x04", "").replace("\x05", "").strip()


TESTS = ["Hello",                                   # verbatim taught
         "what do you think about the ocean",        # novel
         "how are you feeling today",                # novel
         "tell me a story about a dog"]              # novel

for p in TESTS:
    print("=" * 70)
    print("PROMPT:", p)
    print("  OLD  (unit/replay):", repr(gen(p, word_engine.sample_next_unit)[:220]))
    print("  NEW  (cascade):    ", repr(gen(p, word_engine.sample_next)[:220]))
