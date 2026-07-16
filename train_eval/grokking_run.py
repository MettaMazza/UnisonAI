"""THE GENERALISATION TELESCOPE — the counted analogue of grokking, run and measured.

HYPOTHESIS (from the forced laws): a novel prompt is answered from held material iff its
coupling to something held reaches the lock 1/2 (the binding law, reused). Each taught
meaning covers a kin-NEIGHBORHOOD, not a point; generalisation onset is therefore the
measured moment binding COVERAGE over never-taught probes crosses the lock — predictable
from a counted quantity, unlike gradient grokking.

THE RUN (autonomous): rounds over a fixed TEACH set — reply -> calibrated judge -> on BAD,
the teacher corrects with THREE phrasings (re-expression law: >= b held expressions) and the
meaning is taught. At DYADIC checkpoints (rounds 1, 2, 4, 8, 16 — the decode campaign's own
telescope convention) three curves are measured on FROZEN sets:
  memorization : judged GOOD on the taught prompts themselves (re-expressed service)
  near-transfer: judged GOOD on PARAPHRASES of taught topics (never taught; kin must carry)
  far-transfer : judged GOOD on novel topics (never taught)
plus the PREDICTOR: the share of probe prompts whose best pair-binding >= the lock.
Probes are read-only (no feedback, no teaching) — they never contaminate the stores.

Teaches into the LIVE stores (backed up first) so the run leaves the engine generalised for
the manual sessions that follow. Ledger: logs/grokking.jsonl + logs/grokking_run.out.

Run: PYTHONPATH=. python3 train_eval/grokking_run.py
"""
import sys, os, json, re, shutil, urllib.request, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.pair_retrieval import pair_retrieval, TAUGHT_PATH, QUALITY_PATH
from omni.word_engine import tokenize, _content_words
from train_eval.judge import judge, judge_pool

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "gemma-4-31b:latest"
MAX_ROUNDS = 16
CHECKPOINTS = {1, 2, 4, 8, 16}
LEDGER = os.path.join(os.path.dirname(__file__), "..", "logs", "grokking.jsonl")
TEACH_LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "grokking_teach.jsonl")
_REPLY_LOCK = threading.Lock()

# ---- THE TEACH SET (the autonomous curriculum) ----
# When train_eval/curriculum_1000.json exists it REPLACES the built-in 48 (volume run);
# generated once by the teacher, deduped, probe-excluded, contamination-checked below.
CURRICULUM = os.path.join(os.path.dirname(__file__), "curriculum_1000.json")
TEACH = [
    "Hello, how are you?", "how's it going today?", "what should I cook for dinner",
    "do you like music", "what are your hobbies", "recommend a good book",
    "I'm feeling a bit sad today", "what do you think about the ocean",
    "tell me about space", "I just finished a painting", "what makes a good friend",
    "any tips for sleeping better?", "I had a rough day at work",
    "what's your favorite season?", "do you enjoy cooking?", "I'm learning to play guitar",
    "what would you do on a rainy day?", "I love hiking in the mountains",
    "how do I stay motivated?", "tell me something interesting",
    "I'm planning a trip to Italy", "do you like coffee or tea?",
    "my dog did the funniest thing today", "I can't decide what movie to watch",
    "what's a good way to relax?", "I started a new job this week",
    "do you believe in luck?", "what's the best advice you've ever gotten?",
    "I'm trying to eat healthier", "tell me about your day",
    "what music helps you focus?", "I feel nervous about tomorrow",
    "how do you deal with stress?", "what's your idea of a perfect weekend?",
    "I just adopted a kitten", "any good podcast recommendations?",
    "I want to start journaling", "what's something that made you smile recently?",
    "do you prefer mornings or evenings?", "I'm redecorating my room",
    "what hobby should I try next?", "I baked bread for the first time",
    "how important is routine to you?", "what do you think about art?",
    "I've been feeling really grateful lately", "what's a skill worth learning?",
    "tell me a fun fact about animals", "what makes a house feel like home?",
]

if os.path.exists(CURRICULUM):
    TEACH = json.load(open(CURRICULUM))

# ---- FROZEN PROBES (never taught, never feedback) ----
NEAR = [  # paraphrases of taught topics — kin must carry the meaning across wording
    "hey, how are things?", "any ideas for tonight's meal?", "what kind of music do you enjoy?",
    "got any book suggestions?", "I'm feeling down today", "thoughts on the sea?",
    "what do you get up to for fun?", "how can I keep my motivation up?",
    "I finished an art piece today", "tips for winding down in the evening?",
]
FAR = [  # novel topics — coverage must reach them
    "what do you think about gardening?", "I'm nervous about public speaking",
    "do you like board games?", "tell me about the desert",
    "I volunteered at a shelter today", "what's your take on minimalism?",
    "I watched a documentary about whales", "how do people make friends as adults?",
    "I'm saving up for a bicycle", "my neighbour keeps borrowing my tools",
]


def teacher_answer(p, differently_from=None):
    extra = (f", expressing the SAME content as this but worded differently: "
             f"{differently_from!r}" if differently_from else "")
    prompt = (f"The user says: {p!r}. Respond as Unison — natural, warm, in your own voice, "
              f"in 1-2 short sentences{extra}. No preamble, just the reply.")
    body = json.dumps({"model": TEACHER, "prompt": prompt, "stream": False, "think": False,
                       "options": {"temperature": 0.8}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                OLLAMA, data=body, headers={"Content-Type": "application/json"}), timeout=180) as r:
            out = json.loads(r.read().decode()).get("response", "")
        out = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", out).strip()
        return out.split("\n")[0][:300]
    except Exception:
        return ""


def binding_share(prompts):
    """THE PREDICTOR: share of prompts whose best available binding (taught-overlap or
    pair-prompt similarity) reaches the lock 1/2 — the counted quantity the hypothesis
    says gates generalisation."""
    P = pair_retrieval._pairs()
    taught = pair_retrieval._taught_pairs()
    bound = 0
    for p in prompts:
        qcw = set(_content_words(tokenize(p.lower())))
        best = 0.0
        for tp in taught:
            tcw = set(tp["cw"])
            if tcw:
                best = max(best, pair_retrieval.kin_binding(qcw, tcw))
        if best < 0.5:
            for s, pid in pair_retrieval.retrieve(p, topn=3):
                pcw = set(_content_words(tokenize(P["prompts"][pid].lower())))
                if qcw:
                    best = max(best, len(qcw & pcw) / max(len(qcw | pcw), 1))
        bound += (best >= 0.5)
    return bound / len(prompts)


def judged_rate(prompts, transcripts=None, pool=None):
    """POOL verdicts (both independent calibrated judges must agree GOOD) + the actual
    replies saved for human eyes — scores never stand alone. CONCURRENT: the engine's own
    replies are milliseconds; the judge calls fan out across the parallel server."""
    def one(p):
        with _REPLY_LOCK:
            r = (pair_retrieval.reply(p) or "").strip()
        ok, agree = judge_pool(p, r)
        return {"prompt": p, "reply": r, "pool_good": ok, "judges": agree}
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, prompts))
    if transcripts is not None:
        transcripts.extend(rows)
    return sum(r["pool_good"] for r in rows) / len(prompts)


def checkpoint(rnd, taught_count):
    tr = []
    row = {"round": rnd, "taught_meanings": taught_count,
           "memorization": judged_rate(TEACH[::max(1, len(TEACH) // 48)][:48], tr),  # fixed stride sample across the whole set (cost cap)
           "near_transfer": judged_rate(NEAR, tr),
           "far_transfer": judged_rate(FAR, tr),
           "transcripts": tr,
           "binding_share_near": binding_share(NEAR),
           "binding_share_far": binding_share(FAR),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"CHECKPOINT r{rnd}: mem={row['memorization']:.0%} near={row['near_transfer']:.0%} "
          f"far={row['far_transfer']:.0%} | bind(near)={row['binding_share_near']:.0%} "
          f"bind(far)={row['binding_share_far']:.0%} | taught={taught_count}", flush=True)
    return row


def main():
    # back up the live stores before an autonomous teaching run
    for path in (TAUGHT_PATH, QUALITY_PATH):
        if os.path.exists(path):
            shutil.copy(path, path + ".pre_grokking_backup")
    # CONTAMINATION CHECK: no probe may exist verbatim in the pair corpus
    P = pair_retrieval._pairs()
    plow = {q.strip().lower() for q in P["prompts"][:200000]} | {q.strip().lower() for q in P["prompts"][200000:]}
    tlow = {q.strip().lower() for q in TEACH}
    dirty = [q for q in NEAR + FAR if q.strip().lower() in plow or q.strip().lower() in tlow]
    if dirty:
        print(f"CONTAMINATED PROBES (in corpus or teach set): {dirty} — ABORT", flush=True)
        return
    print(f"contamination check: clean (no probe in corpus or teach set)", flush=True)
    print(f"=== THE GENERALISATION TELESCOPE: {len(TEACH)} teach, "
          f"{len(NEAR)}+{len(FAR)} frozen probes, dyadic checkpoints, POOL verdicts ===", flush=True)
    store_lock = threading.Lock()
    for rnd in range(1, MAX_ROUNDS + 1):
        # PHASE 1 (concurrent): evaluate the whole teach set — engine replies are ms;
        # judge calls fan out across the parallel server
        def evaluate(p):
            # the engine reply is milliseconds and touches shared state (last_pids):
            # serialize it and CAPTURE the pids with the reply; only the judge call
            # (the slow, stateless part) runs outside the lock.
            with store_lock:
                r = (pair_retrieval.reply(p) or "").strip()
                pids = list(pair_retrieval.last_pids)
            ok, _ = judge(p, r)
            return p, r, ok, pids
        with ThreadPoolExecutor(max_workers=8) as ex:
            verdicts = list(ex.map(evaluate, TEACH))
        with store_lock:
            for _, _, ok, pids in verdicts:
                pair_retrieval.mark_feedback(ok, pids=pids)
        failures = [p for p, _, ok, _ in verdicts if not ok]
        # PHASE 2 (concurrent): each failure's three teacher phrasings in parallel,
        # store writes serialized under the lock
        def correct(p):
            a1 = teacher_answer(p)
            if not a1:
                return p, None, []
            with ThreadPoolExecutor(max_workers=2) as ex2:
                f2 = ex2.submit(teacher_answer, p, a1)
                f3 = ex2.submit(teacher_answer, p, a1)
                a2, a3 = f2.result(), f3.result()
            return p, a1, [v for v in (a2, a3) if v and v != a1]
        taught_this = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            for p, a1, vs in ex.map(correct, failures):
                if a1:
                    with store_lock:
                        pair_retrieval.add_taught(p, a1, variants=vs)
                        held = next((tp for tp in pair_retrieval._taught_pairs()
                                     if tp["prompt"].strip().lower() == p.strip().lower()), None)
                    with open(TEACH_LOG, "a") as f:
                        f.write(json.dumps({"round": rnd, "prompt": p, "primary": a1,
                                            "offered_variants": vs,
                                            "held_variants": (held or {}).get("variants", []),
                                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                    taught_this += 1
        n_taught = len(pair_retrieval._taught_pairs())
        print(f"round {rnd}: corrected {taught_this}, held meanings {n_taught}", flush=True)
        if rnd in CHECKPOINTS:
            checkpoint(rnd, n_taught)
            # NO auto-crowning: the run completes its full dyadic ladder and reports curves
            # + transcripts only. "Baseline generalised" is declared by Maria from the
            # evidence, never by the harness.
    print("=== RUN COMPLETE — ledger: logs/grokking.jsonl ===", flush=True)


if __name__ == "__main__":
    main()
