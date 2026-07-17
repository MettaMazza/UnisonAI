"""EPOCH TRAINING, TRANSLATED — continuous generalisation over the real data stream.

The 1-1 translation of the standard training loop into the counted engine (no arbitrary
totals — the run is CONTINUOUS until the generalisation criterion is met):

  training dataset -> the pair corpus's real user prompts (649,917), deterministic order
  batch            -> BAND = 32 prompts (forced: 2^(b+c), functional_band.ep)
  epoch            -> BAND batches = 1,024 prompts (BAND^2 — structural, not a knob)
  per-item loss    -> THE OBJECTIVE ITSELF, measured on the item: the training judge's
                      verdict on the engine's actual reply (in gradient training the
                      per-item loss IS the objective; a similarity proxy is not — three
                      proxy generations failed at volume, banked in the binding gate).
                      BAD -> TEACH (three same-meaning phrasings, the re-expression
                      law). Binding remains the calibration-gated serving router and
                      the measured onset PREDICTOR, never the loss.
  validation       -> FROZEN probes (32 near + 32 far, band-sized), never taught, never
                      feedback. Pool-judged per epoch (both calibrated judges, concurrent,
                      both must agree GOOD), with full transcripts.
  early stopping   -> near-transfer >= 1/2 AND far-transfer >= 1/2 (the lock) for
                      b = 2 CONSECUTIVE epochs. The run then STOPS AND REPORTS — the
                      evidence goes to Maria; the harness never declares baseline.
  checkpointing    -> resumable stream cursor (logs/epoch_state.json); ledgers:
                      logs/generalisation_epochs.jsonl (benchmarks + transcripts)
                      logs/epoch_teach.jsonl (every teaching event, full transparency)

Feedback discipline: the memorization benchmark (taught items) marks Laplace feedback with
per-reply captured pids; probes NEVER feed back. No steering signal is a scoreboard.

Run:      PYTHONPATH=. python3 train_eval/generalisation_epochs.py
Preflight (2 batches of 4, tiny probes, then exit — for wiring verification):
          PYTHONPATH=. python3 train_eval/generalisation_epochs.py --preflight
"""
import sys, os, json, re, hashlib, shutil, subprocess, urllib.request, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.pair_retrieval import pair_retrieval, TAUGHT_PATH, QUALITY_PATH, TAUGHT_LOCK
from omni.word_engine import tokenize, _content_words
from train_eval.judge import judge, judge_pool

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "gemma-4-31b:latest"
BAND = 32                      # forced: 2^(b+c) — the batch
BATCHES_PER_EPOCH = BAND       # epoch = BAND batches (BAND^2 prompts)
PERSIST = 2                    # forced: b — consecutive epochs the criterion must hold
LOGS = os.path.join(os.path.dirname(__file__), "..", "logs")
LEDGER = os.path.join(LOGS, "generalisation_epochs.jsonl")
TEACH_LOG = os.path.join(LOGS, "epoch_teach.jsonl")
STATE = os.path.join(LOGS, "epoch_state.json")
PREFLIGHT = "--preflight" in sys.argv
_REPLY_LOCK = threading.Lock()

# ---- FROZEN VALIDATION PROBES (band-sized; never taught, never feedback) ----
NEAR = [  # paraphrases of common conversational territory — kin must carry the meaning
    "hey, how are things?", "any ideas for tonight's meal?", "what kind of music do you enjoy?",
    "got any book suggestions?", "I'm feeling down today", "thoughts on the sea?",
    "what do you get up to for fun?", "how can I keep my motivation up?",
    "I finished an art piece today", "tips for winding down in the evening?",
    "how has your week been going?", "what's a tasty breakfast idea?",
    "know any good films worth seeing?", "I could use some cheering up",
    "what pastimes do you recommend?", "any advice for a beginner cook?",
    "I'm exhausted after today", "what's the nicest place you can imagine visiting?",
    "how do you unwind after a long day?", "my week has been stressful",
    "what snacks go well with a movie night?", "I want to pick up a creative pastime",
    "how do I make my mornings less chaotic?", "what's something fun to do on a weekend?",
    "I've been sleeping badly lately", "what would make a good gift for a friend?",
    "I'm bored, entertain me", "what's a comforting meal for a cold day?",
    "how do people stay positive?", "I just got back from a long walk",
    "tell me about somewhere beautiful", "what's worth celebrating this time of year?",
]
FAR = [  # novel territory — coverage must reach it
    "what do you think about gardening?", "I'm nervous about public speaking",
    "do you like board games?", "tell me about the desert",
    "I volunteered at a shelter today", "what's your take on minimalism?",
    "I watched a documentary about whales", "how do people make friends as adults?",
    "I'm saving up for a bicycle", "my neighbour keeps borrowing my tools",
    "what's it like to live on a houseboat?", "I'm learning sign language",
    "do you find thunderstorms exciting or scary?", "my sourdough starter died",
    "what would you ask a lighthouse keeper?", "I'm thinking of keeping bees",
    "how do orchestras stay in sync?", "I found an old coin in my garden",
    "what makes deserts cold at night?", "my chess club needs new members",
    "I tried fencing for the first time", "what's the appeal of birdwatching?",
    "our town is starting a repair cafe", "I can't stop thinking about glaciers",
    "do you know anything about beekeeping suits?", "my cousin restores vintage motorbikes",
    "what should I name a pet tortoise?", "I'm writing a letter to my future self",
    "how do submarines navigate?", "the museum near me has a new dinosaur exhibit",
    "I want to learn calligraphy", "what's a constellation worth finding first?",
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
        out = out.split("\n")[0][:300]
        # cut at the last complete sentence — a mid-sentence fragment stored as a held
        # expression serves fragments forever (measured: "It's a great way")
        if out and out[-1] not in ".!?":
            m = re.match(r"^(.*[.!?])[^.!?]*$", out, re.S)
            if m:
                out = m.group(1)
        return out
    except Exception:
        return ""


def build_stream():
    """The training dataset: real corpus prompts, counted filters, deduped, probe-excluded,
    DETERMINISTIC order (hash of the normalized string — reproducible, no RNG)."""
    P = pair_retrieval._pairs()
    probes = {q.strip().lower() for q in NEAR + FAR}
    seen, stream = set(), []
    for q in P["prompts"]:
        qs = q.strip()
        norm = qs.lower()
        if norm in seen or norm in probes:
            continue
        words = qs.split()
        if not (3 <= len(words) <= 20):
            continue
        if len(_content_words(tokenize(norm))) < 2:
            continue
        if re.search(r"https?://|[<>{}\[\]\\|`#@_=~^]|\d{4,}", qs):
            continue
        if not re.search(r"[a-zA-Z][.?!']?$", qs):
            continue
        # curriculum cleaning (standard training-data hygiene): a capitalized token that
        # is not sentence-initial marks persona/entity-bound material ("I'm sorry, Sammy",
        # "the trial of Jeromie Cancel") — general conversational curriculum only.
        toks = qs.split()
        named = False
        for i, tk in enumerate(toks):
            core = re.sub(r"^\W+|\W+$", "", tk)
            sent_initial = i == 0 or toks[i - 1].rstrip()[-1:] in ".?!"
            if (core and re.match(r"^[A-Z][a-z]+", core) and not sent_initial
                    and core not in ("I",) and not core.startswith("I'")):
                named = True
                break
        if named:
            continue
        seen.add(norm)
        stream.append(qs)
    # ACT-BALANCED CURRICULUM (deployment-distribution alignment, not test-matching:
    # the engine deploys as an addressed assistant — questions and self-disclosures
    # roughly evenly; the raw pair corpus is 2:1 mid-dialogue statements). Question-form
    # and statement-form interleave 1:1, each queue hash-ordered (deterministic, no RNG).
    # Prompts already held as taught meanings are excluded — resuming with a rebuilt
    # stream re-teaches nothing.
    taught_low = {tp["prompt"].strip().lower() for tp in pair_retrieval._taught_pairs()}
    stream = [q for q in stream if q.strip().lower() not in taught_low]
    key = lambda q: hashlib.md5(q.strip().lower().encode()).hexdigest()
    qs_q = sorted((q for q in stream if q.strip().endswith("?")), key=key)
    qs_s = sorted((q for q in stream if not q.strip().endswith("?")), key=key)
    out = []
    for a, b in zip(qs_q, qs_s):
        out += [a, b]
    longer = qs_q if len(qs_q) > len(qs_s) else qs_s
    out += longer[min(len(qs_q), len(qs_s)):]
    return out


def binding(p):
    """The per-item loss signal: best kin-carried binding of the prompt into any held
    meaning (identical word = 1, counted kin = 1/2, per query word; the lock gates)."""
    best = 0.0
    for tp in pair_retrieval._taught_pairs():
        b = pair_retrieval.taught_binding(p, tp["prompt"], tcw=set(tp["cw"]))
        if b > best:
            best = b
            if best >= 1.0:
                break
    return best


def teach(p, rnd_label):
    a1 = teacher_answer(p)
    if not a1:
        return False
    with ThreadPoolExecutor(max_workers=2) as ex:
        f2 = ex.submit(teacher_answer, p, a1)
        f3 = ex.submit(teacher_answer, p, a1)
        a2, a3 = f2.result(), f3.result()
    vs = [v for v in (a2, a3) if v and v != a1]
    with _REPLY_LOCK:
        pair_retrieval.add_taught(p, a1, variants=vs)
        held = next((tp for tp in pair_retrieval._taught_pairs()
                     if tp["prompt"].strip().lower() == p.strip().lower()), None)
    # RE-EXPRESSION LAW: a meaning needs >= b = 2 held expressions to serve. If drift
    # rejection left fewer, request one replacement phrasing (one retry, then move on —
    # the next epoch pass reaches this prompt again if still unbound).
    if held is not None and len(held.get("variants", [])) < 2:
        a4 = teacher_answer(p, a1)
        if a4:
            with _REPLY_LOCK:
                pair_retrieval.add_taught(p, a1, variants=[a4])
                held = next((tp for tp in pair_retrieval._taught_pairs()
                             if tp["prompt"].strip().lower() == p.strip().lower()), None)
    with open(TEACH_LOG, "a") as f:
        f.write(json.dumps({"epoch": rnd_label, "prompt": p, "primary": a1,
                            "offered_variants": vs,
                            "held_variants": (held or {}).get("variants", []),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    return True


def judged_rate(prompts, transcripts, feedback=False):
    """Pool verdicts, fanned out. Probes: feedback=False ALWAYS (frozen). Memorization
    sample (taught items): feedback=True marks Laplace counts with captured pids."""
    def one(p):
        with _REPLY_LOCK:
            r = (pair_retrieval.reply(p) or "").strip()
            pids = list(pair_retrieval.last_pids)
        ok, agree = judge_pool(p, r)
        return {"prompt": p, "reply": r, "pool_good": ok, "judges": agree, "pids": pids}
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(one, prompts))
    if feedback:
        with _REPLY_LOCK:
            for row in rows:
                pair_retrieval.mark_feedback(row["pool_good"], pids=row["pids"])
    for row in rows:
        row.pop("pids")
    transcripts.extend(rows)
    return (sum(r["pool_good"] for r in rows) / len(prompts)) if prompts else 0.0


def benchmark(epoch, cursor, taught_total, near_set, far_set):
    """The per-epoch validation suite: memorization sample + frozen near/far under pool
    verdicts, binding coverage (the predictor), binding-stratified transfer."""
    taught = [tp["prompt"] for tp in pair_retrieval._taught_pairs()]
    mem_sample = taught[::max(1, len(taught) // 48)][:48]
    tr = []
    mem = judged_rate(mem_sample, tr, feedback=True)
    near = judged_rate(near_set, tr)
    far = judged_rate(far_set, tr)
    bind_near = [binding(p) >= TAUGHT_LOCK for p in near_set]
    bind_far = [binding(p) >= TAUGHT_LOCK for p in far_set]
    by = {t["prompt"]: t["pool_good"] for t in tr}
    def strat(pset, bset):
        bound = [by[p] for p, b in zip(pset, bset) if b]
        unbound = [by[p] for p, b in zip(pset, bset) if not b]
        return (sum(bound) / len(bound) if bound else None,
                sum(unbound) / len(unbound) if unbound else None)
    gb_n, gu_n = strat(near_set, bind_near)
    gb_f, gu_f = strat(far_set, bind_far)
    row = {"epoch": epoch, "cursor": cursor, "taught_meanings": taught_total,
           "memorization": mem, "near_transfer": near, "far_transfer": far,
           "binding_share_near": sum(bind_near) / len(bind_near),
           "binding_share_far": sum(bind_far) / len(bind_far),
           "good_given_bound_near": gb_n, "good_given_unbound_near": gu_n,
           "good_given_bound_far": gb_f, "good_given_unbound_far": gu_f,
           "transcripts": tr, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"EPOCH {epoch}: mem={mem:.0%} near={near:.0%} far={far:.0%} | "
          f"bind(near)={row['binding_share_near']:.0%} bind(far)={row['binding_share_far']:.0%} | "
          f"GOOD|bound near={gb_n if gb_n is None else f'{gb_n:.0%}'} far={gb_f if gb_f is None else f'{gb_f:.0%}'} | "
          f"taught={taught_total} cursor={cursor}", flush=True)
    return row


def certify():
    """FROZEN-SYSTEM CERTIFICATION (Maria's rule, 2026-07-17): a measured run is valid
    only under a fully-gated, frozen system. Every gate runs HERE, at launch; any
    failure refuses the run. The engine commit is pinned into the ledger's first row —
    any code change mid-run voids the run (fix -> re-gate -> re-freeze -> restart)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    gates = {}
    r = subprocess.run([sys.executable, os.path.join(root, "omni", "core.py")],
                       capture_output=True, text=True, env={**os.environ, "PYTHONPATH": root})
    gates["core_locks"] = (r.returncode == 0)
    for name, script in (("judge_pool", "judge_calibration.py"),
                         ("binding", "binding_calibration.py")):
        r = subprocess.run([sys.executable, os.path.join(here, script)],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": root})
        gates[name] = (r.returncode == 0)
        print(r.stdout.strip().split("\n")[-1], flush=True)
    commit = subprocess.run(["git", "-C", os.path.expanduser("~/Desktop/UnisonAI"),
                             "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ok = all(gates.values())
    print(f"CERTIFICATION: {gates} | system commit {commit[:10]} -> "
          f"{'CERTIFIED' if ok else 'REFUSED'}", flush=True)
    if ok:
        with open(LEDGER, "a") as f:
            f.write(json.dumps({"certification": gates, "system_commit": commit,
                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    return ok


def main():
    os.makedirs(LOGS, exist_ok=True)
    if not PREFLIGHT and not certify():
        print("SYSTEM NOT CERTIFIED — run refused.", flush=True)
        return
    for path in (TAUGHT_PATH, QUALITY_PATH):
        if os.path.exists(path) and not os.path.exists(path + ".pre_epochs_backup"):
            shutil.copy(path, path + ".pre_epochs_backup")
    stream = build_stream()
    # contamination check: no probe verbatim in the stream OR the raw corpus
    P = pair_retrieval._pairs()
    plow = {q.strip().lower() for q in P["prompts"]}
    dirty = [q for q in NEAR + FAR if q.strip().lower() in plow]
    if dirty:
        print(f"CONTAMINATED PROBES (verbatim in corpus): {dirty} — ABORT", flush=True)
        return
    print(f"contamination check: clean | stream={len(stream):,} prompts "
          f"(filtered from {len(P['prompts']):,})", flush=True)

    batch_n, batches_per_epoch = BAND, BATCHES_PER_EPOCH
    near_set, far_set = NEAR, FAR
    if PREFLIGHT:
        batch_n, batches_per_epoch = 4, 2
        near_set, far_set = NEAR[:4], FAR[:4]
        print("=== PREFLIGHT: 2 batches of 4, 4+4 probes, then exit ===", flush=True)

    st = {"cursor": 0, "epoch": 0, "stream_v": 2}
    if os.path.exists(STATE) and not PREFLIGHT:
        old = json.load(open(STATE))
        if old.get("stream_v") == 2:
            st = old
            print(f"resuming: epoch {st['epoch']}, cursor {st['cursor']:,}", flush=True)
        else:
            st["epoch"] = old.get("epoch", 0)
            print(f"stream rebuilt (v2, act-balanced, taught-excluded): cursor reset, "
                  f"epoch continues from {st['epoch']}", flush=True)
    print(f"=== EPOCH TRAINING (translated): batch={batch_n}, {batches_per_epoch} batches/epoch, "
          f"loss=training-judge verdict, stop=near&far>={TAUGHT_LOCK} for {PERSIST} epochs, "
          f"POOL validation ===", flush=True)

    streak = 0
    while True:
        st["epoch"] += 1
        taught_e = skipped_e = 0
        for _ in range(batches_per_epoch):
            batch = stream[st["cursor"]:st["cursor"] + batch_n]
            st["cursor"] += len(batch)
            if not batch:
                print("STREAM EXHAUSTED", flush=True)
                break
            # PER-ITEM LOSS = THE OBJECTIVE, measured on the item (the honest 1-1: in
            # gradient training the loss IS the objective per item). The engine replies
            # (ms, serialized), the training judge verdicts (parallel, ~5 min for the
            # whole epoch on the 8-way server). Binding is NOT the loss — it is the
            # serving router (calibration-gated) and the measured onset PREDICTOR.
            # Failed proxy generations are banked in binding_calibration.py.
            def item_loss(pr):
                with _REPLY_LOCK:
                    r = (pair_retrieval.reply(pr) or "").strip()
                    pids = list(pair_retrieval.last_pids)
                ok, _ = judge(pr, r)
                return pr, ok, pids
            with ThreadPoolExecutor(max_workers=8) as ex:
                verdicts = list(ex.map(item_loss, batch))
            with _REPLY_LOCK:
                for _, ok, pids in verdicts:
                    pair_retrieval.mark_feedback(ok, pids=pids)
            todo = [pr for pr, ok, _ in verdicts if not ok]
            skipped_e += len(batch) - len(todo)
            with ThreadPoolExecutor(max_workers=8) as ex:
                taught_e += sum(ex.map(lambda p: teach(p, st["epoch"]), todo))
        n_held = len(pair_retrieval._taught_pairs())
        print(f"epoch {st['epoch']} pass: taught {taught_e}, already-bound {skipped_e}, "
              f"held {n_held}", flush=True)
        row = benchmark(st["epoch"], st["cursor"], n_held, near_set, far_set)
        if not PREFLIGHT:
            json.dump(st, open(STATE, "w"))
        streak = streak + 1 if (row["near_transfer"] >= TAUGHT_LOCK
                                and row["far_transfer"] >= TAUGHT_LOCK) else 0
        if streak >= PERSIST:
            print(f"=== CRITERION MET: near & far >= {TAUGHT_LOCK} for {PERSIST} consecutive "
                  f"epochs. STOPPING — evidence to Maria; the harness declares nothing. ===",
                  flush=True)
            break
        if PREFLIGHT:
            print("=== PREFLIGHT COMPLETE ===", flush=True)
            break
        if st["cursor"] >= len(stream):
            print("=== STREAM EXHAUSTED before criterion — full report in ledger ===", flush=True)
            break


if __name__ == "__main__":
    main()
