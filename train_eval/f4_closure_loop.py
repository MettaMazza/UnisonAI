"""F4 — BEST-OF-N WITH TEACHER CLOSURE: the learning law applied to free generation.

Self-play with earned retention (the established lineage), closed by the teacher (the
scaffold's existing correction role — a throughput choice, as documented):

  1. For each stream prompt the FREE ARM generates N = 8 candidates (its own output,
     its own distribution — nothing is retrieved verbatim).
  2. The TEACHER closes: picks the best candidate and minimally repairs it into a
     natural reply (hold -> close by observation — the Learning Law's arc).
  3. The closure is HELD (omni/free_closures.pkl) and FEEDS BACK: closed texts become
     a high-tier conditioning overlay for future generations (verified own-output is
     retained; unverified output is never self-reinforced — the retention law).

Measured effect: the RAW judged rate of the free arm (gen_free_harness, pool judges,
untouched) as closures accumulate. The teacher never scores the scoreboard; the pool
does. Ledger: logs/f4_closures.jsonl (prompt, candidates, choice, closure).

Run: PYTHONPATH=. python3 train_eval/f4_closure_loop.py [n_prompts]
"""
import sys, os, json, re, random, pickle, urllib.request, time, threading
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.free_gen import free_gen
from train_eval.generalisation_epochs import build_stream

OLLAMA = "http://localhost:11434/api/generate"
TEACHER = "gemma-4-31b:latest"
N = 8
HERE = os.path.dirname(os.path.abspath(__file__))
CLOSURES = os.path.abspath(os.path.join(HERE, "..", "omni", "free_closures.pkl"))
LOG = os.path.join(HERE, "..", "logs", "f4_closures.jsonl")
_GEN_LOCK = threading.Lock()


def teacher_close(prompt, candidates):
    listing = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
    ask = (f"A user says: {prompt!r}\nA young language model produced these candidate "
           f"replies:\n{listing}\n\nPick the most promising candidate and minimally "
           f"repair it into ONE natural, warm reply of 1-2 short sentences — keep as "
           f"much of the candidate's own wording as possible. Return ONLY the repaired "
           f"reply, no preamble, no numbering.")
    body = json.dumps({"model": TEACHER, "prompt": ask, "stream": False, "think": False,
                       "options": {"temperature": 0.3}}).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(
                OLLAMA, data=body, headers={"Content-Type": "application/json"}),
                timeout=180) as r:
            out = json.loads(r.read().decode()).get("response", "")
        out = re.sub(r"(?is)<think(?:ing)?>.*?</think(?:ing)?>", "", out).strip()
        out = out.split("\n")[0].strip()
        if out and out[-1] not in ".!?":
            m = re.match(r"^(.*[.!?])[^.!?]*$", out, re.S)
            if m:
                out = m.group(1)
        return out[:300]
    except Exception:
        return ""


def candidates_for(prompt):
    """N candidates from the engine's own free arm — generation is single-threaded
    under a lock (shared caches); it is local and fast."""
    cands = []
    with _GEN_LOCK:
        for i in range(N):
            rng = random.Random(i * 2654435761 + 97)
            try:
                r = (free_gen.generate_planned(prompt, rng=rng) if i % 2 == 0
                     else free_gen.generate(prompt, rng=rng))
                c = r[0] if isinstance(r, tuple) else r
            except Exception:
                c = ""
            c = (c or "").strip()
            if c and c not in cands:
                cands.append(c)
    return cands


def main():
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    stream = build_stream()
    prompts = stream[:n_prompts]
    try:
        held = pickle.load(open(CLOSURES, "rb"))
    except Exception:
        held = []
    seen = {h["prompt"].strip().lower() for h in held}
    prompts = [p for p in prompts if p.strip().lower() not in seen][:n_prompts]
    print(f"F4 closure loop: {len(prompts)} prompts, N={N}, held so far {len(held)}",
          flush=True)

    def work(p):
        cands = candidates_for(p)
        if not cands:
            return None
        closed = teacher_close(p, cands)
        if not closed:
            return None
        return {"prompt": p, "candidates": cands, "closed": closed,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for row in ex.map(work, prompts):
            if row is None:
                continue
            held.append(row)
            with open(LOG, "a") as f:
                f.write(json.dumps(row) + "\n")
            done += 1
            if done % 25 == 0:
                pickle.dump(held, open(CLOSURES, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
                print(f"  {done} closures held", flush=True)
    pickle.dump(held, open(CLOSURES, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print(f"F4 round complete: {done} new closures, {len(held)} total -> {CLOSURES}",
          flush=True)


if __name__ == "__main__":
    main()
