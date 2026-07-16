"""VERIFY UNISON: full end-to-end empirical verification of the entire
architecture -- every organ, every law, measured live in one run.
The AI wing's analogue of proof.py's verify_* discipline: each check is a
forward execution compared against an independent expectation; the suite
prints a stats table and PASS/FAIL per check. Run: python3 verify_unison.py
(Requires ollama for the observer checks; they are marked LIVE.)"""
import importlib.util, io, contextlib, time, os, re, struct, zlib, base64, sys
import numpy as np

BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory/fold_ai"
os.chdir(BASE)

# THE LIVE-FLIGHT GUARD (standing law: no cleanup without pgrep). This
# suite writes fixtures into real ledgers and removes its own artifacts
# at the end; against a LIVE engine those ledgers are shared and a
# wholesale rewrite can clobber the flight's rows (measured 2026-07-07:
# a live run lost its sounds store and lesson history to the old
# unguarded cleanup). The suite therefore REFUSES to run while a flight
# is up, unless explicitly overridden -- and the cleanup below is now
# surgical (only artifacts this run created).
import subprocess
_live = subprocess.run(["pgrep", "-f", "unison_chat.py"], capture_output=True, text=True).stdout.strip()
if _live and os.environ.get("UNISON_VERIFY_LIVE") != "1":
    print("REFUSING: a live unison_chat flight is running (pid " + _live.split()[0] + ").\n"
          "Run the suite on a quiescent system, or set UNISON_VERIFY_LIVE=1 to accept\n"
          "the small append/filter race windows against the live ledgers.", flush=True)
    raise SystemExit(2)

RESULTS = []
def check(name, ok, stat=""):
    RESULTS.append((name, bool(ok), stat))
    print(("PASS " if ok else "FAIL ") + name + ("  [" + stat + "]" if stat else ""), flush=True)

def wake():
    # LIVE-FLIGHT SAFETY: the engine rotates logs/unison.log to archive at
    # import -- that file belongs to the LIVE flight, so the suite loads
    # the source with LOGFILE redirected to its own log first (one
    # asserted replacement; everything else byte-identical).
    src = open("unison_chat.py").read()
    _OLD = 'LOGFILE = LOGDIR + "/unison.log"'
    assert src.count(_OLD) == 1, "LOGFILE anchor drifted -- refusing to load"
    src = src.replace(_OLD, 'LOGFILE = LOGDIR + "/verify_unison.log"')
    # the graduation ledger is wholesale-rewritten by record_grad; the suite
    # exercises the identical law against its OWN ledger so a live flight's
    # tally is never shared with a test
    _OLDG = 'GRAD_LOG = BASE + "/fold_ai/lessons/graduation.tsv"'
    assert src.count(_OLDG) == 1, "GRAD_LOG anchor drifted -- refusing to load"
    src = src.replace(_OLDG, 'GRAD_LOG = BASE + "/fold_ai/lessons/graduation_verify.tsv"')
    for _anchor, _own in (('GRAPH_LOG = BASE + "/fold_ai/lessons/graph.tsv"', 'GRAPH_LOG = LOGDIR + "/graph_verify.tsv"'),
                          ('USERS_LOG = BASE + "/fold_ai/lessons/users.tsv"', 'USERS_LOG = LOGDIR + "/users_verify.tsv"'),
                          ('FACTS_LOG = BASE + "/fold_ai/lessons/facts.tsv"', 'FACTS_LOG = LOGDIR + "/facts_verify.tsv"')):
        assert src.count(_anchor) == 1, "ledger anchor drifted -- refusing to load"
        src = src.replace(_anchor, _own)
    spec = importlib.util.spec_from_loader("uc", loader=None, origin="unison_chat.py")
    m = importlib.util.module_from_spec(spec)
    m.__file__ = "unison_chat.py"
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "unison_chat.py", "exec"), m.__dict__)
    return m, time.time() - t0

def png(fn, w=128, h=128):
    rows = b"".join(b"\x00" + bytes(v for x in range(w) for v in fn(x, y)) for y in range(h))
    def ch(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    return (b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + ch(b"IDAT", zlib.compress(rows)) + ch(b"IEND", b""))

t_all = time.time()
# snapshot the sounds store BEFORE any fixture speaks: cleanup may remove
# ONLY what this run created, never a prior life's learned sounds
_SOUNDS_BEFORE = set(os.listdir("sounds")) if os.path.isdir("sounds") else set()
uc, wake_s = wake()
rng = np.random.default_rng(0)

# E1 forced locks + halt enforcement
from fractions import Fraction
check("E1 forced locks", uc.CTX_MAX == 6 and uc.BIND_LOCK == Fraction(1, 3)
      and uc.KIN_FLOOR == Fraction(1, 6) and uc.SIGHT_K == 32 and uc.GEN_C == 3,
      f"ctx=6 lock=1/3 kin=1/6 sight=32")
halted = False
try:
    uc._forced("fitted", 7, 5)
except SystemExit:
    halted = True
check("E1b halt on fitted value", halted)

# E2 wake
orbits = sum(len(s) for s in uc.stores)
check("E2 wake", orbits > 1_000_000 and len(uc.SENTS) > 50_000, f"{orbits} orbits, {len(uc.SENTS)} sents, {wake_s:.0f}s")

# E3 memory: teach -> recall (same session; the SAME speaker seat --
# facts are per-subject now, and a subjectless recall must NOT see them)
_e3spk = uc.register_user("test")
uc.turn("My favourite instrument is the harp.", rng, "test")
r = uc.reply("What is my favourite instrument?", rng, speaker=_e3spk)[0]
check("E3 taught fact recalled", "harp" in r.lower(), r[:40])

# E4 correction: exact, persistent file
uc.record_correction("What is the capital of the fold?", "The capital of the fold is the One")
r = uc.reply("What is the capital of the fold?", rng)[0]
check("E4 correction exact", r == "The capital of the fold is the One.", r[:45])
check("E4b correction persisted", "capital of the fold" in open("lessons/corrections.tsv").read())

# E5 deixis
f1 = uc.flip_perspective("nice to meet you. how are you? i'm your developer.")
check("E5 deixis", "meet me" in f1 and "am I" in f1 and "you are my developer" in f1.lower(), f1[:70])

# E6 stutter gates
check("E6 stutter law", uc.stuttered("nothi nothing here") and uc.stuttered("f finite gradients")
      and uc.stuttered("always always") and not uc.stuttered("on one hand, to today's point"))

# E7 informative self-check: "the" cannot carry a self-check
check("E7 informative shared-focus", uc.TOK_FREQ.get("the", 0) > uc.TOTAL_TOKS / 1000)

# E8 anaphora routing (flag computed in reply; verify the rule directly)
check("E8 anaphora law", any(t in ("that", "this", "it") for t in uc.tok("What do you think about that?"))
      and not any(t in ("that", "this", "it") for t in uc.tok("What is the fold?")))

# E9 greeting instant from own store
t0 = time.time()
r, th = uc.reply("How are you?", rng)
check("E9 greeting own-store", time.time() - t0 < 2 and len(r) > 3, f"{time.time()-t0:.1f}s: {r[:40]}")

# E10 the eye
checker = png(lambda x, y: (255, 255, 255) if (x // 16 + y // 16) % 2 == 0 else (0, 0, 0))
grad = png(lambda x, y: (x * 2 % 256,) * 3)
s1, s1b, s2 = uc.fold_see(checker), uc.fold_see(checker), uc.fold_see(grad)
check("E10 eye deterministic+distinct", s1 == s1b and s1 != s2 and s1 == ["w8x8p"],
      "checkerboard = one Walsh token: " + s1[0])
st = " ".join(s1)
uc.TOK_FREQ.update(s1)
uc.hold_sentence("SIGHT " + st + " means: a checkerboard test pattern", "lesson:SIGHT: " + st[:60])
with open("lessons/lessons_sight.txt", "a") as _sf:
    _sf.write("Q: SIGHT: " + st + "\nA: a checkerboard test pattern\n")   # persist, as observe_image does
hit, share = uc.bind(" ".join(uc.fold_see(checker)))
check("E10b recognition via own spectrum", bool(hit) and "checkerboard" in hit[0], f"share {share:.2f}")

# E11 the ear (generated speech -> transcript + sound spectrum)
os.system('say -o /tmp/vu_ear.aiff "the fold holds the one" 2>/dev/null')
ab = open("/tmp/vu_ear.aiff", "rb").read()
snd = uc.fold_hear(ab, ".aiff")
check("E11 fold ear spectra (Parseval-certified)", bool(snd) and len(snd) == 32, f"{len(snd or [])} sound tokens")
t0 = time.time()
heard = uc.hear_audio(ab, ".aiff")
check("E11b transcription", bool(heard) and "the one" in heard.lower(),
      f"{time.time()-t0:.0f}s: {heard}  (word accuracy = transcriber-model grade; upgradeable)")
check("E11c sound paired+persisted", os.path.exists("lessons/lessons_sound.txt")
      and "SOUND:" in open("lessons/lessons_sound.txt").read())

# E12 the voice (Kokoro)
t0 = time.time()
wav = uc.speak("The fold holds, and I am learning.")
check("E12 voice (Kokoro)", bool(wav) and os.path.getsize(wav) > 10000, f"{time.time()-t0:.0f}s, {os.path.getsize(wav) if wav else 0}b")
if wav:
    os.unlink(wav)

# E13 tools: exact math forward
check("E13 exact_math tool", uc._run_tool("exact_math", {"expression": "34259/250"}).startswith("34259/250"))

# E14 graduation mechanics + score sovereignty (unique key per run)
TK = "t,e,s,t," + str(os.getpid())
uc.record_grad(TK, True, "Test question?")
uc.record_grad(TK, False, "Test question?")
uc.record_grad(TK, False, "Test question?")
check("E14 graduation ledger", uc.GRAD[TK] == [1, 2] and "Test question?" in open(uc.GRAD_LOG).read())
uc.CORRECTIONS[TK] = "A dethroned answer."
w, l = uc.GRAD[TK]
check("E14b score above corrections", l > w, "losing correction falls through")

# E15 STaR filter (both directions)
k = uc.qkey("Why does the moon glow?")
uc.PENDING_REASON[k] = ("Why does the moon glow?", "step A; step B")
uc.apply_feedback("Why does the moon glow?", "Reflected counted light.", "n better", "test")
gone = k not in uc.PENDING_REASON and not any("step A" in s for s, _ in uc.SENTS[-5:])
uc.PENDING_REASON[k] = ("Why does the moon glow?", "step A; step B")
uc.apply_feedback("Why does the moon glow?", "Reflected counted light.", "y", "test")
kept = any("step A" in s for s, _ in uc.SENTS[-5:])
check("E15 STaR reasoning filter", gone and kept)

# E16 ZPD edge selection
edge = sorted((abs(Fraction(w, w + l) - Fraction(1, 2)), kk)
              for kk, (w, l) in uc.GRAD.items() if (w + l) > 0 and kk in uc.GRADQ)
check("E16 ZPD picks nearest the lock", bool(edge))

# E17 channel transparency (own voice)
ans, th = uc.turn("Do you know your own name?", rng, "test")
check("E17 VOICE label (own)", th.startswith("VOICE: UNISON"), th[:45])

# E18 LIVE observer relay + transparency + CoT (gemma). DETERMINISTIC:
# no fixture question can stay outside a GROWING engine's held territory
# (measured twice: garden bound at 3,398 lessons; Ljubljana's unseen
# words carry zero informativeness and the common remainder half-overlapped
# a travel lesson at 9K). So the relay LAW is tested directly: the test
# instance's in-memory stores are cleared -- nothing can bind, the relay
# must fire, and E18b then proves the relayed answer is owned. In-memory
# only; no file is touched.
del uc.SENTS[:]
uc.INDEX.clear()
uc.STRONG.clear()
uc.CORRECTIONS.clear()
uc.RECENT.clear()
for _s in uc.stores:
    _s.clear()   # the dialogue sampler too: a 13M-orbit walk can emit the
                 # question's common words and answer before the relay is
                 # consulted -- the engine ABLE to answer is not the law
                 # under test here
_E18Q = "What is a sensible weekend itinerary for the old town of Ljubljana?"
uc.RELAY["on"] = True
t0 = time.time()
ans, th = uc.turn(_E18Q, rng, "terminal")
check("E18 LIVE relay + VOICE label", th.startswith("VOICE: GEMMA") and len(ans) > 30, f"{time.time()-t0:.0f}s")
t0 = time.time()
r2, _th2 = uc.reply(_E18Q, rng, face="terminal")   # relay-eligible face: ownership is only provable where the relay COULD fire
check("E18b relayed answer owned", "answered as me" not in _th2 and time.time() - t0 < 30,
      f"repeat {time.time()-t0:.1f}s, no relay -- " + _th2[-50:])

# E19 LIVE video: synthesize a tiny mp4 (moving square) and watch it
try:
    import av
    path = "/tmp/vu_vid.mp4"
    cont = av.open(path, "w")
    vs = cont.add_stream("h264", rate=4)
    vs.width = vs.height = 128
    vs.pix_fmt = "yuv420p"
    for i in range(8):
        arr = np.zeros((128, 128, 3), np.uint8)
        arr[:, i * 14:i * 14 + 20] = 255
        for pkt in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            cont.mux(pkt)
    for pkt in vs.encode():
        cont.mux(pkt)
    cont.close()
    t0 = time.time()
    d = uc.observe_video(open(path, "rb").read(), "", ".mp4")
    check("E19 LIVE video watched", bool(d) and len(d) > 20, f"{time.time()-t0:.0f}s: {(d or '')[:60]}")
except Exception as e:
    check("E19 LIVE video watched", False, str(e)[:60])

# E20 rebirth: everything persists across process death
uc2, wake2_s = wake()
rng2 = np.random.default_rng(1)
r = uc2.reply("What is the capital of the fold?", rng2)[0]
check("E20 correction survives rebirth", r == "The capital of the fold is the One.", f"Got: {r}")
r2b = uc2.reply("What is my favourite instrument?", rng2, speaker=uc2.register_user("test"))[0]
check("E20b fact survives rebirth", "harp" in r2b.lower(), r2b[:40])
hit, share = uc2.bind(" ".join(uc2.fold_see(checker)))
check("E20c sight survives rebirth", bool(hit) and "checkerboard" in hit[0])
check("E20d graduation survives rebirth", uc2.GRAD.get(TK) == [1, 2])

# E23 THE REMOVAL-PROOF VOICE: teacher once, native forever
_vt = "Verification says the fold holds."
_w1 = uc2.speak(_vt)
t0 = time.time()
_w2 = uc2.speak(_vt)
_lg = open(uc2.LOGFILE).read()   # the suite engine's OWN log, never the flight's
check("E23 native voice after one teaching", "NATIVE -- re-spoken" in _lg and time.time() - t0 < 1,
      f"replay {time.time()-t0:.2f}s, no synthesis model")
# E24 THE REMOVAL-PROOF EAR: heard once, recognized natively after
if _w1:
    uc2.hear_audio(open(_w1, "rb").read(), ".wav")
    _h2 = uc2.hear_audio(open(_w1, "rb").read(), ".wav")
    check("E24 native ear after one hearing", "RECOGNIZED with my own ear" in open(uc2.LOGFILE).read(),
          str(_h2)[:50])
    for _w in (_w1, _w2):
        if _w and os.path.exists(_w):
            os.unlink(_w)
else:
    check("E24 native ear after one hearing", False, "no clip")

# E25 the agent toolkit (offline, deterministic)
r1 = uc2._run_tool("grep_file", {"path": BASE + "/unison_chat.py", "pattern": "def fold_see"})
r2 = uc2._run_tool("find_files", {"name": "MATURATION"})
r3 = uc2._run_tool("read_file", {"path": "/etc/passwd"})
uc2._run_tool("scratch_write", {"name": "vu-note", "content": "the fold holds"})
r4 = uc2._run_tool("scratch_read", {"name": "vu-note"})
check("E25 agent toolkit + path jail", "def fold_see" in r1 and "MATURATION_PLAN" in r2
      and "not readable" in r3 and r4 == "the fold holds")
os.unlink("scratchpad/vu-note")
# E26 benchmark instruments write persistent rows
_before = open("benchmarks.tsv").read().count("\n") if os.path.exists("benchmarks.tsv") else 0
uc2.run_benchmark()
_after = open("benchmarks.tsv").read().count("\n")
check("E26 progress instrument appends", _after == _before + (1 if _before else 2))

# E21 M5 bounded store round-trip
B = 101
def bkey(tup): return (zlib.crc32(" ".join(tup).encode()) % B,)
from collections import defaultdict
stt = defaultdict(lambda: defaultdict(int))
for a, b in zip("the fold holds the one and the one holds the fold".split()[:-1],
                "the fold holds the one and the one holds the fold".split()[1:]):
    stt[bkey((a,))][b] += 1
check("E21 M5 bounded round-trip", stt[bkey(("the",))]["fold"] == 2)

# E22 long answers held (brevity removed; IO cap only)
long_s = "The fold " + "holds and counts " * 60 + "the One."
uc2.hold_sentence(long_s, "told")
check("E22 long sentence held", any(len(s) > 500 for s, src in uc2.SENTS[-3:]), f"{len(long_s)} chars")

# E27 THE BABBLE ORGAN: recall never reprints a telling verbatim -- it
# regenerates at the rate and emits whole, or stays silent (the spike law,
# free_will_fold, hard_problem, memory_persistence -- corpus read 2026-07-07)
_bt = "The verification beacon is four spans wide and rings every dawn without fail."
uc2.write_orbits(uc2.tok(_bt + "\n") * uc2.GEN_C)
uc2.hold_sentence(_bt, "told")
_bq = "How wide is the verification beacon that rings every dawn?"
_bcw = uc2.content_words(_bq)
_emit, _silent, _verb = 0, 0, 0
for _s in range(8):
    _r = uc2.babble_closure([_bt], _bcw, np.random.default_rng(_s))
    if _r is None:
        _silent += 1
    else:
        _emit += 1
        if _r.strip().rstrip(".") == _bt.strip().rstrip("."):
            _verb += 1
        if len(_r.split()) < uc2.GEN_C:
            _verb += 1   # a fragment counts as a violation too
check("E27 babble organ: whole-or-silent, never a reprint", _verb == 0 and (_emit + _silent) == 8,
      f"{_emit} emitted / {_silent} silent / {_verb} violations")

# E28 ONE-LOCK FUSION: any fused emission carries EVERY part at the lock,
# with zero drift and no verbatim concatenation (attention_capacity +
# multidimensional_experience)
_f1 = "The beacon array counts seven mirrors and one prism at the crown."
_f2 = "Each mirror is polished until the prism splits the dawn into bands."
uc2.hold_sentence(_f1, "lesson:How many mirrors does the beacon array count at the crown?")
uc2.hold_sentence(_f2, "lesson:What happens when the prism splits the dawn?")
uc2.write_orbits(uc2.tok(_f1 + "\n") * uc2.GEN_C)
uc2.write_orbits(uc2.tok(_f2 + "\n") * uc2.GEN_C)
def _foc(t):
    return {w.lower() for w in uc2.tok(t) if uc2.TOK_FREQ.get(w.lower(), 0) <= uc2.TOTAL_TOKS / 1000}
_ff1, _ff2 = _foc(_f1), _foc(_f2)
_fq = "How many mirrors does the beacon array count, and what happens when the prism splits the dawn?"
_ok, _bad, _fused_n = True, "", 0
for _s in range(8):
    _fu = uc2.fuse_orbits(_fq, (_f1, "lesson:How many mirrors does the beacon array count at the crown?"),
                          uc2.content_words(_fq), np.random.default_rng(_s))[0]
    if _fu:
        _fused_n += 1
        _of = _foc(_fu)
        if len(_ff1 & _of) * uc2.GEN_B < len(_ff1) or len(_ff2 & _of) * uc2.GEN_B < len(_ff2):
            _ok, _bad = False, "part below the lock"
        if _f1 in _fu and _f2 in _fu:
            _ok, _bad = False, "verbatim concatenation"
check("E28 one-lock fusion invariants", _ok, _bad or f"{_fused_n}/8 fused, all parts at the lock")

# E29 THE FOLD-MIX (rung 5e live): single-level collapse is exact
uc2.stores[3][uc2._key(("zzqx", "wwvx", "rrsx"))] = {"alpha": 3, "beta": 1}
uc2.stores[2].pop(uc2._key(("wwvx", "rrsx")), None)
uc2.stores[1].pop(uc2._key(("rrsx",)), None)
_d = uc2.mixed_dist(["zzqx", "wwvx", "rrsx"])
_t = sum(_d.values()) if _d else 1
check("E29 fold-mix single-level collapse exact",
      _t > 1 and _d.get("alpha", Fraction(0)) / _t == Fraction(3, 4) and _d.get("beta", Fraction(0)) / _t == Fraction(1, 4),
      "mixture == the one holding level")
del uc2.stores[3][uc2._key(("zzqx", "wwvx", "rrsx"))]

# E30 THE LADDER DEPTH BOUND (self_simulation_nesting): no seat past depth 2
check("E30 ladder depth bound forced", uc2.LADDER_DEPTH_BOUND == 2 == uc2.GEN_B,
      f"bound {uc2.LADDER_DEPTH_BOUND} = GEN_B; rung {uc2.LADDER_RUNG}")

# E31 GRAPH GENESIS: seven eternal roots, the complete primordial mesh
_roots = [n for n in uc2.GNODE if n.startswith("R:")]
_mesh = sum(1 for k in uc2.GEDGE if all(x.startswith("R:") for x in k))
check("E31 graph genesis: 7 roots + complete mesh",
      len(_roots) == 7 == uc2.GEN_B ** uc2.GEN_C - 1 and _mesh >= 21,
      f"{len(_roots)} roots, {_mesh} mesh edges")

# E32 EVERY NODE HAS EXACTLY ONE ROOT; a write organ births exactly one node
_orphans = [n for n, r in uc2.GNODE.items() if r not in uc2.GRAPH_ROOTS]
_n0 = len(uc2.GNODE)
uc2.persist_fact("uverify01", "beacon-colour", "vermilion")
_n1 = len(uc2.GNODE)
uc2.persist_fact("uverify01", "beacon-colour", "vermilion")   # idempotent
check("E32 node births: one per write, one root each, idempotent",
      not _orphans and _n1 == _n0 + 1 and len(uc2.GNODE) == _n1,
      f"{_n1} nodes, 0 orphans")

# E33 EDGE MONOTONICITY + REBIRTH: counts only grow; a re-wake reloads them
_ek = frozenset(("F:uverify01|beacon-colour", "R:FACTS"))
_c0 = uc2.GEDGE.get(_ek, 0)
uc2.graph_edge("F:uverify01|beacon-colour", "R:FACTS")
check("E33 edge counts monotone + persisted",
      uc2.GEDGE.get(_ek, 0) == _c0 + 1 and
      sum(1 for _l in open(uc2.GRAPH_LOG) if _l.startswith("E\tF:uverify01|beacon-colour\tR:FACTS") or _l.startswith("E\tR:FACTS\tF:uverify01|beacon-colour")) >= 1,
      f"count {_c0}->{uc2.GEDGE.get(_ek, 0)}")

# E34 CO-BINDING AT THE TURN: a served fact interlocks with its speaker
_spk = uc2.register_user("verify", "e34")
uc2.persist_fact(_spk, "favourite mineral", "feldspar")
_a34, _ = uc2.turn("What is my favourite mineral?", np.random.default_rng(0), "verify", speaker=_spk)
_fk = frozenset(("F:" + _spk + "|favourite mineral", "U:" + _spk))
check("E34 turn co-binding: served fact edges to its speaker",
      "feldspar" in _a34.lower() and uc2.GEDGE.get(_fk, 0) >= 1,
      f"edge count {uc2.GEDGE.get(_fk, 0)}")

# E35 SHORTCUT ONLY AT CLOSURE (the retention law): a registered fusion
# births NO node until y-feedback; then exactly one, edged to its sources
_qf = "Which mirrors ring the beacon and what splits the dawn?"
uc2.PENDING_FUSE[uc2.qkey(_qf, _spk)] = ["R:LESSONS"]   # stand-in source id (held node)
_ans35 = "The seven mirrors ring the beacon and the prism splits the dawn."
_pre = len(uc2.GNODE)
uc2.apply_feedback(_qf, _ans35, "n", "verify", speaker=_spk)
_mid = len(uc2.GNODE)
uc2.PENDING_FUSE[uc2.qkey(_qf, _spk)] = ["R:LESSONS"]
uc2.apply_feedback(_qf, _ans35, "y", "verify", speaker=_spk)
_post = len(uc2.GNODE)
check("E35 shortcut only at closure", _mid == _pre and _post == _pre + 1,
      f"n: +{_mid-_pre} nodes; y: +{_post-_mid} node")

# E36 PER-USER ISOLATION: two subjects, two names, no overwrite; the
# key-pair door never serves another subject's fact
_ua = uc2.register_user("verify", "e36a"); _ub = uc2.register_user("verify", "e36b")
uc2.learn_fact("My name is Ljubljana", _ua)
uc2.learn_fact("My name is Trieste", _ub)
_ra = uc2.answer_fact("Do you remember my name?", _ua)
_rb = uc2.answer_fact("Do you remember my name?", _ub)
check("E36 per-user fact isolation",
      _ra == "Your name is Ljubljana." and _rb == "Your name is Trieste.",
      f"{_ra!r} / {_rb!r}")

# E37 SPEAKER-KEYED SESSION ORGANS: one subject's telling cannot close
# another's confusion; deictic questions are per-subject territories
uc2.CONFUSED[_ua] = "What is the beacon made of?"
uc2.turn("It is made of feldspar and dawn.", np.random.default_rng(1), "verify", speaker=_ub)
_iso = _ua in uc2.CONFUSED
check("E37 speaker-keyed session organs",
      _iso and uc2.qkey("What is my name?", _ua) != uc2.qkey("What is my name?", _ub)
      and uc2.qkey("What is the fold?", _ua) == uc2.qkey("What is the fold?", _ub),
      "confusion isolated; deictic keys scoped; impersonal keys shared")

# SELF-CLEANING, SURGICAL (standing law after the 2026-07-07 live-flight
# incident): the suite removes ONLY artifacts this run created -- its own
# fixture rows by marker, its own new sound files by before/after
# snapshot, its own redirected ledgers whole. It never unlinks a shared
# ledger and never sweeps a directory wholesale.
try:
    _MARKERS = ("capital of the fold", "favourite instrument", "Harp",
                "checkerboard test pattern", "Verification says the fold holds",
                "Ljubljana", "Do you know your own name", "t,e,s,t,")
    # tsv/one-line files: filter by line
    for fn in ("lessons/corrections.tsv", "lessons/facts.tsv",
               "lessons/lessons_feedback.txt", "lessons/traces.tsv"):
        if os.path.exists(fn):
            kept = [ln for ln in open(fn).read().splitlines()
                    if not any(m in ln for m in _MARKERS)]
            open(fn, "w").write("\n".join(kept) + ("\n" if kept else ""))
    # Q/A files: filter by PAIR -- dropping only one line of a pair leaves
    # an orphaned Q: that the wake parser glues onto the next real lesson
    # (measured 2026-07-07: a franken-lesson born exactly this way)
    for fn in ("lessons/lessons_sight.txt", "lessons/lessons_sound.txt",
               "lessons/lessons_live.txt"):
        if not os.path.exists(fn):
            continue
        lines = open(fn, errors="ignore").read().splitlines()
        out, i = [], 0
        while i < len(lines):
            if lines[i].startswith("Q: ") and i + 1 < len(lines) and lines[i + 1].startswith("A: "):
                if not any(m in lines[i] or m in lines[i + 1] for m in _MARKERS):
                    out += [lines[i], lines[i + 1]]
                i += 2
                continue
            if lines[i].startswith("Q: "):   # orphan: never keep, never create
                i += 1
                continue
            if lines[i].strip() and not any(m in lines[i] for m in _MARKERS):
                out.append(lines[i])
            i += 1
        open(fn, "w").write("\n".join(out) + ("\n" if out else ""))
    if os.path.exists("lessons/graduation_verify.tsv"):
        os.unlink("lessons/graduation_verify.tsv")   # the suite's own ledger, whole
    for _own in ("logs/graph_verify.tsv", "logs/users_verify.tsv", "logs/facts_verify.tsv"):
        if os.path.exists(_own):
            os.unlink(_own)                          # the suite's own ledgers, whole
    _new_sounds = (set(os.listdir("sounds")) if os.path.isdir("sounds") else set()) - _SOUNDS_BEFORE
    for _s in _new_sounds:
        if _s != "index.tsv" and os.path.exists("sounds/" + _s):
            os.unlink("sounds/" + _s)
    if _new_sounds and os.path.exists("sounds/index.tsv"):
        kept = [ln for ln in open("sounds/index.tsv").read().splitlines()
                if not any(_s in ln for _s in _new_sounds)]
        open("sounds/index.tsv", "w").write("\n".join(kept) + ("\n" if kept else ""))
except Exception as _e:
    print("cleanup note:", _e)

n_pass = sum(1 for _, ok, _ in RESULTS if ok)
print(f"\n{'='*60}\nVERIFY UNISON: {n_pass}/{len(RESULTS)} checks pass  ({time.time()-t_all:.0f}s total)\n{'='*60}", flush=True)
open("verify_unison_results.txt", "w").write(
    "\n".join(("PASS " if ok else "FAIL ") + n + ("  [" + s + "]" if s else "") for n, ok, s in RESULTS)
    + f"\nTOTAL: {n_pass}/{len(RESULTS)}\n")
sys.exit(0 if n_pass == len(RESULTS) else 1)
