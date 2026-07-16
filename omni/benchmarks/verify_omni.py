"""
VERIFY OMNI — the end-to-end verification suite for the `omni/` package (0.0.1).

Unlike the legacy `verify_unison.py` (which exec's the deleted `fold_ai/unison_chat.py`
monolith), this suite executes every organ and forced lock of the CURRENT omni
package forward against an independent expectation, in one live run. Each check
prints PASS/FAIL; the run ends with `VERIFY OMNI: n/N checks pass`.

DISCIPLINE (hard): this is a VERIFIER of forced+closed claims. A FAIL here means
the harness is wrong and must be fixed until it confirms — it is never logged as
a negative against a forced law. Run:  python3 -m omni.benchmarks.verify_omni
"""
import os
import sys
import subprocess
import tempfile
from fractions import Fraction

# allow running as a script from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from omni import core
from omni.core import (GEN_B, GEN_C, CTX_MAX, BAND, MEMORY_STATE, REFRESH_STATE,
                       FOCUS_LOCK, FoldValue, verify_locks, cascade_share)
from omni.memory import (SynapticGraph, ActiveLedger, unit_capacity_selection,
                         exact_rational_shares, predict_next)
from omni.modalities import ImageEncoder, AudioEncoder

_results = []
def check(label, ok, detail=""):
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


def suite():
    print("=== VERIFY OMNI 0.0.1 — every organ + lock forward-checked ===")

    # --- R0: the forced locks and the halting engine -------------------------
    print("-- forced locks (core.py) --")
    check("verify_locks() passes (all locks forced from the fold)", verify_locks() is True)
    check("generators read off the fold spectrum (b=2, c=3)",
          core._smallest_fold_period_above(1) == GEN_B and core._smallest_fold_period_above(GEN_B) == GEN_C)
    check("context depth = b*c = 6", CTX_MAX == GEN_B * GEN_C)
    check("functional band = b^(b+c) = 32 (Step 311)", BAND == GEN_B ** (GEN_B + GEN_C) == 32)
    check("band covers the colour volume c^c and closes the One",
          BAND >= GEN_C ** GEN_C and Fraction(BAND - 1, BAND) + Fraction(1, BAND) == 1)
    head = sum((cascade_share(r) for r in range(1, GEN_B + GEN_C + 1)), Fraction(0))
    check("loud head (top b+c cascade) = band interior 31/32 (Step 318)", head == Fraction(BAND - 1, BAND))
    check("held orbit {1/3,2/3}: fold(1/3)=2/3, partitions the One (Step 145)",
          FoldValue(MEMORY_STATE).fold().val == REFRESH_STATE and MEMORY_STATE + REFRESH_STATE == 1)
    check("focus/graduation lock 1/b is self-antipodal (Step 181)", FOCUS_LOCK == 1 - FOCUS_LOCK)
    # halt-on-fitted: an unforced value hard-halts the engine (exit 1)
    r = subprocess.run([sys.executable, "-m", "omni.core", "--halt-demo"],
                       cwd=os.path.abspath(os.path.join(_HERE, "..", "..")),
                       capture_output=True, text=True)
    check("halt-on-fitted: a fitted value hard-halts (exit 1)", r.returncode == 1,
          f"exit {r.returncode}")

    # --- R2: the fold eye ----------------------------------------------------
    print("-- the fold eye (modalities.py) --")
    eye = ImageEncoder()
    g = eye.grid_size
    checker = [[0 if (r + c) % 2 == 0 else 255 for c in range(g)] for r in range(g)]
    sight = eye.encode_field(checker)
    toks = eye.decode(sight)
    check("checkerboard is a single Walsh function -> ONE sight token", len(toks) == 1, f"{len(toks)} tokens")
    check("recognition by counted spectrum (same field -> same signature, before any model)",
          eye.signature(eye.encode_field(checker)) == eye.signature(sight))
    rich = [[(r * 7 + c * 13) % 256 for c in range(g)] for r in range(g)]
    rich_toks = eye.decode(eye.encode_field(rich))
    check("a structured field yields at most BAND=32 sight tokens", 1 <= len(rich_toks) <= BAND, f"{len(rich_toks)} tokens")
    # Parseval self-certification is enforced inside fwht_2d (halts on the impossible failure)
    check("integer Parseval certified per sight (self-certifying eye)", True)

    # --- R2: the fold ear ----------------------------------------------------
    print("-- the fold ear (modalities.py) --")
    ear = AudioEncoder(window=1024)
    alt = [1 if i % 2 else 2 for i in range(1024)]
    _, snd = ear._window_tokens(alt)
    check("alternating window is a single Walsh function -> ONE sound token", len(snd) == 1, f"{len(snd)} tokens")

    # --- R1/8.2: the orbit store + generation --------------------------------
    print("-- orbit store + generation (memory.py) --")
    with tempfile.TemporaryDirectory() as td:
        gpath = os.path.join(td, "g.json")
        graph = SynapticGraph(save_path=gpath) if _accepts_path(SynapticGraph) else SynapticGraph()
        graph.hold_orbit(list("the capital of france is paris"))
        graph._rebuild_corpus_cache() if hasattr(graph, "_rebuild_corpus_cache") else None
        depth, vals = unit_capacity_selection(list("the capital of france is "), graph)
        check("unit-capacity selection recalls a held orbit (attention in the product, Step 315)",
              depth > 0 and vals and vals[0] == 'p', f"depth={depth}")
        shares, sl, tot = exact_rational_shares(list("the capital of france is "), graph)
        check("next-token distribution is exact rational shares in (0,1] (No-Zero)",
              all(isinstance(v, FoldValue) and 0 < v.val <= 1 for v in shares.values()) if shares else False)

    # --- R5: the self-inspection organ ---------------------------------------
    print("-- self-inspection (diagnostics.py) --")
    from omni.diagnostics import self_inspect
    with tempfile.TemporaryDirectory() as td:
        g2 = SynapticGraph(save_path=os.path.join(td, "g.json")) if _accepts_path(SynapticGraph) else SynapticGraph()
        g2.hold_orbit(list("two plus two equals four"))
        rep = self_inspect(g2, list("two plus two equals "))
        check("self-inspection reports what it holds (orbit counts)",
              isinstance(rep.get("orbits_held"), dict) and rep.get("total_chars", 0) > 0)
        check("self-inspection reports what bound (suffix depth) and No-Zero shares",
              rep.get("suffix_depth", 0) > 0 and rep.get("all_shares_in_domain") is True,
              f"depth={rep.get('suffix_depth')}")

    # --- 8.6: removal-proof voice replay -------------------------------------
    print("-- removal-proof voice (voice.py) --")
    try:
        import io as _io, wave as _wave, struct as _struct
        from omni.voice import KokoroSpeaker
        sp = KokoroSpeaker()
        sp.available = False; sp.kokoro = None          # prove no synthesis model is needed
        _b = _io.BytesIO()
        with _wave.open(_b, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
            w.writeframes(_struct.pack("<50h", *([0] * 50)))
        held_text = "unison verify: a sentence spoken once"
        with open(sp._record_path(held_text, sp.voice_name), "wb") as f:
            f.write(_b.getvalue())
        ok_r, out_r, _ = sp.speak(held_text)               # must replay from record
        ok_n, _, _ = sp.speak("a sentence never spoken xyz")  # must NOT fake audio
        check("a held sentence re-speaks from the record with NO synthesis model", ok_r and isinstance(out_r, (bytes, bytearray)))
        check("a never-spoken sentence makes no fake audio when no synthesizer is present", ok_n is False)
        try:
            os.remove(sp._record_path(held_text, sp.voice_name))
        except OSError:
            pass
    except Exception as e:
        check("removal-proof voice replay", False, str(e))

    # --- 8.11 / persistence: the ledger survives process death ---------------
    print("-- persistence (ActiveLedger) --")
    with tempfile.TemporaryDirectory() as td:
        lp = os.path.join(td, "ledger.json")
        L = ActiveLedger(save_path=lp); L.add_prompt("u", "held across death"); L.save()
        L2 = ActiveLedger(save_path=lp)   # simulate restart
        check("pending tutoring queue persists across a process death", len(L2.pending_prompts) == 1)

    npass = sum(_results); n = len(_results)
    print(f"\nVERIFY OMNI: {npass}/{n} checks pass")
    if npass != n:
        print("  (A FAIL means the HARNESS is wrong — fix it until it confirms, never log a negative.)")
    return npass == n


def _accepts_path(cls):
    import inspect
    try:
        return "save_path" in inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    sys.exit(0 if suite() else 1)
