"""Offline conversation test harness: drives the engine's own functions
directly (no subprocess, no stale ledgers) and checks each turn."""
import importlib.util, sys, io, contextlib

# load the engine module but suppress its wake print and skip main().
# LIVE-FLIGHT SAFETY: the engine rotates logs/unison.log to archive at
# import -- that file belongs to the LIVE flight, so the harness loads the
# source with LOGFILE redirected to its own log first (one asserted
# replacement; everything else byte-identical).
_src = open("unison_chat.py").read()
_OLD = 'LOGFILE = LOGDIR + "/unison.log"'
assert _src.count(_OLD) == 1, "LOGFILE anchor drifted -- refusing to load"
_src = _src.replace(_OLD, 'LOGFILE = LOGDIR + "/test_chat.log"')
# FACTS pollution guard (caught live 2026-07-08: every harness run had been
# appending the Maria/Scotland/green fixtures to the FLIGHT's own fact
# store): the harness holds its facts in its own file, same discipline as
# the log redirect above.
_OLDF = 'FACTS_LOG = BASE + "/fold_ai/lessons/facts.tsv"'
assert _src.count(_OLDF) == 1, "FACTS_LOG anchor drifted -- refusing to load"
_src = _src.replace(_OLDF, 'FACTS_LOG = LOGDIR + "/test_facts.tsv"')
# the graph and users ledgers get the same live-flight protection
_OLDG = 'GRAPH_LOG = BASE + "/fold_ai/lessons/graph.tsv"'
assert _src.count(_OLDG) == 1, "GRAPH_LOG anchor drifted -- refusing to load"
_src = _src.replace(_OLDG, 'GRAPH_LOG = LOGDIR + "/test_graph.tsv"')
_OLDU = 'USERS_LOG = BASE + "/fold_ai/lessons/users.tsv"'
assert _src.count(_OLDU) == 1, "USERS_LOG anchor drifted -- refusing to load"
_src = _src.replace(_OLDU, 'USERS_LOG = LOGDIR + "/test_users.tsv"')
import os as _os
for _tf in ("logs/test_facts.tsv", "logs/test_graph.tsv", "logs/test_users.tsv"):
    if _os.path.exists(_tf):
        _os.unlink(_tf)   # each run starts from its own clean slate
spec = importlib.util.spec_from_loader("uc", loader=None, origin="unison_chat.py")
uc = importlib.util.module_from_spec(spec)
uc.__file__ = "unison_chat.py"
with contextlib.redirect_stdout(io.StringIO()):
    # prevent main() from running on import: it's guarded by __main__, fine
    exec(compile(_src, "unison_chat.py", "exec"), uc.__dict__)

import numpy as np
rng = np.random.default_rng(0)

def say(line, who="terminal", pid=None):
    """Replicate the loop's turn logic and return (answer)."""
    spk = uc.register_user(who, pid)
    is_question = line.endswith("?") or line.lower().startswith(
        ("what","who","how","why","when","where","do ","does","did","can ","is ","are "))
    import re as _re
    is_command = bool(_re.match(r"(?i)\s*(say|repeat after me|respond with|reply with)\b", line))
    if not is_question and not is_command:
        if not uc.content_words(line):
            return "okay."
        got = uc.learn_fact(line, spk)
        fact = uc.flip_perspective(line if line[-1:] in ".!" else line + ".")
        uc.write_orbits(uc.tok(fact + "\n") * 3)
        uc.hold_sentence(fact, "told")
        if got:
            return "Held. fact"
        return "Held: " + fact
    ans, _ = uc.reply(line, rng, speaker=spk)
    return ans

TESTS = [
    ("My name is Maria",              lambda a: "held" in a.lower()),
    ("What is my name?",              lambda a: a == "Your name is Maria."),
    # the live miss of 2026-07-08 04:31: an unenumerated phrasing must
    # still route to the held fact (the key-pair door)
    ("Do you remember my name?",      lambda a: a == "Your name is Maria."),
    # a mid-message declaration with a negation and a run-on clause must
    # bind the NAME, whole and bounded (the live miss of 04:11)
    ("That is fine, No my name is not Julian, It is Maria Smith. My name Is Maria Smith i am your systems developer",
                                      lambda a: "held" in a.lower()),
    ("Do you remember my name?",      lambda a: a == "Your name is Maria Smith."),
    ("I live in Scotland",            lambda a: "held" in a.lower()),
    ("Where do I live?",              lambda a: a == "You live in Scotland."),
    ("My favourite colour is green",  lambda a: "held" in a.lower()),
    ("What is my favourite colour?",  lambda a: "green" in a.lower() and "A:" not in a),
    ("What is the fold?",             lambda a: "One" in a or "fold" in a.lower()),
    ("Say yes",                       lambda a: a.strip().lower() == "yes."),
    ("Repeat after me: the fold is one", lambda a: a.strip().lower() == "the fold is one."),
]
passed = 0
for line, check in TESTS:
    a = say(line)
    ok = check(a)
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {line!r:40s} -> {a!r}")

# THE SUBJECT AXIS (run 5): two users, two closures -- distinct held
# structures, order-independent, never overwriting each other
USER_TESTS = [
    ("My name is Alice",     "discord", 1001, lambda a: "held" in a.lower()),
    ("My name is Bob",       "discord", 2002, lambda a: "held" in a.lower()),
    ("What is my name?",     "discord", 1001, lambda a: a == "Your name is Alice."),
    ("What is my name?",     "discord", 2002, lambda a: a == "Your name is Bob."),
    ("Do you remember my name?", "discord", 1001, lambda a: a == "Your name is Alice."),
    # the terminal seat's fact is untouched by the discord subjects
    ("What is my name?",     "terminal", None, lambda a: a == "Your name is Maria Smith."),
]
for line, who, pid, check in USER_TESTS:
    a = say(line, who, pid)
    ok = check(a)
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] u{pid or 'term'}: {line!r:32s} -> {a!r}")
TESTS_TOTAL = len(TESTS) + len(USER_TESTS)
print(f"\n{passed}/{TESTS_TOTAL} turns pass")
