"""THE FULL-POOL SOTA SWEEP: every local model on this machine, plus
unison-own, through the engine's OWN registered instrument
(run_sota_bench) -- same probe cache, same counted scoring, same tsv
schema. Nothing duplicated, nothing re-implemented, so the rows cannot
drift from the engine's.

LIVE-FLIGHT SAFETY (the standing law): the engine module rotates
logs/unison.log to archive at import -- that file belongs to the LIVE
flight. This runner therefore loads the engine source with LOGFILE
redirected to logs/sota_sweep.log BEFORE the rotation block runs (one
asserted replacement; everything else byte-identical). All engine state
loads read-only; no watcher threads start (they live in main() only);
RELAY stays off, so the unison-own row is truly own-channels.

Rows post to the locked Discord channel as they land, via the same bot
token the face uses.

POOL NOTE (honest record): DeepSeek-R1-671B is NOT in this pool. Its
q4 weights are ~404 GB; this machine has 162 GB of free disk. It fits
the 549 GB RAM but cannot be pulled. Its PUBLISHED MMLU number appears
in SOTA_TABLE.md's published column, cited, clearly marked as measured
elsewhere.
"""
import importlib.util, json, os, sys, urllib.request

BASE = "/Users/mettamazza/Desktop/Smithian Fold Theory"
ENGINE = BASE + "/fold_ai/unison_chat.py"

# --- load the engine with its log target redirected (live flight owns unison.log)
src = open(ENGINE).read()
OLD = 'LOGFILE = LOGDIR + "/unison.log"'
NEW = 'LOGFILE = LOGDIR + "/sota_sweep.log"'
assert src.count(OLD) == 1, "LOGFILE anchor drifted -- refusing to load"
src = src.replace(OLD, NEW)
spec = importlib.util.spec_from_loader("unison_engine", loader=None, origin=ENGINE)
U = importlib.util.module_from_spec(spec)
U.__file__ = ENGINE
sys.modules["unison_engine"] = U
print("loading engine state (read-only)...", flush=True)
exec(compile(src, ENGINE, "exec"), U.__dict__)
assert U.LOGFILE.endswith("sota_sweep.log"), "log redirect failed -- refusing to run"
assert not U.RELAY["on"], "relay must stay off for the own-channels row"
print("engine loaded:", sum(len(s) for s in U.stores), "orbits;",
      len(U.SENTS), "held sentences", flush=True)

# --- Discord poster: same token and locked channel as the face
TOKEN_PATH = os.path.expanduser("~/.unison_discord_token")
CHANNEL = 1523685773998555227
def _post(msg):
    tok = open(TOKEN_PATH).read().strip()
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{CHANNEL}/messages",
        data=json.dumps({"content": msg[:1900]}).encode(),
        headers={"Authorization": "Bot " + tok, "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()
if os.path.exists(TOKEN_PATH):
    U.ANNOUNCE[0] = lambda m: _post(m)

# --- the pool: everything actually local, giants first (unison-own is
#     prepended by run_sota_bench itself)
MODELS = ["gemma4:26b", "qwen3.6-27b:latest", "gpt-oss-20b:latest",
          "qwen3:8b", "llama3.2:3b", "llama3.2:1b"]

if __name__ == "__main__":
    n = U.GEN_B ** 7   # the full committed probe: 2^7 = 128 items
    print(U.run_sota_bench(models=MODELS, n=n), flush=True)
