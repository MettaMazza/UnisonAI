"""Conversational fluency substrate — dialogue/assistant-chat, NOT literary prose.
The engine's task is conversational coherence, so its fluency + coupling stores must be
built from conversation. Streams chat datasets and writes flat conversational text to
conv_corpus.txt (turns joined), which build_fluency.py / build_coupling.py then consume."""
import os, sys
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "conv_corpus.txt")
STAGE = OUT + ".building"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 80_000_000

# assistant-chat + casual dialogue, in register with Unison's warm first-person voice
SOURCES = [
    ("HuggingFaceH4/ultrachat_200k", "train_sft"),
    ("OpenAssistant/oasst1", "train"),
    ("daily_dialog", "train"),
]


def turns_of(ex):
    for key in ("messages", "conversation", "conversations", "dialog", "turns"):
        v = ex.get(key)
        if isinstance(v, list):
            out = []
            for m in v:
                if isinstance(m, dict):
                    out.append(str(m.get("content") or m.get("text") or m.get("value") or ""))
                elif isinstance(m, str):
                    out.append(m)
            return [t for t in out if t]
    for key in ("text", "content"):
        if isinstance(ex.get(key), str) and ex[key].strip():
            return [ex[key]]
    return []


written = 0
try:
    with open(STAGE, "w", encoding="utf-8") as f:
        for dsid, split in SOURCES:
            if written >= CAP:
                break
            try:
                print(f"streaming {dsid} [{split}] ...", flush=True)
                ds = load_dataset(dsid, split=split, streaming=True)
                for ex in ds:
                    for t in turns_of(ex):
                        f.write(t.strip()); f.write("\n")
                        written += len(t) + 1
                    f.write("\n")
                    if written % 10_000_000 < 200:
                        print(f"  ...{written//1_000_000} MB", flush=True)
                    if written >= CAP:
                        break
                print(f"  {dsid}: total now {written//1_000_000} MB", flush=True)
            except Exception as e:
                print(f"  failed {dsid}: {repr(e)[:150]}", flush=True)
    if written <= 0:
        raise RuntimeError("all conversational sources failed; refusing an empty corpus")
    os.replace(STAGE, OUT)
    print(f"DONE: {written:,} bytes conversational -> {OUT}", flush=True)
except BaseException:
    try:
        os.remove(STAGE)
    except FileNotFoundError:
        pass
    raise
