"""Stream a large public-domain corpus (Project Gutenberg, PG-19) to a flat text
file as fuel for the engine's distributional kinship. Byte-capped; resumable-safe
(overwrites). Public domain -> no license friction, same literary domain as held-out."""
import sys, os
from datasets import load_dataset

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000_000   # default 400MB
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pg_corpus.txt")

CANDIDATES = [("pg19", None), ("sedthh/gutenberg_english", None), ("manu/project_gutenberg", None)]

def text_of(ex):
    best = ""
    for v in ex.values():
        if isinstance(v, str) and len(v) > len(best):
            best = v
    return best

for dsid, _ in CANDIDATES:
    try:
        print(f"trying {dsid} ...", flush=True)
        ds = load_dataset(dsid, split="train", streaming=True)
        written = 0
        with open(OUT, "w", encoding="utf-8") as f:
            for ex in ds:
                t = text_of(ex)
                if not t:
                    continue
                f.write(t); f.write("\n\n")
                written += len(t)
                if written % 20_000_000 < len(t):
                    print(f"  ...{written//1_000_000} MB", flush=True)
                if written >= CAP:
                    break
        print(f"DONE: {written:,} bytes from {dsid} -> {OUT}", flush=True)
        break
    except Exception as e:
        print(f"failed {dsid}: {repr(e)[:200]}", flush=True)
