"""LEDGER BACKFILL -- one-time ingestion of the committed rung record into
results.jsonl. Parses only the structured verdict lines each result file
already carries; every row is stamped backfill=true with its source file
and line, so the prose records stay the registration-grade originals and
the ledger rows are their queryable index. Idempotent by content: re-running
skips rows already present (matched on source+line)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldprobe import LEDGER, Run

HERE = os.path.dirname(os.path.abspath(__file__))

BACKFILL_REG = {
    "name": "ledger-backfill",
    "objects": ["llm_presence_results.txt", "rung2f_results.txt",
                "rung2f_rowblock_results.txt", "rung2g_results.txt",
                "rung2h_results.txt", "kimi_results.txt"],
    "statistic": "verbatim margins/medians as committed in the rung record",
    "verdict_rule": "a row is ingested iff its line parses against the file's "
                    "own committed format; unparsed verdict lines are counted "
                    "and reported, never silently dropped",
    "margin_clause": "not applicable -- no new measurement is made; recorded "
                     "margins are carried verbatim",
}

PRESENCE = re.compile(r"^(\S.*?\S)\s{2,}(PASS|FAIL)\s+\((\d)/3 fractions; margin ([\d.]+)x\)")
MEDIAN = re.compile(r"^(\S.*?\S)\s+blocks=(\d+)\s+MEDIAN\s+([\d.]+)x\s+\(min ([\d.]+) max ([\d.]+)\)")
SCALING = re.compile(r"^\s{2}(\S+)\s+(\S+)\s+\[(\w+\d*)\]\s+margin\s+([\d.]+)x")
SCALEFAIL = re.compile(r"^\s{2}(\S+)\s+(\S+): FAIL (.+)$")
BASISMAP = re.compile(r"^\s{2}(\S.*?\S)\s{2,}(\S[\w() -]*?\S)\s+([\d.]+)x(\s+<-- WAKE)?\s*$")


def existing_keys():
    seen = set()
    if os.path.exists(LEDGER):
        with open(LEDGER) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("backfill"):
                    seen.add((row.get("source"), row.get("source_line")))
    return seen


def main():
    run = Run(BACKFILL_REG)
    seen = existing_keys()
    ingested = unparsed = skipped = 0

    def emit(source, lineno, **row):
        nonlocal ingested, skipped
        if (source, lineno) in seen:
            skipped += 1
            return
        run.record(backfill=True, source=source, source_line=lineno, **row)
        ingested += 1

    jobs = (
        ("llm_presence_results.txt", "presence"),
        ("rung2f_results.txt", "scaling"),
        ("rung2f_rowblock_results.txt", "median"),
        ("rung2g_results.txt", "basismap"),
        ("rung2h_results.txt", "median"),
        ("kimi_results.txt", "median"),
    )
    for fname, kind in jobs:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            print(f"  {fname}: missing, skipped", flush=True)
            continue
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if kind == "presence":
                    m = PRESENCE.match(line)
                    if m:
                        emit(fname, lineno, instrument="battery", model="GPT-2",
                             object=m.group(1).strip(), verdict=m.group(2),
                             fractions_passed=int(m.group(3)), margin=float(m.group(4)))
                        continue
                elif kind == "median":
                    m = MEDIAN.match(line.strip())
                    if m:
                        emit(fname, lineno, instrument="rowblocks",
                             object=m.group(1).strip(), blocks=int(m.group(2)),
                             median_margin=float(m.group(3)),
                             min_margin=float(m.group(4)), max_margin=float(m.group(5)))
                        continue
                elif kind == "scaling":
                    m = SCALING.match(line)
                    if m:
                        emit(fname, lineno, instrument="battery", model=m.group(1),
                             object=m.group(2), packing=m.group(3), margin=float(m.group(4)))
                        continue
                    m = SCALEFAIL.match(line)
                    if m:
                        emit(fname, lineno, instrument="battery", model=m.group(1),
                             object=m.group(2), failed=m.group(3),
                             note="non-measurement -- instrument failure, not a null")
                        continue
                elif kind == "basismap":
                    m = BASISMAP.match(line)
                    if m:
                        emit(fname, lineno, instrument="basishunt",
                             object=m.group(1).strip(), map=m.group(2).strip(),
                             margin=float(m.group(3)), wake=bool(m.group(4)))
                        continue
                if re.search(r"margin|MEDIAN", line) and "VERDICT" not in line \
                        and "best margin" not in line and not line.startswith("#"):
                    unparsed += 1
                    print(f"  UNPARSED {fname}:{lineno}: {line.strip()[:90]}", flush=True)

    print(f"\nBACKFILL: {ingested} rows ingested, {skipped} already present, "
          f"{unparsed} verdict-bearing lines unparsed", flush=True)


if __name__ == "__main__":
    main()
