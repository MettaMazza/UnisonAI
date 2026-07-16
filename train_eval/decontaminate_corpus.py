"""Remove held-out leakage from the training corpus. The held-out is the Tom Sawyer
tail; a Gutenberg dump contains Tom Sawyer in full, so it must be excised or the
'generalisation' number is just memorised test data. Method: shingle the held book
(corpus_raw.txt) into 40-char spans; scan pg_corpus; any position that matches a
held shingle has its +/- 600KB neighbourhood removed (excises whole book copies).
Writes pg_corpus_clean.txt and verifies zero held-phrase hits."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
SH = 40           # shingle length (chars)
STRIDE = 13       # sampling stride for both build + scan
PAD = 600_000     # remove this much around each match (excise a whole book)


def clean(t):
    t = t.replace("\r\n", "\n").lower()
    return t


def match_positions(text, shingles, sh, stride):
    marks = []; i = 0; n = len(text)
    while i < n - sh:
        if text[i:i + sh] in shingles:
            marks.append(i); i += sh
        else:
            i += stride
    return marks


def excise(text, marks, pad):
    intervals = []
    for m in marks:
        lo, hi = max(0, m - pad), min(len(text), m + pad)
        if intervals and lo <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], hi)
        else:
            intervals.append([lo, hi])
    out = []; prev = 0
    for lo, hi in intervals:
        out.append(text[prev:lo]); prev = hi
    out.append(text[prev:])
    return "".join(out), sum(hi - lo for lo, hi in intervals), len(intervals)


def main():
    tom = clean(open(os.path.join(HERE, "corpus_raw.txt"), encoding="utf-8").read())
    # full-book shingles (excise whole Tom Sawyer copies) + dense held-eval shingles (guarantee)
    book_sh = set(tom[i:i + SH] for i in range(0, len(tom) - SH, STRIDE))
    cut = int(len(tom) * 0.90)
    held = tom[cut:]                       # the eval tail
    held_sh = set(held[i:i + 25] for i in range(0, len(held) - 25, 5))
    print(f"book shingles {len(book_sh):,} | held shingles {len(held_sh):,}", flush=True)

    pg = clean(open(os.path.join(HERE, "pg_corpus.txt"), encoding="utf-8", errors="replace").read())
    # pass 1: excise whole-book copies
    pg, rem, nreg = excise(pg, match_positions(pg, book_sh, SH, STRIDE), PAD)
    print(f"pass1 removed {rem/1e6:.1f} MB / {nreg} regions -> {len(pg)/1e6:.0f} MB", flush=True)
    # iterate: remove any residual held-eval shingle neighbourhood until zero
    for it in range(6):
        marks = match_positions(pg, held_sh, 25, 5)
        if not marks:
            print(f"pass{it+2}: 0 residual held shingles -- clean", flush=True); break
        pg, rem, nreg = excise(pg, marks, 100_000)
        print(f"pass{it+2}: {len(marks)} residual -> removed {rem/1e6:.2f} MB, now {len(pg)/1e6:.0f} MB", flush=True)

    outp = os.path.join(HERE, "pg_corpus_clean.txt")
    open(outp, "w", encoding="utf-8").write(pg)
    print(f"clean corpus: {len(pg)/1e6:.0f} MB -> {outp}", flush=True)
    for p in ["choking sensation in his throat", "implored him to come back every little while",
              "adventures of tom sawyer", "hollow world after", "roused. she said she would wait"]:
        print(f"  verify '{p[:28]}...' hits: {pg.count(p)}", flush=True)


if __name__ == "__main__":
    main()
