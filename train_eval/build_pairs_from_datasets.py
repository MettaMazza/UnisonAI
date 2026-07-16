"""Stage 1 (corrected) — extract PROMPT->RESPONSE pairs from the datasets' OWN turn structure.

The text-file intermediate (conv_corpus.txt) is line-oriented and loses turn boundaries
(multi-paragraph turns split into lines -> the pair builder paired adjacent lines inside one
essay; judged ceiling 6%). The established pipeline extracts pairs from the datasets'
explicit roles/turn lists — unambiguous user->assistant / speaker-alternation pairs.

Sources (vetted allowlist, dialogue-structured): daily_dialog, allenai/soda,
google/Synthetic-Persona-Chat, HuggingFaceH4/no_robots, HuggingFaceH4/ultrachat_200k
(assistant turns truncated to their first sentences — response-selection practice for
long-form sources). Same register/leak filters as before; the standalone filter is relaxed
for genuine dialogue acts ("Sure, I'd love to!") because true pairing restores their context.

Saves omni/pairs.pkl in the same schema as build_pairs.py.
Run: PYTHONPATH=. python3 train_eval/build_pairs_from_datasets.py
"""
import os, sys, re, pickle, time, hashlib
from collections import defaultdict, Counter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize, _content_words

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "pairs.pkl"))
CAP_PER_DS = 160_000
CAP_ULTRACHAT = 260_000        # knowledge coverage: the topical source gets a larger share
CAP_TOTAL = 800_000
_CODE = set("{}[]<>=`~^|\\")
_LEAK = ("main function", "prompt the user", "variable named", "best regards", "sincerely",
         "sign off", "conclude by", "as follows", "click here", "dear friend", "http", "www.",
         ".com", "step 1", "step 2", "in this article", "in this tutorial", "as an ai",
         "here are", "here's a list", "the following", "just a bot", "i am a bot",
         "i'm a bot", "language model", "(name", "[name", "(user", "[user")

DATASETS = [
    ("OpenRL/daily_dialog", "train"),                 # parquet mirror — clean casual dialogue
    ("google/Synthetic-Persona-Chat", "train"),       # "User N:"-markered chit-chat
    ("allenai/soda", "train"),
    ("HuggingFaceH4/no_robots", "train"),
    ("databricks/databricks-dolly-15k", "train"),     # human-written instruction->response (knowledge)
    ("HuggingFaceH4/ultrachat_200k", "train_sft"),    # topical/knowledge coverage
]


def detok(s):
    """Undo PTB-style tokenization (daily_dialog: 'beers after dinner ? ')."""
    s = re.sub(r"\s+([.,!?;:%])", r"\1", s)
    s = re.sub(r"\s+([’'])\s*", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def first_sentences(text, max_words=30):
    """The servable unit of a long turn: its leading sentence(s), whole, <= max_words."""
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    n = 0
    for p in parts:
        w = len(p.split())
        if n + w > max_words:
            break
        out.append(p); n += w
        if n >= 8:      # one or two sentences is the unit
            break
    return " ".join(out).strip()


def clean_reply(s):
    s = s.strip()
    w = s.split()
    if not (3 <= len(w) <= 30):
        return None
    low = s.lower()
    letters = sum(c.isalpha() or c in " '’,.!?-" for c in s)
    if letters / max(len(s), 1) < 0.9:
        return None
    if any(c in _CODE for c in s):
        return None
    if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.02:
        return None
    if any(p in low for p in _LEAK):
        return None
    if re.match(r"^[A-Za-z][a-z]*:", s) or re.match(r"^[a-z0-9][.)]\s", low):
        return None
    if re.search(r":\s*\d+\.?$", s):           # truncated enumeration ("...: 1.")
        return None
    if not s[-1] in ".!?…":
        return None
    return s


def good_prompt(s):
    w = s.split()
    return 2 <= len(w) <= 60 and any(c.isalpha() for c in s)


def turn_list(ex):
    """Ordered turns from an example, using explicit structure where present."""
    for key in ("messages", "conversation", "conversations"):
        v = ex.get(key)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            out = []
            for m in v:
                role = (m.get("role") or m.get("from") or "").lower()
                txt = str(m.get("content") or m.get("text") or m.get("value") or "")
                if txt.strip():
                    out.append((role, txt))
            return out
    # "User 1: ... / User 2: ..." markered conversation string (Synthetic-Persona-Chat)
    conv = ex.get("Best Generated Conversation")
    if isinstance(conv, str) and "User 1:" in conv:
        out = []
        for line in conv.splitlines():
            m = re.match(r"\s*User\s*\d+\s*:\s*(.+)$", line)
            if m and m.group(1).strip():
                out.append(("", m.group(1).strip()))
        return out
    for key in ("dialogue", "dialog", "turns", "utterances"):
        v = ex.get(key)
        if isinstance(v, list) and v and isinstance(v[0], str):
            return [("", detok(t)) for t in v if t.strip()]
    # instruction -> response (dolly-15k; role-correct by construction). Context-dependent
    # rows (non-empty "context") are skipped: their answers lean on an unseen passage.
    inst, resp = ex.get("instruction"), ex.get("response")
    if isinstance(inst, str) and isinstance(resp, str) and inst.strip() and resp.strip():
        if str(ex.get("context") or "").strip():
            return []
        return [("user", inst), ("assistant", resp)]
    return []


def fold_prompt(p):
    """Exact-FAQ normalization: casefold, strip punctuation, collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", p.lower())).strip()


def main():
    from datasets import load_dataset
    t0 = time.time()
    responses, prompts, plen, src_certain = [], [], [], []
    inv = defaultdict(list)
    exact = defaultdict(list)          # folded prompt -> [pid] (the exact-FAQ tier)
    tf_extra, seen = {}, set()

    def add_pair(p, r, certain):
        r2 = clean_reply(first_sentences(r))
        p2 = " ".join(p.split())[:400]
        if r2 is None or not good_prompt(p2):
            return 0
        h = hashlib.md5((p2.lower() + "\x00" + r2.lower()).encode()).digest()
        if h in seen:
            return 0
        seen.add(h)
        pcw = _content_words(tokenize(p2.lower()))
        fp = fold_prompt(p2)
        if not pcw and not fp:
            return 0
        pid = len(responses)
        responses.append(r2); prompts.append(p2); plen.append(max(len(pcw), 1))
        src_certain.append(1 if certain else 0)
        if fp and len(exact[fp]) < 20:
            exact[fp].append(pid)
        for w, c in Counter(pcw).items():
            inv[w].append(pid)
            if c > 1:
                tf_extra[(w, pid)] = c
        return 1

    for name, split in DATASETS:
        cap_ds = CAP_ULTRACHAT if "ultrachat" in name else CAP_PER_DS
        if len(responses) >= CAP_TOTAL:
            break
        try:
            ds = load_dataset(name, split=split, streaming=True)
        except Exception as e:
            print(f"  {name}: unavailable ({repr(e)[:60]}), skipping", flush=True)
            continue
        n_ds = 0
        try:
            for ex in ds:
                turns = turn_list(ex)
                for i in range(len(turns) - 1):
                    ra, ta = turns[i]
                    rb, tb = turns[i + 1]
                    # role-aware: only user->assistant when roles exist; else alternation.
                    # CERTAINTY: every current source is either role-keyed or a STRICT
                    # alternation dialogue, so each (turn_i -> turn_i+1) is a genuine
                    # exchange — certain=True for all. (The 1/2 prior remains in the
                    # ranker for future genuinely-unknown-structure sources; the v5 run
                    # measured that down-weighting alternation corpora REGRESSED e2e —
                    # it demoted the chit-chat source the good replies came from.)
                    has_roles = bool(ra and rb)
                    if has_roles and not (ra in ("user", "human", "prompter") and rb in ("assistant", "gpt", "bot")):
                        continue
                    n_ds += add_pair(ta, tb, certain=True)
                    if n_ds >= cap_ds or len(responses) >= CAP_TOTAL:
                        break
                if n_ds >= cap_ds or len(responses) >= CAP_TOTAL:
                    break
        except Exception as e:
            print(f"  {name}: stopped early ({repr(e)[:60]})", flush=True)
        print(f"  {name}: +{n_ds} pairs (total {len(responses)}) | {time.time()-t0:.0f}s", flush=True)

    inv2 = {w: lst[:1500] for w, lst in inv.items()}
    N = len(responses)
    avgdl = (sum(plen) / N) if N else 1.0
    with open(OUT, "wb") as fo:
        pickle.dump({"responses": responses, "prompts": prompts, "inv": inv2,
                     "tf_extra": tf_extra, "plen": plen, "avgdl": avgdl, "N": N,
                     "src_certain": src_certain, "exact": dict(exact)}, fo,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved TRUE-pair index: {N:,} pairs, {len(inv2):,} prompt-words, avgdl {avgdl:.1f} "
          f"-> {OUT} ({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
