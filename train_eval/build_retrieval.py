"""Build a sentence RETRIEVAL index over the conversational foundation, so generation can
pull whole COHERENT spans that lock to a schema (instead of composing word-by-word, which
fragments). Filters to clean conversational sentences (no code/noise). Saves an inverted
index content_word -> sentence-ids + the sentence list. Recombination at generation time
keeps it non-verbatim."""
import os, sys, re, pickle, time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omni.word_engine import tokenize, _content_words

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "conv_corpus.txt")
OUT = os.path.abspath(os.path.join(HERE, "..", "omni", "retrieval.pkl"))
CAP = int(sys.argv[2]) * 1_000_000 if len(sys.argv) > 2 else 60_000_000
MAX_SENTS = 900_000
_CODE = set("{}[]<>=/\\|`~^*_#@")
# essay / instructional / list register — the UltraChat noise we do NOT want as replies
_ESSAY = {"furthermore", "moreover", "however", "therefore", "consequently", "additionally",
          "firstly", "secondly", "thirdly", "fourthly", "conclusion", "overall", "various",
          "provides", "including", "utilize", "utilizing", "aforementioned", "respectively",
          # letter / formal-closing register (leaked as replies: "goodbye ... dear friend")
          "regards", "sincerely", "faithfully", "cordially", "salutations",
          # code / instructional register
          "function", "variable", "palindrome", "array", "integer", "string", "boolean",
          "parameter", "algorithm", "compiler", "syntax",
          # essay/tutorial framing
          "additionally", "notably", "specifically", "essentially", "subsequently"}
_ESSAY_START = ("step ", "note:", "example", "in this", "the following", "here are", "here's a list",
                "in conclusion", "in summary", "for instance", "to summarize", "the program",
                "the function", "the above", "as follows", "in the main", "once the", "bring this",
                "to whom", "dear ", "best regards", "yours ", "sincerely", "location:")
# instructional/code/letter/schedule phrases that must never be served as conversation
_LEAK_PHRASES = ("main function", "prompt the user", "variable named", "for instance",
                 "best regards", "sign off", "conclude by", "as follows", "click here",
                 "in this article", "in this tutorial", "dear friend", "dear colleague",
                 "to whom it may", "electronic source", "citing", "constructive criticism")
# conversational markers — first/second person, questions, casual address
_CONV = {"i", "you", "your", "we", "my", "me", "we're", "i'm", "you're", "let's", "that's",
         "i've", "you've", "we've", "i'll", "you'll", "don't", "it's", "yeah", "oh", "hey"}


def clean_sentence(s):
    s = s.strip()
    w = s.split()
    if not (4 <= len(w) <= 22):                    # short = conversational
        return None
    letters = sum(c.isalpha() or c == " " for c in s)
    if letters / max(len(s), 1) < 0.86:
        return None
    if any(c in _CODE for c in s):
        return None
    if sum(c.isdigit() for c in s) / max(len(s), 1) > 0.02:
        return None
    low = s.lower()
    # SAFETY/QUALITY: never index URLs, injection-style role resets, or spam — these could
    # otherwise be served verbatim inside a reply.
    if "http" in low or "www." in low or "@" in low or ".com" in low:
        return None
    if any(p in low for p in ("ignore previous", "ignore all previous", "disregard previous",
                              "click here", "you are now a", "system prompt", "jailbreak",
                              "as an ai language model", "dan mode")):
        return None
    if any(p in low for p in _LEAK_PHRASES):
        return None
    # list / schedule / contact register: "1. ...", "2:00pm", enumerations
    if re.search(r"\b\d{1,2}[:.]\d", s) or re.search(r"(^|\s)\d{1,2}\.\s", s):
        return None
    words = set(low.replace("'", "'").replace(",", " ").replace(".", " ").split())
    if words & _ESSAY:
        return None
    if low.startswith(_ESSAY_START) or low[0].isdigit():
        return None
    # KEEP only if conversational: has a first/second-person marker, or is a question/exclaim
    if not (words & _CONV) and not s.endswith(("?", "!")):
        return None
    return s


def main():
    t0 = time.time()
    text = open(CORPUS, encoding="utf-8", errors="replace").read(CAP)
    raw = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    sentences, inv = [], defaultdict(list)
    for s in raw:
        cs = clean_sentence(s)
        if cs is None:
            continue
        sid = len(sentences)
        sentences.append(cs)
        for cw in set(_content_words(tokenize(cs.lower()))):
            inv[cw].append(sid)
        if sid % 50000 == 0 and sid:
            print(f"  ...{sid} sentences, {len(inv)} index words | {time.time()-t0:.0f}s", flush=True)
        if sid >= MAX_SENTS:
            break
    # cap posting lists so common words don't bloat / dominate
    inv = {w: (lst[:800]) for w, lst in inv.items()}
    with open(OUT, "wb") as f:
        pickle.dump({"sentences": sentences, "inv": inv}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved retrieval index: {len(sentences):,} sentences, {len(inv):,} index words -> {OUT} "
          f"({os.path.getsize(OUT)/1e6:.0f}MB) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
