"""Autonomous conversational-data scraper (the /scrape command).

Discovers high-quality CONVERSATIONAL datasets on the HuggingFace Hub (ranked by
downloads, plus a curated seed list), downloads the ones not yet scraped (tracked in a
manifest so each run adds FRESH data), extracts their turns, appends them to the
foundation corpus, and rebuilds the language stores (word_fluency + word_coupling).
This grows the conversational FOUNDATION — the language the engine speaks from. Orbits
(taught lessons / conversation) are never touched here.
"""
import os, json, subprocess, sys
from omni.logging_config import get_logger

logger = get_logger("OmniScraper", "scraper.log")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONV_CORPUS = os.path.join(ROOT, "train_eval", "conv_corpus.txt")
MANIFEST = os.path.join(ROOT, "train_eval", "scraped_manifest.json")
BUILD_FLUENCY = os.path.join(ROOT, "train_eval", "build_fluency.py")
BUILD_COUPLING = os.path.join(ROOT, "train_eval", "build_coupling.py")

# VETTED ALLOWLIST — only reputable, human-reviewed conversational datasets. Autonomous
# discovery of arbitrary Hub datasets is DISABLED: it pulled leaderboard-eval junk,
# wrong-language sets, off-domain code, and untrusted/promotional data — a poisoning /
# quality risk. The foundation grows only from this curated list.
SEED_DATASETS = [
    ("HuggingFaceH4/ultrachat_200k", "train_sft"),   # curated GPT SFT chat (HF-filtered)
    ("HuggingFaceH4/no_robots", "train"),            # 10k human-written, high quality
    ("OpenAssistant/oasst1", "train"),               # community assistant, reviewed
    ("OpenAssistant/oasst2", "train"),
    ("daily_dialog", "train"),                        # clean human daily dialogue
    ("allenai/soda", "train"),                        # social dialogue
    ("google/Synthetic-Persona-Chat", "train"),       # persona chit-chat
    ("blended_skill_talk", "train"),                  # blended casual dialogue
    ("databricks/databricks-dolly-15k", "train"),     # human-written instructions
]
_TRY_SPLITS = ["train_sft", "train", "train_gen", "sft", "validation"]


def _turns(ex):
    """Extract conversational turns from an arbitrary chat example."""
    for key in ("messages", "conversation", "conversations", "dialog", "turns", "chosen"):
        v = ex.get(key)
        if isinstance(v, list):
            out = []
            for m in v:
                if isinstance(m, dict):
                    out.append(str(m.get("content") or m.get("text") or m.get("value") or ""))
                elif isinstance(m, str):
                    out.append(m)
            if out:
                return [t for t in out if t]
    # instruction/response pairs
    inst = ex.get("instruction") or ex.get("prompt") or ex.get("question")
    resp = ex.get("response") or ex.get("output") or ex.get("answer") or ex.get("chosen")
    if isinstance(inst, str) and isinstance(resp, str):
        return [inst, resp]
    for key in ("text", "content"):
        if isinstance(ex.get(key), str) and ex[key].strip():
            return [ex[key]]
    return []


def discover(limit=25):
    """Return the VETTED allowlist only. Arbitrary Hub discovery is deliberately disabled
    (poisoning / quality risk) — the foundation grows only from human-reviewed datasets."""
    seen, out = set(), []
    for did, split in SEED_DATASETS:
        if did not in seen:
            seen.add(did); out.append((did, split))
    return out


def _load_manifest():
    if os.path.exists(MANIFEST):
        try:
            return json.load(open(MANIFEST))
        except Exception:
            pass
    return {"scraped": [], "bytes": 0}


def scrape_and_extend(cap_bytes=200_000_000, max_new=6, progress=None):
    """Download up to max_new NEW datasets (capped at cap_bytes total), append to the
    foundation corpus, and rebuild the language stores. Returns (added_ids, added_bytes)."""
    from datasets import load_dataset

    def say(m):
        logger.info(m)
        if progress:
            try: progress(m)
            except Exception: pass

    man = _load_manifest()
    scraped = set(man.get("scraped", []))
    candidates = [(d, s) for d, s in discover() if d not in scraped]
    say(f"discovered {len(candidates)} new candidate datasets; taking up to {max_new}")
    added_ids, added_bytes = [], 0

    os.makedirs(os.path.dirname(CONV_CORPUS), exist_ok=True)
    with open(CONV_CORPUS, "a", encoding="utf-8") as f:
        for did, split in candidates:
            if len(added_ids) >= max_new or added_bytes >= cap_bytes:
                break
            ds = None
            for sp in ([split] + _TRY_SPLITS):
                try:
                    ds = load_dataset(did, split=sp, streaming=True)
                    break
                except Exception:
                    ds = None
            if ds is None:
                say(f"  {did}: no usable split, skipping"); continue
            n = 0
            try:
                for ex in ds:
                    for t in _turns(ex):
                        f.write(t.strip()); f.write("\n"); n += len(t) + 1; added_bytes += len(t) + 1
                    f.write("\n")
                    if added_bytes >= cap_bytes:
                        break
                scraped.add(did); added_ids.append(did)
                say(f"  {did}: +{n//1_000_000}MB (total +{added_bytes//1_000_000}MB)")
            except Exception as e:
                logger.error(f"scrape {did} failed: {e}")
                say(f"  {did}: failed ({repr(e)[:60]})")

    man["scraped"] = sorted(scraped)
    man["bytes"] = man.get("bytes", 0) + added_bytes
    json.dump(man, open(MANIFEST, "w"), indent=2)

    if added_bytes > 0:
        say("rebuilding conversational fluency store from the augmented corpus…")
        subprocess.run([sys.executable, BUILD_FLUENCY, CONV_CORPUS], check=False)
        say("rebuilding conversational coupling graph…")
        subprocess.run([sys.executable, BUILD_COUPLING, CONV_CORPUS], check=False)
        say("foundation rebuilt.")
    return added_ids, added_bytes
