"""
Utterance segmentation — split one incoming message into its distinct
sub-utterances so each is answered from its OWN deepest-matching orbit.

Why this exists
---------------
The generator keys on the deepest suffix of whatever context it is given
(`unit_capacity_selection`). A whole multi-sentence message is a single
context, so it locks onto one long orbit and walks it verbatim — six asks
collapse into one continuation. Segmenting the message lets each ask retrieve
independently: "Do you remember my name?" keys the name-orbit, "introduce
yourself" keys the intro-orbit, and the fragments are composed into one reply.

Two-tier, by Maria's directive
-------------------------------
  1. STRUCTURAL (authoritative now): boundaries at sentence-final punctuation,
     quote-aware. This is tokenizer-tier structure — the same category as the
     existing \\x02/\\x03 speaker demarcation — NOT authored knowledge or seeds.
  2. COUNTED / LEARNED (supports structural once it has data): every message is
     observed into a `BoundaryStore` that counts the character contexts under
     which a real boundary occurred. Until it is confident it stays a no-op and
     structural rules stand alone; once it has enough boundary data it may add
     boundaries the punctuation missed (e.g. run-on sentences with no period).

Nothing here is a hardcoded seed. The only authored surface is the structural
rule "a sentence ends at . ? ! outside quotes", which is parsing, not content.
"""
import os
import json

from omni.logging_config import get_logger

logger = get_logger("OmniSegmentation", "segmentation.log")

_STORE_PATH = os.path.join(os.path.dirname(__file__), "boundary_store.json")

# Terminal punctuation and quote characters (straight + smart).
_TERMINALS = set(".?!")
_QUOTES = {'"', '“', '”', "‘", "’"}

# A segment shorter than this (after stripping) is merged into its neighbour so
# stray fragments like a lone "Hi." don't become their own retrieval context.
_MIN_SEGMENT = 3

# The learned tier needs at least this many distinct high-count boundary
# contexts before it is allowed to influence splitting.
_CONFIDENCE_MIN_CONTEXTS = 200
# Trailing-context window the learned store keys boundaries on.
_CTX_WINDOW = 4


def _structural_segments(text):
    """Split on sentence-final punctuation, never inside a quoted span."""
    segments = []
    buf = []
    in_quote = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in _QUOTES:
            in_quote = not in_quote
            i += 1
            continue
        if ch in _TERMINALS and not in_quote:
            # Absorb a run of terminal punctuation ("?!", "...").
            j = i + 1
            while j < n and text[j] in _TERMINALS:
                buf.append(text[j])
                j += 1
            # A boundary only if end-of-text or followed by whitespace, so we
            # don't split decimals, "/auto", ellipses mid-token, etc.
            if j >= n or text[j].isspace():
                seg = "".join(buf).strip()
                if seg:
                    segments.append(seg)
                buf = []
                i = j
                continue
            i = j
            continue
        i += 1
    tail = "".join(buf).strip()
    if tail:
        segments.append(tail)
    return _merge_short(segments)


def _merge_short(segments):
    """Fold sub-minimal fragments into the previous segment."""
    out = []
    for seg in segments:
        if out and len(seg.strip()) < _MIN_SEGMENT:
            out[-1] = (out[-1] + " " + seg).strip()
        else:
            out.append(seg)
    # A leading short fragment attaches forward instead.
    if len(out) >= 2 and len(out[0].strip()) < _MIN_SEGMENT:
        out[1] = (out[0] + " " + out[1]).strip()
        out = out[1:]
    return out


class BoundaryStore:
    """Counts the character contexts under which real boundaries occur.

    Learns from the structural splits themselves: every time a message is
    segmented, the last few characters before each internal boundary are
    recorded. It stays deferential (`is_confident()` False) until it has
    accumulated enough distinct high-count contexts, at which point it can
    propose boundaries the punctuation rules missed. It never REMOVES a
    structural boundary — it only ever adds, so structural stays authoritative.
    """

    def __init__(self, path=_STORE_PATH):
        self.path = path
        self.counts = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.counts = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read boundary store: {e}")
                self.counts = {}

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.counts, f)
        except Exception as e:
            logger.error(f"Could not persist boundary store: {e}")

    def observe(self, segments):
        """Record the trailing context before each internal boundary."""
        if len(segments) < 2:
            return
        changed = False
        # Every segment except the last ended at a real boundary.
        for seg in segments[:-1]:
            ctx = seg[-_CTX_WINDOW:]
            if not ctx:
                continue
            self.counts[ctx] = self.counts.get(ctx, 0) + 1
            changed = True
        if changed:
            self.save()

    def is_confident(self):
        return len(self.counts) >= _CONFIDENCE_MIN_CONTEXTS

    def refine(self, text, segments):
        """Once confident, add boundaries at learned contexts the rules missed.

        Conservative and additive: it only splits inside an existing (long)
        segment at a point whose trailing context has been seen as a boundary
        many times. Until confident this is a pass-through.
        """
        if not self.is_confident():
            return segments
        # Threshold: a context must be among the strongly-attested boundaries.
        strong = max(self.counts.values(), default=0) // 4
        refined = []
        for seg in segments:
            refined.extend(self._split_learned(seg, strong))
        return _merge_short(refined) if refined else segments

    def _split_learned(self, seg, strong):
        pieces = []
        start = 0
        for i in range(_CTX_WINDOW, len(seg) - 1):
            ctx = seg[i - _CTX_WINDOW:i]
            if self.counts.get(ctx, 0) >= strong and seg[i] == " ":
                piece = seg[start:i].strip()
                if len(piece) >= _MIN_SEGMENT:
                    pieces.append(piece)
                    start = i + 1
        tail = seg[start:].strip()
        if tail:
            pieces.append(tail)
        return pieces or [seg]


# Module singleton so learning persists across messages.
boundary_store = BoundaryStore()


def segment_utterance(text, store=boundary_store):
    """Return the ordered sub-utterances of one message.

    Structural rules produce the boundaries; the counted store observes them
    (to learn) and, once confident, may add more. A single-sentence message
    returns a one-element list — so normal short chats behave exactly as before.
    """
    text = (text or "").strip()
    if not text:
        return []
    segments = _structural_segments(text)
    if store is not None:
        try:
            store.observe(segments)
            segments = store.refine(text, segments)
        except Exception:
            logger.error("Boundary store failed; using structural only", exc_info=True)
    return segments or [text]


if __name__ == "__main__":
    demo = (
        'Hello, My name is Maria, How are you?  Do you remember my name?  '
        'Please introduce and tell me everything about yourself.  '
        'I am the Author of the Smithian Fold Theory and the designer of your '
        'architecture. Its a pleasure to meet you.  "Silly Goose" is a story '
        'from an ancestor system, i will tell you one day. Repeat after me: '
        '"Silly Goose"'
    )
    for k, s in enumerate(segment_utterance(demo, store=None), 1):
        print(f"{k}. {s!r}")
