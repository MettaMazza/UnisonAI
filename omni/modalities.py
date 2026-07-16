"""
Multimodal Encoders — the FOLD EYE and FOLD EAR.

Every modality enters as its exact integer mathematics and is read in the
integer Walsh family (the binary generator's harmonics). The forced laws:
- constants/two_family_harmonics.ep (Step 308): the eye/ear read in the INTEGER
  Walsh family (whole-valued coefficients) -- the fold's own basis.
- constants/functional_band.ep (Step 311): a field's function is carried by its
  top BAND = b^(b+c) = 32 coefficients; the sight/sound is those coefficients.
- constants/family_signature.ep (Step 316): the percept is the VALUE the law
  sits at, not its loudness -- so the token IS the (position, integer value) pair.

Self-certifying per act: the integer Parseval identity sum(x^2)*N == sum(X^2)
holds over the integers or the percept is discarded (fwht_2d certifies it). The
DC coefficient (overall brightness/level) is held separately -- the sight tokens
are STRUCTURE (non-DC), so a checkerboard (a single Walsh function) yields
exactly ONE sight token. Recognition runs before any model: two percepts match
iff their top-BAND signatures match.

Sentinels:  \\x06 IMAGE_START  \\x07 IMAGE_END   \\x0E AUDIO_START  \\x0F AUDIO_END
"""
import io
from omni.core import GEN_C, BAND, fwht_1d, fwht_2d, halt_violation
from omni.logging_config import get_logger

modality_logger = get_logger("OmniModality", "modality.log")

IMAGE_START = '\x06'
IMAGE_END = '\x07'
AUDIO_START = '\x0E'
AUDIO_END = '\x0F'

# GEN_C intensity levels — used ONLY for the coarse teacher-facing description,
# never for the engine's native (Walsh) representation.
_INTENSITY_CHARS = ['░', '▒', '▓']


def _is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def _top_band_tokens(coeffs_flat):
    """The forced sight/sound: the top-BAND NONZERO coefficients (No-Zero: a zero
    coefficient is not a token), loudest first then by position. DC (position 0)
    is EXCLUDED -- it is the level, not the structure. Returns [(position, value)]."""
    nz = [(abs(v), pos, v) for pos, v in enumerate(coeffs_flat) if pos != 0 and v != 0]
    nz.sort(key=lambda t: (-t[0], t[1]))
    return [(pos, v) for _, pos, v in nz[:BAND]]


class ModalityEncoder:
    def encode(self, data):
        raise NotImplementedError
    def decode(self, chars):
        raise NotImplementedError


class TextEncoder(ModalityEncoder):
    """Identity encoder — text is already characters."""
    def encode(self, text):
        return list(text)
    def decode(self, chars):
        return "".join(chars)


class ImageEncoder(ModalityEncoder):
    """
    THE FOLD EYE (paper Sec 8.5). An image enters as its exact mathematics:
    grayscale integer field -> a GRID x GRID integer grid -> 2D Walsh-Hadamard in
    pure integer arithmetic (Parseval-certified or discarded) -> the top-BAND
    non-DC coefficients as sight tokens. Recognition is counted-spectrum match,
    before any model. Grid size (GRID) is engineering (perceptual resolution); the
    BAND=32 token count is forced (Step 311).
    """
    GRID = 64  # 2^6 — power of two for the Walsh transform

    def __init__(self, grid_size=None):
        self.grid_size = grid_size or self.GRID
        if not _is_power_of_two(self.grid_size):
            halt_violation("fold eye grid must be a power of two")
        self.total_pixels = self.grid_size * self.grid_size

    # --- the two entry points: raw exact field, or image bytes -----------------
    def _spectrum(self, field):
        """field: GRID x GRID list of ints. Returns (dc, [(pos, val)] sight tokens)."""
        g = self.grid_size
        if len(field) != g or any(len(r) != g for r in field):
            halt_violation(f"fold eye field must be {g}x{g}")
        coeffs = fwht_2d(field)                    # self-certifies Parseval or halts
        flat = [coeffs[r][c] for r in range(g) for c in range(g)]
        return flat[0], _top_band_tokens(flat)

    def encode_field(self, field):
        """Encode an exact integer GRID x GRID field (used by the verify harness)."""
        try:
            dc, tokens = self._spectrum(field)
        except SystemExit:
            raise
        except Exception as e:
            modality_logger.error(f"fold eye discarded a sight: {e}")
            return [IMAGE_START, IMAGE_END]
        body = f"DC={dc}|" + ";".join(f"{p}:{v}" for p, v in tokens)
        return [IMAGE_START] + list(body) + [IMAGE_END]

    def _field_from_bytes(self, image_bytes):
        from PIL import Image
        g = self.grid_size
        img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((g, g), Image.LANCZOS)
        px = list(img.getdata())
        return [[int(px[r * g + c]) for c in range(g)] for r in range(g)]

    def encode(self, image_bytes):
        try:
            field = self._field_from_bytes(image_bytes)
        except ImportError:
            modality_logger.error("Pillow required for image bytes; install Pillow.")
            return [IMAGE_START, IMAGE_END]
        except Exception as e:
            modality_logger.error(f"image decode failed: {e}")
            return [IMAGE_START, IMAGE_END]
        return self.encode_field(field)

    # --- recognition (before any model) ---------------------------------------
    def signature(self, chars):
        """The recognition key: the top-BAND (position, value) pairs, canonical."""
        pairs = self.decode(chars)
        return tuple(pairs)

    def decode(self, chars):
        body = "".join(c for c in chars if c not in (IMAGE_START, IMAGE_END))
        if "|" in body:
            body = body.split("|", 1)[1]
        pairs = []
        for tok in body.split(";"):
            if ":" in tok:
                p, v = tok.split(":", 1)
                try:
                    pairs.append((int(p), int(v)))
                except ValueError:
                    pass
        return pairs

    def describe_for_teacher(self, chars):
        """A coarse, human/teacher-readable summary of the sight (the teacher
        closes the percept with words; the engine holds the Walsh tokens)."""
        pairs = self.decode(chars)
        if not pairs:
            return "[Empty image]"
        body = "".join(c for c in chars if c not in (IMAGE_START, IMAGE_END))
        dc = body.split("|", 1)[0].replace("DC=", "") if "|" in body else "?"
        return (f"[Fold-eye sight: {len(pairs)} Walsh coefficient(s) of the top-{BAND}, "
                f"level(DC)={dc}, dominant positions {[p for p, _ in pairs[:5]]}]")


class AudioEncoder(ModalityEncoder):
    """
    THE FOLD EAR (paper Sec 8.6). Audio enters as its exact integer mathematics:
    integer samples -> power-of-two windows -> 1D Walsh-Hadamard per window (pure
    integer, Parseval-certified) -> the top-BAND non-DC coefficients per window as
    sound tokens. Heard once, a sound is thereafter recognized from the engine's
    own spectrum with no transcriber. Window size is engineering; BAND forced.
    """
    def __init__(self, window=1024):
        if not _is_power_of_two(window):
            halt_violation("fold ear window must be a power of two")
        self.window = window

    def _window_tokens(self, samples):
        """samples: list of window ints. Returns (dc, [(pos, val)])."""
        if len(samples) != self.window:
            halt_violation(f"fold ear window must be {self.window} samples")
        coeffs = fwht_1d(samples)                  # exact integer
        # integer Parseval per hearing (1D): sum(x^2)*N == sum(X^2)
        n = self.window
        if sum(int(x) ** 2 for x in samples) * n != sum(int(X) ** 2 for X in coeffs):
            halt_violation("fold ear Parseval failed -- hearing discarded")
        return coeffs[0], _top_band_tokens(coeffs)

    def encode_samples(self, samples):
        """Encode a list of integer samples (length a multiple of the window)."""
        out = [AUDIO_START]
        n = self.window
        nwin = len(samples) // n
        if nwin == 0:
            return [AUDIO_START, AUDIO_END]
        for w in range(nwin):
            try:
                dc, tokens = self._window_tokens(samples[w * n:(w + 1) * n])
            except SystemExit:
                raise
            except Exception as e:
                modality_logger.error(f"fold ear discarded a window: {e}")
                continue
            out += list(f"DC={dc}|" + ";".join(f"{p}:{v}" for p, v in tokens))
            out.append('|')
        out.append(AUDIO_END)
        return out

    def encode(self, audio_bytes):
        """Decode audio bytes (wav) to integer samples, then encode. Falls back to
        a clean empty percept if no decoder is available (never a fake sound)."""
        try:
            import wave
            with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                sw = wf.getsampwidth()
            import struct
            fmt = {1: 'b', 2: 'h', 4: 'i'}.get(sw)
            if not fmt:
                return [AUDIO_START, AUDIO_END]
            samples = list(struct.unpack(f"<{len(frames)//sw}{fmt}", frames))
            # shift signed samples into a positive integer domain (No-Zero) by +offset
            off = 1 - min(samples) if samples else 1
            samples = [s + off for s in samples]
            # trim to a whole number of windows
            samples = samples[: (len(samples) // self.window) * self.window]
            return self.encode_samples(samples)
        except Exception as e:
            modality_logger.warning(f"audio decode failed ({e}); empty percept.")
            return [AUDIO_START, AUDIO_END]

    def decode(self, chars):
        return None


def extract_modality_segments(char_sequence):
    """Split a mixed sequence into ('text'|'image'|'audio', chars) segments."""
    segments = []
    current_text = []
    i = 0
    while i < len(char_sequence):
        c = char_sequence[i]
        if c in (IMAGE_START, AUDIO_START):
            end = IMAGE_END if c == IMAGE_START else AUDIO_END
            kind = 'image' if c == IMAGE_START else 'audio'
            if current_text:
                segments.append(('text', current_text)); current_text = []
            chunk = [c]; i += 1
            while i < len(char_sequence) and char_sequence[i] != end:
                chunk.append(char_sequence[i]); i += 1
            if i < len(char_sequence):
                chunk.append(char_sequence[i]); i += 1
            segments.append((kind, chunk))
        else:
            current_text.append(c); i += 1
    if current_text:
        segments.append(('text', current_text))
    return segments


if __name__ == "__main__":
    # The forced checkerboard property: a GRID x GRID checkerboard is a single
    # Walsh function, so it yields exactly ONE sight token.
    eye = ImageEncoder()
    g = eye.grid_size
    checker = [[0 if (r + c) % 2 == 0 else 255 for c in range(g)] for r in range(g)]
    sight = eye.encode(checker) if False else eye.encode_field(checker)
    tokens = eye.decode(sight)
    print(f"fold eye: checkerboard -> {len(tokens)} sight token(s) (forced: 1)")
    assert len(tokens) == 1, f"checkerboard should yield ONE token, got {len(tokens)}"
    # recognition: the same field re-recognized
    assert eye.signature(eye.encode_field(checker)) == eye.signature(sight)
    print("fold eye: recognition by counted spectrum OK (before any model)")
