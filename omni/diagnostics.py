"""
Diagnostics & Profiling for the Unison Omni AI.

Provides precise timing instrumentation across every critical path:
- Per-character prediction latency
- TTFT (Time To First Token)
- Total generation time & throughput (chars/sec)
- Suffix search depth & match quality
- Teacher model round-trip latency
- Memory persistence I/O latency
- Graph size & orbit statistics
"""
import time
import json
import os
import datetime
from omni.logging_config import get_logger

logger = get_logger("OmniDiag", "diagnostics.log")

from omni.logging_config import LOG_DIR as DIAG_LOG_DIR
os.makedirs(DIAG_LOG_DIR, exist_ok=True)
_DIAG_JSONL = os.path.join(DIAG_LOG_DIR, "diagnostics.jsonl")


class Timer:
    """Simple high-resolution timer context manager."""
    def __init__(self):
        self.start = None
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


class GenerationDiagnostics:
    """
    Collects per-generation diagnostics for a single response.
    Create one per on_message / auto_loop iteration.
    """
    def __init__(self, prompt_text="", ukey=""):
        self.prompt_text = prompt_text[:120]
        self.ukey = ukey
        self.start_time = time.perf_counter()

        # Timing buckets
        self.tokenization_ms = 0.0
        self.orbit_bank_ms = 0.0
        self.char_latencies_ms = []      # per-character predict_next latency
        self.suffix_depths = []          # k value per character (how deep the match was)
        self.candidate_counts = []       # how many candidates per character
        self.ttft_ms = None              # time to first token
        self.total_gen_ms = None         # total generation wall time
        self.post_gen_bank_ms = 0.0      # time to bank the generated orbit
        self.discord_send_ms = 0.0       # time to send to Discord

        # Result
        self.chars_generated = 0
        self.final_text = ""

    def record_char(self, latency_ms, suffix_depth, num_candidates):
        """Record timing for a single predicted character."""
        self.char_latencies_ms.append(latency_ms)
        self.suffix_depths.append(suffix_depth)
        self.candidate_counts.append(num_candidates)
        if self.ttft_ms is None:
            self.ttft_ms = (time.perf_counter() - self.start_time) * 1000.0

    def finish(self, chars_generated, final_text=""):
        """Finalise the diagnostics after generation completes."""
        self.total_gen_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.chars_generated = chars_generated
        self.final_text = final_text[:200]

    def summary_dict(self):
        """Return a full diagnostics dictionary."""
        char_lats = self.char_latencies_ms
        avg_char_ms = sum(char_lats) / len(char_lats) if char_lats else 0
        min_char_ms = min(char_lats) if char_lats else 0
        max_char_ms = max(char_lats) if char_lats else 0
        p50 = sorted(char_lats)[len(char_lats)//2] if char_lats else 0
        p95_idx = int(len(char_lats) * 0.95) if char_lats else 0
        p95 = sorted(char_lats)[p95_idx] if char_lats else 0

        avg_depth = sum(self.suffix_depths) / len(self.suffix_depths) if self.suffix_depths else 0
        max_depth = max(self.suffix_depths) if self.suffix_depths else 0
        avg_cands = sum(self.candidate_counts) / len(self.candidate_counts) if self.candidate_counts else 0

        chars_per_sec = (self.chars_generated / (self.total_gen_ms / 1000.0)) if self.total_gen_ms and self.total_gen_ms > 0 else 0

        return {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "ukey": self.ukey,
            "prompt": self.prompt_text,
            "chars_generated": self.chars_generated,
            "ttft_ms": round(self.ttft_ms, 2) if self.ttft_ms else None,
            "total_gen_ms": round(self.total_gen_ms, 2) if self.total_gen_ms else None,
            "chars_per_sec": round(chars_per_sec, 1),
            "tokenization_ms": round(self.tokenization_ms, 2),
            "orbit_bank_ms": round(self.orbit_bank_ms, 2),
            "post_gen_bank_ms": round(self.post_gen_bank_ms, 2),
            "discord_send_ms": round(self.discord_send_ms, 2),
            "per_char": {
                "avg_ms": round(avg_char_ms, 2),
                "min_ms": round(min_char_ms, 2),
                "max_ms": round(max_char_ms, 2),
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
            },
            "suffix_search": {
                "avg_depth": round(avg_depth, 1),
                "max_depth": max_depth,
                "avg_candidates": round(avg_cands, 1),
            },
            "output_preview": self.final_text,
        }

    def log(self):
        """Log the diagnostics summary to both the logger and the JSONL file."""
        d = self.summary_dict()

        logger.info(
            f"GEN | prompt=\"{d['prompt'][:60]}\" | "
            f"chars={d['chars_generated']} | "
            f"TTFT={d['ttft_ms']}ms | "
            f"total={d['total_gen_ms']}ms | "
            f"chars/s={d['chars_per_sec']} | "
            f"avg_char={d['per_char']['avg_ms']}ms | "
            f"p95_char={d['per_char']['p95_ms']}ms | "
            f"avg_depth={d['suffix_search']['avg_depth']} | "
            f"max_depth={d['suffix_search']['max_depth']}"
        )

        try:
            with open(_DIAG_JSONL, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception:
            logger.error("Failed to write diagnostics.jsonl", exc_info=True)

    def discord_summary(self):
        """Return a compact Discord-friendly diagnostics string."""
        d = self.summary_dict()
        return (
            f"```\n"
            f"── Diagnostics ──────────────────────\n"
            f"  Chars Generated : {d['chars_generated']}\n"
            f"  TTFT            : {d['ttft_ms']} ms\n"
            f"  Total Gen Time  : {d['total_gen_ms']} ms\n"
            f"  Throughput      : {d['chars_per_sec']} chars/sec\n"
            f"  Per-Char Avg    : {d['per_char']['avg_ms']} ms\n"
            f"  Per-Char P95    : {d['per_char']['p95_ms']} ms\n"
            f"  Per-Char Max    : {d['per_char']['max_ms']} ms\n"
            f"  Avg Suffix Depth: {d['suffix_search']['avg_depth']}\n"
            f"  Max Suffix Depth: {d['suffix_search']['max_depth']}\n"
            f"  Avg Candidates  : {d['suffix_search']['avg_candidates']}\n"
            f"  Orbit Bank      : {d['orbit_bank_ms']} ms\n"
            f"─────────────────────────────────────\n"
            f"```"
        )


class TeacherDiagnostics:
    """Collects per-query diagnostics for Teacher model calls."""
    def __init__(self, prompt_text=""):
        self.prompt_text = prompt_text[:120]
        self.start_time = time.perf_counter()
        self.ollama_ms = 0.0
        self.parse_ms = 0.0
        self.response_chars = 0
        self.thought_chars = 0
        self.answer_chars = 0

    def finish(self, response_chars=0, thought_chars=0, answer_chars=0):
        self.total_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.response_chars = response_chars
        self.thought_chars = thought_chars
        self.answer_chars = answer_chars

    def log(self):
        d = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "type": "teacher_query",
            "prompt": self.prompt_text,
            "ollama_roundtrip_ms": round(self.ollama_ms, 2),
            "parse_ms": round(self.parse_ms, 2),
            "total_ms": round(getattr(self, 'total_ms', 0), 2),
            "response_chars": self.response_chars,
            "thought_chars": self.thought_chars,
            "answer_chars": self.answer_chars,
        }
        logger.info(
            f"TEACHER | prompt=\"{d['prompt'][:60]}\" | "
            f"ollama={d['ollama_roundtrip_ms']}ms | "
            f"parse={d['parse_ms']}ms | "
            f"total={d['total_ms']}ms | "
            f"resp_chars={d['response_chars']} | "
            f"thought_chars={d['thought_chars']}"
        )
        try:
            with open(_DIAG_JSONL, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception:
            logger.error("Failed to write diagnostics.jsonl", exc_info=True)


class MemoryDiagnostics:
    """Tracks persistence I/O and graph statistics."""

    @staticmethod
    def log_save(elapsed_ms, orbit_count, total_chars):
        d = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "type": "memory_save",
            "elapsed_ms": round(elapsed_ms, 2),
            "orbit_count": orbit_count,
            "total_chars": total_chars,
        }
        logger.info(f"SAVE | {elapsed_ms:.1f}ms | orbits={orbit_count} | chars={total_chars}")
        try:
            with open(_DIAG_JSONL, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception:
            pass

    @staticmethod
    def log_load(elapsed_ms, orbit_count, total_chars):
        d = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "type": "memory_load",
            "elapsed_ms": round(elapsed_ms, 2),
            "orbit_count": orbit_count,
            "total_chars": total_chars,
        }
        logger.info(f"LOAD | {elapsed_ms:.1f}ms | orbits={orbit_count} | chars={total_chars}")
        try:
            with open(_DIAG_JSONL, "a") as f:
                f.write(json.dumps(d) + "\n")
        except Exception:
            pass

    @staticmethod
    def log_graph_stats(graph):
        """Snapshot of current graph size."""
        total_orbits = sum(len(v) for v in graph.orbits.values())
        total_chars = sum(sum(len(seq) for seq in v) for v in graph.orbits.values())
        ukeys = list(graph.orbits.keys())
        avg_orbit_len = total_chars / total_orbits if total_orbits > 0 else 0
        logger.info(
            f"GRAPH | orbits={total_orbits} | total_chars={total_chars} | "
            f"avg_orbit_len={avg_orbit_len:.0f} | ukeys={ukeys}"
        )
        return total_orbits, total_chars


# =============================================================================
# THE SELF-INSPECTION ORGAN  (paper Sec 8.12 / the persona's epistemic law)
# =============================================================================
def self_inspect(graph, context=None, ukey="public"):
    """The engine looking at ITSELF -- reporting, structurally, what it holds,
    what bound, and what it would select, over EXACT counts (not float weights).
    This is the counted analogue of foldprobe (which reads trained nets); the
    engine has no trained weights to probe -- its structure IS the held orbits.
    Grounds the persona's law: report internal state as what it holds / bound /
    closed / changed, never dramatised. Reads only; forces nothing.
    """
    from fractions import Fraction
    from omni.memory import exact_rational_shares
    from omni.core import GEN_B, GEN_C
    report = {
        # what it HOLDS
        "orbits_held": {k: len(v) for k, v in graph.orbits.items()},
        "total_chars": sum(len(seq) for v in graph.orbits.values() for seq in v),
    }
    if context:
        shares, suffix_len, total = exact_rational_shares(context, graph, ukey)
        ranked = sorted(shares.items(), key=lambda kv: kv[1].val, reverse=True)
        report["suffix_depth"] = suffix_len          # what BOUND: the longest held suffix
        report["total_paths"] = total
        # the loud head: the top b+c continuations carry the mass (Step 313/318)
        report["loud_head"] = [(c, str(fv.val)) for c, fv in ranked[:GEN_B + GEN_C]]
        # No-Zero: every share sits strictly in the domain (0, 1]
        report["all_shares_in_domain"] = all(0 < fv.val <= 1 for _, fv in ranked)
    return report
