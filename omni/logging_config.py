"""
Centralised logging configuration for the Unison Omni AI.
Each subsystem gets its own dedicated rotating log file.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

# One clearly-labeled logs directory at the project root: ./logs/ holds EVERYTHING.
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_FORMATTER = logging.Formatter(_FMT)

def _make_file_handler(filename, max_bytes=5_000_000, backup_count=3):
    path = os.path.join(LOG_DIR, filename)
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(_FORMATTER)
    return handler

def _make_console_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(_FORMATTER)
    return handler

def get_logger(name, filename):
    """
    Returns a logger that writes to BOTH the console and a dedicated log file.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_make_file_handler(filename))
        logger.addHandler(_make_console_handler())
    return logger

# ── MASTER SYSTEM LOG: EVERYTHING in one file, for live monitoring ──────────
# Every subsystem logger propagates to the root, so one root file handler captures
# the entire system — messages, generation, teaching, feedback, ratings, graduation,
# benchmarks — in one place. This is the default monitoring log.
_MASTER_PATH = os.path.join(LOG_DIR, "unison_system.log")
_root_logger = logging.getLogger()
if not any(getattr(h, "_is_master", False) for h in _root_logger.handlers):
    _root_logger.setLevel(logging.INFO)
    _master_handler = RotatingFileHandler(_MASTER_PATH, maxBytes=25_000_000, backupCount=5)
    _master_handler.setFormatter(_FORMATTER)
    _master_handler._is_master = True
    _root_logger.addHandler(_master_handler)

# ── Pre-built loggers for every subsystem ──────────────────────────────────
core_logger     = get_logger("OmniCore",     "core.log")
memory_logger   = get_logger("OmniMemory",   "memory.log")
teacher_logger  = get_logger("OmniTeacher",  "teacher.log")
discord_logger  = get_logger("OmniBot",      "discord.log")
# The full conversation flow — message in, generation out, ratings, teaching, feedback.
chat_logger     = get_logger("OmniChat",     "chat.log")

# ── Structured learning journal ────────────────────────────────────────────
import json
import datetime

_LEARNING_LOG_PATH = os.path.join(LOG_DIR, "learning.jsonl")

def log_learning_event(event_type, prompt, teacher_response, thought_extracted, answer_mapped, ukey="unknown"):
    """
    Appends a structured JSONL record for every single learning interaction.
    This is the permanent, human-readable audit trail of everything Unison has ever been taught.
    """
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "ukey": ukey,
        "prompt": prompt,
        "teacher_response_raw": teacher_response,
        "thought_extracted": thought_extracted,
        "answer_mapped": answer_mapped,
    }
    try:
        with open(_LEARNING_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        core_logger.error("Failed to write to learning.jsonl", exc_info=True)
