"""
SessionManager — Per-user conversational session tracking with working memory.

SFT Compliance:
- Working memory IS the live orbit. It is the exact character sequence of the
  current conversation session, maintained with \x02/\x03 speaker demarcation.
- When a session ends, the full session is banked as a single coherent orbit
  via hold_orbit(). This is fold closure — the conversation becomes permanent
  geometric structure in the graph.
- No lossy truncation. No sliding window. The exact sequence is preserved.
- CTX_MAX (6 = GEN_B × GEN_C) governs structural turn counting, not raw
  character length. The suffix search depth is bounded, not the context.
"""
import os
import json
import time
import uuid
import datetime
from omni.logging_config import get_logger

session_logger = get_logger("OmniSession", "session.log")

from omni.memory import SynapticGraph

class Session:
    """
    A single conversational session for one user (paper Sec 8.11: context and
    memory are ONE object at different ages). There is no context window: the
    working_context is simply the YOUNGEST memory -- the live orbit, the exact
    concatenation of this session's turns with \x02/\x03 demarcation. Binding is
    content-addressed over everything ever held; attention over the conversation
    HALVES with each step of age -- the fold factor 1/b = 1/2, not a tuned decay
    (constants/attention_in_the_product.ep Step 315, the cascade Step 313) -- and
    the current question always outvotes its past. On close the whole sequence is
    banked as one held orbit (fold closure).
    """
    def __init__(self, ukey, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.ukey = ukey
        self.start_time = time.time()
        self.turns = []                # List of (role, text) tuples
        self.working_context = []      # The live char list — the exact orbit
        self.turn_count = 0
        # APPEND-ONLY conversation record for the teacher's cross-turn context. Unlike
        # `turns` (which the correction / severing / feedback paths trim and which was
        # observed to reset to empty every turn in the live process — session.log 2026-07-15),
        # this list is written exactly once per completed turn and is NEVER trimmed, so the
        # teacher always sees the real conversation (e.g. a name the user gave earlier).
        self.history_log = []          # List of (role, text): finalized, shown exchanges
        self.episodic_memory = SynapticGraph(save_path=None)  # The Episodic Domain (0,1]

    def record_exchange(self, user_text, unison_text):
        """Record ONE finalized turn (the user message + the reply actually shown) into
        the append-only history_log. Called once at the end of a turn. Never trimmed."""
        self.history_log.append(("user", user_text))
        self.history_log.append(("unison", unison_text))
        
    def add_turn(self, role, text):
        """
        Append a conversational turn to this session.
        
        role: 'user' or 'unison'
        text: the raw message text
        
        User turns are wrapped with \x02...\x03 (start/end of user speech).
        Unison turns are appended directly (they follow the user's \x03).
        """
        self.turns.append((role, text))
        
        if role == "user":
            chars = ['\x02'] + list(text) + ['\x03']
        else:
            # Unison's response — the chars that follow the user's turn
            chars = list(text) + ['\x02']  # \x02 signals "back to user"
        
        self.working_context.extend(chars)
        self.turn_count += 1
        session_logger.debug(
            f"Session {self.session_id}: +{role} turn ({len(text)} chars), "
            f"working_context now {len(self.working_context)} chars, "
            f"{self.turn_count} turns total"
        )
        
    def get_full_sequence(self):
        """Return the complete session as a single character sequence for orbit banking."""
        return list(self.working_context)
    
    def summary(self):
        """Return a summary dict for logging/diagnostics."""
        elapsed = time.time() - self.start_time
        return {
            "session_id": self.session_id,
            "ukey": self.ukey,
            "turn_count": self.turn_count,
            "context_length": len(self.working_context),
            "duration_seconds": round(elapsed, 1),
            "turns": [(r, t[:50] + "..." if len(t) > 50 else t) for r, t in self.turns],
        }


class SessionManager:
    """
    Manages per-user conversational sessions.
    
    Each user has at most one active session. When a session ends (/new),
    the full conversation is banked as a coherent orbit and a new session starts.
    Session transcripts are archived to disk for analysis.
    """
    def __init__(self, archive_dir=None):
        self.active_sessions = {}  # ukey -> Session
        self.archive_dir = archive_dir or os.path.join(os.path.dirname(__file__), "sessions")
        os.makedirs(self.archive_dir, exist_ok=True)
        session_logger.info(f"SessionManager initialised. Archive dir: {self.archive_dir}")
    
    def get_or_create(self, ukey):
        """Get the active session for this user, or create one if none exists."""
        if ukey not in self.active_sessions:
            return self.start_session(ukey)
        return self.active_sessions[ukey]
    
    def start_session(self, ukey):
        """Create a fresh session for this user."""
        session = Session(ukey)
        self.active_sessions[ukey] = session
        session_logger.info(f"Started session {session.session_id} for ukey={ukey}")
        return session
    
    def end_session(self, ukey):
        """
        End the current session for this user.
        
        Returns the full session character sequence for orbit banking.
        Archives the session transcript to disk.
        Returns None if no active session.
        """
        if ukey not in self.active_sessions:
            return None
            
        session = self.active_sessions[ukey]
        full_sequence = session.get_full_sequence()
        
        # Archive the session transcript
        self._archive(session)
        
        # Remove from active sessions
        del self.active_sessions[ukey]
        
        session_logger.info(
            f"Ended session {session.session_id} for ukey={ukey} | "
            f"{session.turn_count} turns | {len(full_sequence)} chars"
        )
        
        return full_sequence
    
    def end_all(self):
        """End all active sessions. Returns list of (ukey, full_sequence) tuples."""
        results = []
        for ukey in list(self.active_sessions.keys()):
            seq = self.end_session(ukey)
            if seq:
                results.append((ukey, seq))
        return results
    
    def get_working_context(self, ukey):
        """Get the working context (live char list) for a user's current session."""
        session = self.get_or_create(ukey)
        return session.working_context
    
    def _archive(self, session):
        """Archive a completed session transcript to disk as JSON."""
        try:
            archive_data = {
                "session_id": session.session_id,
                "ukey": session.ukey,
                "start_time": datetime.datetime.fromtimestamp(session.start_time).isoformat(),
                "end_time": datetime.datetime.utcnow().isoformat(),
                "turn_count": session.turn_count,
                "context_length": len(session.working_context),
                "turns": [{"role": r, "text": t} for r, t in session.turns],
            }
            
            filename = f"{session.session_id}_{session.ukey}.json"
            filepath = os.path.join(self.archive_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(archive_data, f, indent=2)
            
            session_logger.info(f"Archived session {session.session_id} to {filepath}")
        except Exception as e:
            session_logger.error(f"Failed to archive session {session.session_id}", exc_info=True)
