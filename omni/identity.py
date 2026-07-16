"""
UnisonIdentity — Token-level provenance and user fingerprinting.

Every token Unison produces is stamped with an HMAC-SHA256 fingerprint derived
from a persistent secret key, the session ID, turn index, and character index.
This allows Unison to verify whether it generated a given piece of text.

User fingerprints combine Discord metadata into a persistent per-user profile
so Unison never confuses who it's talking to across sessions.

SFT Compliance:
- Token IDs are metadata ABOUT the orbit, not part of it. They do not enter
  the SynapticGraph or affect prediction. The graph contains only exact chars.
- No floats, no zeros — IDs are hex strings derived from cryptographic hashing.
"""
import os
import json
import hmac
import hashlib
import time
import datetime
from omni.logging_config import get_logger

identity_logger = get_logger("OmniIdentity", "identity.log")

# ── Persistent paths ──────────────────────────────────────────────────────
_IDENTITY_DIR = os.path.join(os.path.dirname(__file__), "identity")
_KEY_PATH = os.path.join(_IDENTITY_DIR, "identity_key.bin")
_USERS_DIR = os.path.join(_IDENTITY_DIR, "users")


class UnisonIdentity:
    """
    Token provenance (paper Sec 8.12 support). Every character Unison generates
    can be traced back to the exact session, turn, and position where it was
    produced. Identity is METADATA only -- it never enters the graph, so it can
    never be confused with held content (the epistemic law: report internal state
    structurally -- what it holds, bound, closed -- carried in UNISON_PERSONA).
    """
    def __init__(self):
        os.makedirs(_IDENTITY_DIR, exist_ok=True)
        self.secret_key = self._load_or_create_key()
        identity_logger.info("UnisonIdentity initialised.")

    def _load_or_create_key(self):
        """Load persistent secret key, or generate one on first boot."""
        if os.path.exists(_KEY_PATH):
            with open(_KEY_PATH, "rb") as f:
                key = f.read()
            identity_logger.info("Identity key loaded from disk.")
            return key
        else:
            key = os.urandom(32)  # 256-bit key
            with open(_KEY_PATH, "wb") as f:
                f.write(key)
            identity_logger.info("Generated new identity key.")
            return key

    def stamp(self, session_id, turn_index, char_index, char):
        """
        Generate a provenance stamp for a single character.
        
        Returns first 8 hex chars of HMAC-SHA256(key, session_id|turn|charIdx|char).
        """
        message = f"{session_id}|{turn_index}|{char_index}|{char}".encode("utf-8")
        h = hmac.new(self.secret_key, message, hashlib.sha256)
        return h.hexdigest()[:8]

    def stamp_sequence(self, session_id, turn_index, chars):
        """
        Stamp every character in a sequence. Returns list of token_ids.
        """
        return [
            self.stamp(session_id, turn_index, i, c)
            for i, c in enumerate(chars)
        ]

    def verify(self, token_id, session_id, turn_index, char_index, char):
        """Verify a single token ID matches the expected provenance."""
        expected = self.stamp(session_id, turn_index, char_index, char)
        return hmac.compare_digest(token_id, expected)

    def verify_text(self, text, session_id, turn_index):
        """
        Verify a claimed Unison output against provenance stamps.
        
        Returns (verified: bool, match_ratio: float).
        """
        stamps = self.stamp_sequence(session_id, turn_index, list(text))
        # We can't verify without the claimed stamps — but we can RE-GENERATE
        # what the stamps SHOULD be. If the caller provides claimed stamps,
        # this method compares them. For now it returns the expected stamps
        # for the text, allowing external comparison.
        return stamps

    def generate_response_proof(self, session_id, turn_index, response_text):
        """
        Generate a compact proof-of-authorship for a full response.
        
        Returns a dict with the response hash, stamp chain root, and metadata.
        """
        stamps = self.stamp_sequence(session_id, turn_index, list(response_text))
        
        # Chain hash: hash of all individual stamps concatenated
        chain = "".join(stamps)
        chain_root = hashlib.sha256(chain.encode()).hexdigest()[:16]
        
        # Content hash
        content_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16]
        
        proof = {
            "session_id": session_id,
            "turn_index": turn_index,
            "char_count": len(response_text),
            "content_hash": content_hash,
            "chain_root": chain_root,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        
        identity_logger.debug(
            f"Proof generated: session={session_id} turn={turn_index} "
            f"chars={len(response_text)} chain={chain_root}"
        )
        return proof

    def verify_response_proof(self, proof, response_text):
        """
        Verify a response against a proof-of-authorship.
        
        Returns True if the text matches the proof.
        """
        # Re-generate stamps
        stamps = self.stamp_sequence(
            proof["session_id"], proof["turn_index"], list(response_text)
        )
        chain = "".join(stamps)
        chain_root = hashlib.sha256(chain.encode()).hexdigest()[:16]
        content_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()[:16]
        
        content_match = hmac.compare_digest(content_hash, proof["content_hash"])
        chain_match = hmac.compare_digest(chain_root, proof["chain_root"])
        
        return content_match and chain_match


class UserFingerprint:
    """
    Persistent per-user identity profile.
    
    Combines Discord metadata with interaction history so Unison
    always knows who it's talking to and never confuses users.
    """
    def __init__(self):
        os.makedirs(_USERS_DIR, exist_ok=True)
        self._cache = {}  # ukey -> profile dict
        self._load_all()

    def _load_all(self):
        """Load all user profiles from disk."""
        if not os.path.isdir(_USERS_DIR):
            return
        for fname in os.listdir(_USERS_DIR):
            if fname.endswith(".json"):
                try:
                    with open(os.path.join(_USERS_DIR, fname), "r") as f:
                        profile = json.load(f)
                    self._cache[profile["ukey"]] = profile
                except Exception:
                    pass
        identity_logger.info(f"Loaded {len(self._cache)} user profiles.")

    def get_or_create(self, ukey, display_name="unknown", discord_id=None):
        """Get existing profile or create a new one."""
        if ukey in self._cache:
            profile = self._cache[ukey]
            # Update display name if changed
            if display_name != "unknown":
                profile["display_name"] = display_name
            return profile

        profile = {
            "ukey": ukey,
            "discord_id": discord_id,
            "display_name": display_name,
            "first_seen": datetime.datetime.utcnow().isoformat() + "Z",
            "interaction_count": 0,
            "session_history": [],
            "preferences": {},
        }
        self._cache[ukey] = profile
        self._save(ukey)
        identity_logger.info(f"Created new user profile: {ukey} ({display_name})")
        return profile

    def record_interaction(self, ukey, session_id=None):
        """Record an interaction from this user."""
        if ukey not in self._cache:
            return
        profile = self._cache[ukey]
        profile["interaction_count"] += 1
        if session_id and session_id not in profile["session_history"]:
            profile["session_history"].append(session_id)
            # Keep last 100 session IDs
            if len(profile["session_history"]) > 100:
                profile["session_history"] = profile["session_history"][-100:]
        # Save every 10 interactions (debounced)
        if profile["interaction_count"] % 10 == 0:
            self._save(ukey)

    def _save(self, ukey):
        """Save a user profile to disk."""
        if ukey not in self._cache:
            return
        filepath = os.path.join(_USERS_DIR, f"{ukey}.json")
        try:
            with open(filepath, "w") as f:
                json.dump(self._cache[ukey], f, indent=2)
        except Exception:
            identity_logger.error(f"Failed to save profile for {ukey}", exc_info=True)

    def save_all(self):
        """Force save all profiles."""
        for ukey in self._cache:
            self._save(ukey)

    def summary(self, ukey):
        """Return a summary string for diagnostics."""
        if ukey not in self._cache:
            return f"Unknown user: {ukey}"
        p = self._cache[ukey]
        return (
            f"User: {p['display_name']} ({ukey}) | "
            f"Interactions: {p['interaction_count']} | "
            f"First seen: {p['first_seen']} | "
            f"Sessions: {len(p['session_history'])}"
        )
