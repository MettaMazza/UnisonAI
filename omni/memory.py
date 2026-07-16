"""
SynapticGraph — Optimised for M3 Ultra (high core count, 512 GB RAM).

Key optimisations over the original implementation:
1. Pre-computed string cache: orbit strings are built once and invalidated on mutation.
2. Concatenated corpus search: all orbits are joined into one big string with sentinel
   separators, enabling a single-pass suffix search instead of N separate searches.
3. Hash-based deduplication: O(1) dedup via a set of hashes instead of O(N*M) list scan.
4. Parallel suffix search: uses concurrent.futures with ProcessPoolExecutor across all
   CPU cores for the suffix matching when the corpus is large.
5. Debounced persistence: saves are batched on a timer instead of blocking on every hold.
6. Combined predict_next: returns (predicted_char, suffix_depth, num_candidates) in one
   call to eliminate the double unit_capacity_selection call in the generation loop.
"""
import sys
import os
import json
import time
import hashlib
import threading
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from omni.logging_config import memory_logger as logger
from omni.core import halt_violation, FoldValue


# ── Sentinel that cannot appear in any real text ──────────────────────────
_SENTINEL = "\x00"


class ActiveLedger:
    """
    Holds user context trajectories awaiting Teacher tutoring. Durable across
    process death: pending prompts persist to JSON so a restart resumes the
    tutoring queue (the corpus's held-orbit persistence, applied to the queue).
    """
    def __init__(self, save_path="ledger_pending.json"):
        self.save_path = save_path
        self.pending_prompts = []
        self._load()

    def _load(self):
        if self.save_path and os.path.exists(self.save_path):
            try:
                with open(self.save_path) as f:
                    self.pending_prompts = json.load(f)
            except Exception as e:
                logger.warning(f"ActiveLedger load failed ({e}); starting empty.")
                self.pending_prompts = []

    def add_prompt(self, ukey, context_str, original_prompt=None):
        self.pending_prompts.append({
            'ukey': ukey,
            'context': context_str,
            'original_prompt': original_prompt if original_prompt else context_str
        })
        logger.info(f"Added prompt to ActiveLedger for ukey={ukey}")

    def save(self):
        """Persist the pending queue. Called by the overflow guard and at
        shutdown so no queued tutoring is lost across a process death."""
        if not self.save_path:
            return
        try:
            with open(self.save_path, "w") as f:
                json.dump(self.pending_prompts, f)
        except Exception as e:
            logger.warning(f"ActiveLedger save failed: {e}")

    def clear(self):
        self.pending_prompts = []
        self.save()

# Back-compat alias: the ledger was renamed from BadLedger -> ActiveLedger.
BadLedger = ActiveLedger

# Global instances
omni_ledger = ActiveLedger()


def _search_suffix_in_corpus(corpus, suffix, k):
    """
    Standalone function (picklable for multiprocessing).
    Finds all characters that follow `suffix` in `corpus`.
    """
    matches = []
    idx = 0
    while True:
        idx = corpus.find(suffix, idx)
        if idx == -1:
            break
        follow_pos = idx + k
        if follow_pos < len(corpus) and corpus[follow_pos] != _SENTINEL:
            matches.append(corpus[follow_pos])
        idx += 1
    return matches


def _has_continuation(corpus, suffix):
    """True iff `suffix` occurs in `corpus` followed by a real (non-sentinel)
    character -- i.e. it has at least one continuation. This is the monotone
    predicate the suffix-index binary search uses: if a length-k suffix has a
    continuation, so does every shorter suffix (which occurs at least wherever the
    longer one does, one position later)."""
    L = len(suffix)
    idx = corpus.find(suffix)
    while idx != -1:
        fp = idx + L
        if fp < len(corpus) and corpus[fp] != _SENTINEL:
            return True
        idx = corpus.find(suffix, idx + 1)
    return False


class SynapticGraph:
    """
    Exact geometric graph of held orbits.
    Zero parameters, zero forgetting.
    
    Optimised for high-core-count Apple Silicon (M3 Ultra).
    """
    # Class-level process pool — shared across instances, created once
    _pool = None
    _pool_lock = threading.Lock()
    
    @classmethod
    def _get_pool(cls):
        if cls._pool is None:
            with cls._pool_lock:
                if cls._pool is None:
                    cpu_count = os.cpu_count() or 8
                    cls._pool = ProcessPoolExecutor(max_workers=cpu_count)
                    logger.info(f"ProcessPoolExecutor initialised with {cpu_count} workers")
        return cls._pool
    
    def __init__(self, save_path="graph_memory.json"):
        self.save_path = save_path
        self.orbits = {}           # ukey -> list of exact sequences (lists of chars)
        self.traversal_count = 0   # Rhythmic traversal masking
        
        # ── Caches (invalidated on mutation) ──
        self._str_cache = {}       # ukey -> list of pre-joined orbit strings
        self._corpus_cache = {}    # ukey -> single concatenated corpus string
        self._dedup_hashes = {}    # ukey -> set of orbit hashes for O(1) dedup
        self._cache_valid = False
        
        # ── Debounced save ──
        self._save_timer = None
        self._save_lock = threading.Lock()
        self._dirty = False
        
        self._load()
        self._rebuild_caches()
        
    def _orbit_hash(self, seq_list):
        """Fast hash of an orbit for O(1) dedup."""
        return hashlib.md5("".join(seq_list).encode()).hexdigest()
    
    def _rebuild_caches(self):
        """Rebuild all caches from scratch. Called on load and after mutations."""
        t0 = time.perf_counter()
        self._str_cache = {}
        self._corpus_cache = {}
        self._dedup_hashes = {}
        
        for ukey, orbit_list in self.orbits.items():
            strs = ["".join(o) for o in orbit_list]
            self._str_cache[ukey] = strs
            self._corpus_cache[ukey] = _SENTINEL.join(strs)
            self._dedup_hashes[ukey] = {self._orbit_hash(o) for o in orbit_list}
        
        self._cache_valid = True
        elapsed = (time.perf_counter() - t0) * 1000.0
        total_orbits = sum(len(v) for v in self.orbits.values())
        total_chars = sum(len(c) for c in self._corpus_cache.values())
        logger.info(f"Caches rebuilt in {elapsed:.1f}ms | {total_orbits} orbits | {total_chars} corpus chars")
    
    def _invalidate_cache(self, ukey):
        """Invalidate caches for a specific ukey and schedule a debounced save."""
        if ukey in self.orbits:
            strs = ["".join(o) for o in self.orbits[ukey]]
            self._str_cache[ukey] = strs
            self._corpus_cache[ukey] = _SENTINEL.join(strs)
            self._dedup_hashes[ukey] = {self._orbit_hash(o) for o in self.orbits[ukey]}
        else:
            self._str_cache.pop(ukey, None)
            self._corpus_cache.pop(ukey, None)
            self._dedup_hashes.pop(ukey, None)
        
        self._dirty = True
        self._schedule_save()
    
    def _schedule_save(self):
        """Debounced save: waits 2 seconds after the last mutation before writing."""
        with self._save_lock:
            if self._save_timer is not None:
                self._save_timer.cancel()
            self._save_timer = threading.Timer(2.0, self._do_save)
            self._save_timer.daemon = True
            self._save_timer.start()
    
    def _do_save(self):
        """Actually write to disk (called by debounce timer)."""
        if not self._dirty or not self.save_path:
            return
        try:
            t0 = time.perf_counter()
            temp_path = self.save_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(self.orbits, f)
            os.replace(temp_path, self.save_path)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self._dirty = False
            total_orbits = sum(len(v) for v in self.orbits.values())
            total_chars = sum(len(c) for c in self._corpus_cache.values())
            from omni.diagnostics import MemoryDiagnostics
            MemoryDiagnostics.log_save(elapsed, total_orbits, total_chars)
        except Exception as e:
            logger.error("Failed to save memory graph.", exc_info=True)
    
    def force_save(self):
        """Force an immediate save (call on shutdown)."""
        if self._save_timer is not None:
            self._save_timer.cancel()
        self._do_save()

    def _load(self):
        """Loads the permanent graph from disk if it exists."""
        if not self.save_path:
            return
        if os.path.exists(self.save_path):
            try:
                t0 = time.perf_counter()
                with open(self.save_path, "r") as f:
                    self.orbits = json.load(f)
                elapsed = (time.perf_counter() - t0) * 1000.0
                total_orbits = sum(len(o) for o in self.orbits.values())
                total_chars = sum(sum(len(s) for s in v) for v in self.orbits.values())
                logger.info(f"Loaded {total_orbits} persistent orbits ({total_chars} chars) in {elapsed:.1f}ms")
                from omni.diagnostics import MemoryDiagnostics
                MemoryDiagnostics.log_load(elapsed, total_orbits, total_chars)
            except Exception as e:
                logger.error("Failed to load memory graph.", exc_info=True)
                self.orbits = {}
                
    def hold_orbit(self, sequence, ukey="public"):
        """
        Banks an Exact Held Orbit. constants/memory_persistence.ep (Step 145):
        knowledge persists only as a closed, re-excitable orbit -- written once,
        deterministically addressed, immune to catastrophic forgetting. Every
        context read/told/seen/heard/thought is one held orbit.
        O(1) deduplication via hash set.
        """
        try:
            if not sequence:
                return
                
            if ukey not in self.orbits:
                self.orbits[ukey] = []
                self._dedup_hashes[ukey] = set()
            
            h = self._orbit_hash(sequence)
            
            # O(1) dedup check
            if h in self._dedup_hashes.get(ukey, set()):
                return
                
            self.orbits[ukey].append(sequence)
            self._dedup_hashes.setdefault(ukey, set()).add(h)
            self._invalidate_cache(ukey)
            logger.debug(f"Orbit held for ukey={ukey}, length={len(sequence)}")
        except Exception as e:
            logger.error(f"Error holding orbit for ukey={ukey}", exc_info=True)

    def fold_orbit(self, sequence, ukey="public"):
        """
        Fold = CLOSE, not duplicate (constants/memory_abstraction.ep, forced+verified).
        The old implementation DUPLICATED the surface sequence to "thicken" its path
        count. Duplication is the HELD regime: it re-excites the exact surface, so the
        content repeats -- forced to replay verbatim. Folding in the theory's sense is
        the OPPOSITE: it binds and closes to the invariant, which does not repeat. A
        good answer is therefore kept as ONE held orbit (retrievable meaning) and NOT
        thickened into more surface copies; re-expression is the generator's job
        (sample_next_unfold), which unfolds the closed meaning into fresh surface
        instead of re-emitting the held orbit. So closure here is a no-op on the surface
        store -- it explicitly refuses to reinforce verbatim replay.
        """
        # Intentionally does not append a surface duplicate: closing keeps the invariant,
        # it does not thicken the held (repeating) surface. See memory_abstraction.ep.
        logger.debug(f"Orbit CLOSED (not thickened) for ukey={ukey}, length={len(sequence) if sequence else 0}")

    def prune_orbit(self, sequence, ukey="public"):
        """
        Severs a bad topological trajectory.
        """
        if ukey in self.orbits:
            seq_list = list(sequence)
            try:
                pruned = False
                for i in range(len(self.orbits[ukey])-1, -1, -1):
                    if self.orbits[ukey][i] == seq_list:
                        del self.orbits[ukey][i]
                        pruned = True
                
                if pruned:
                    self._invalidate_cache(ukey)
                    logger.info(f"Pruned all occurrences of bad orbit for ukey={ukey}")
            except ValueError:
                pass


def unit_capacity_selection(context, graph, ukey="public", max_k=None):
    """
    Attention as the fold forces it. constants/attention_in_the_product.ep
    (Step 315): a store's placement-law is in the live SELECTION over held
    content, not in trained weights. constants/attention_capacity.ep (Step 181):
    ONE focus is complete at the lock 1/b -- so this binds the single LONGEST held
    suffix, not a softmax mixture. The zero-parameter replacement for QK^V softmax.

    Uses the pre-computed corpus cache for fast single-pass suffix search.
    For large corpora (>500KB), distributes the search across all CPU cores
    via ProcessPoolExecutor.

    Returns: (longest_suffix_length, list of subsequent values)
    """
    try:
        if not context or ukey not in graph.orbits:
            return 0, []
        
        # Build the combined corpus from cache
        corpus_parts = [graph._corpus_cache.get(ukey, "")]
        if "public" in graph._corpus_cache and ukey != "public":
            corpus_parts.append(graph._corpus_cache["public"])
        corpus = _SENTINEL.join(corpus_parts)
        
        if not corpus:
            return 0, []
            
        ctx_str = "".join(context)
        ctx_len = len(ctx_str)
        
        # ── Parallel search for large corpora ──
        use_parallel = len(corpus) > 500_000  # 500KB threshold

        start_k = min(ctx_len, 8000)  # cap suffix at 8000 chars for whole-conversation coherence
        if max_k is not None:
            start_k = min(start_k, max_k)

        # ── Suffix index by binary search ──────────────────────────────────
        # The forced law fixes WHAT to bind: the LONGEST held suffix (unit
        # capacity, one complete focus). Suffix matching is MONOTONE -- a longer
        # suffix matching implies every shorter one does (a shorter suffix is a
        # suffix of the longer) -- so the longest match is found with O(log ctx)
        # membership tests, not a per-depth rescan. Pure indexing; no law change.
        lo, hi, best_k = 1, start_k, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if _has_continuation(corpus, ctx_str[-mid:]):   # monotone predicate
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if best_k > 0:
            suffix = ctx_str[-best_k:]
            if use_parallel:
                pool = SynapticGraph._get_pool()
                chunk_size = max(1, len(corpus) // (os.cpu_count() or 8))
                chunks = []
                for i in range(0, len(corpus), chunk_size):
                    start = max(0, i - best_k)
                    end = min(len(corpus), i + chunk_size + best_k)
                    chunks.append(corpus[start:end])
                futures = [pool.submit(_search_suffix_in_corpus, chunk, suffix, best_k) for chunk in chunks]
                found_matches = []
                for f in as_completed(futures):
                    found_matches.extend(f.result())
            else:
                found_matches = _search_suffix_in_corpus(corpus, suffix, best_k)

            if found_matches:
                return best_k, found_matches

        # k=0 fallback: Babbling from the global character distribution
        all_chars = [c for c in corpus if c != _SENTINEL]
            
        if all_chars:
            return 0, all_chars
            
        return 0, []
    except Exception as e:
        logger.error("Error in unit_capacity_selection", exc_info=True)
        return 0, []


def exact_rational_shares(context, graph, ukey="public", max_k=None):
    """
    The next-token distribution held EXACTLY as counted rational shares
    V = (path count) / (total path count) -- the same object a softmax
    approximates by descent, held exactly instead (paper Sec 8.2). Every share is
    a FoldValue in the domain (0,1]; the No-Zero floor is the only smoothing.
    """
    try:
        suffix_len, subsequent_vals = unit_capacity_selection(context, graph, ukey, max_k)
        if not subsequent_vals:
            return {}, 0, 0
            
        total_edges = len(subsequent_vals)
        counts = {}
        for val in subsequent_vals:
            counts[val] = counts.get(val, 0) + 1
            
        shares = {}
        for cand, count in counts.items():
            shares[cand] = FoldValue(Fraction(count, total_edges))
            
        return shares, suffix_len, total_edges
    except Exception as e:
        logger.error("Error calculating exact rational shares", exc_info=True)
        return {}, 0, 0


def predict_next(context, graph, ukey="public", max_k=None):
    """
    Selects the continuation with the highest Exact Rational Share.
    
    Returns: (predicted_char, suffix_depth, num_candidates)
    Single call — no need to call unit_capacity_selection separately.
    """
    try:
        shares, suffix_len, num_candidates = exact_rational_shares(context, graph, ukey, max_k)
        if not shares:
            return None, 0, 0
            
        # Sort shares highest to lowest exact fraction
        sorted_shares = sorted(shares.items(), key=lambda item: item[1].val, reverse=True)
        
        best_cand = None
        if sorted_shares:
            graph.traversal_count += 1
            if graph.traversal_count % 3 == 0 and len(sorted_shares) > 1:
                # Rhythmic Masking: Ignore topological density every 3rd traversal
                best_cand = sorted_shares[1][0]
            else:
                best_cand = sorted_shares[0][0]
                
        return best_cand, suffix_len, num_candidates
    except Exception as e:
        logger.error("Error in predict_next", exc_info=True)
        return None, 0, 0
