"""
ObserverTeacher — Quality control supervisor for distillation.

Uses a lightweight model (separate from the main Teacher and the model
being distilled) to monitor the distillation pipeline for:
- Teacher stagnation (repetitive responses)
- Unison babble quality degradation
- Orbit pollution (dominant substring contamination)
- Rating agreement trends

The Observer runs periodically (every N iterations) and can intervene
by skipping seeds, switching models, or triggering Bad Ledger cleanup.
"""
import time
import json
import os
import requests
from collections import deque
from omni.logging_config import get_logger

observer_logger = get_logger("OmniObserver", "observer.log")

# ── Default observer model — lightweight, fast ──
_DEFAULT_OBSERVER_MODEL = "qwen3:8b"


class ObserverTeacher:
    """
    Lightweight supervisory agent that watches the distillation pipeline.
    """
    def __init__(self, model_name=None, check_interval=10, ollama_url="http://localhost:11434"):
        self.model_name = model_name or _DEFAULT_OBSERVER_MODEL
        self.ollama_url = ollama_url
        self.check_interval = check_interval
        
        # Rolling windows for trend detection
        self.recent_teacher_responses = deque(maxlen=50)
        self.recent_babble_depths = deque(maxlen=50)
        self.recent_ratings = deque(maxlen=50)  # (self_rating, teacher_rating)
        self.recent_agreements = deque(maxlen=50)
        
        # Stagnation tracking
        self.iteration_count = 0
        self.last_intervention = 0
        
        observer_logger.info(
            f"ObserverTeacher initialised: model={self.model_name}, "
            f"check_interval={self.check_interval}"
        )

    def record(self, teacher_response, babble_depth, self_rating, teacher_rating, agree):
        """Record metrics from a distillation iteration."""
        self.recent_teacher_responses.append(teacher_response[:200])
        self.recent_babble_depths.append(babble_depth)
        self.recent_ratings.append((self_rating, teacher_rating))
        self.recent_agreements.append(agree)
        self.iteration_count += 1

    def should_check(self):
        """Return True if it's time for an observer check."""
        return (self.iteration_count > 0 and 
                self.iteration_count % self.check_interval == 0)

    def check(self):
        """
        Run all quality checks. Returns a dict with findings and any interventions.
        """
        findings = {
            "iteration": self.iteration_count,
            "stagnation": self._check_stagnation(),
            "babble_trend": self._check_babble_trend(),
            "rating_trend": self._check_rating_trend(),
            "interventions": [],
        }
        
        # Decide interventions
        if findings["stagnation"]["is_stagnant"]:
            findings["interventions"].append("switch_model")
            observer_logger.warning(
                f"STAGNATION DETECTED at iteration {self.iteration_count}: "
                f"{findings['stagnation']['reason']}"
            )
        
        if findings["babble_trend"]["declining"]:
            findings["interventions"].append("cleanup_bad_ledger")
            observer_logger.warning(
                f"BABBLE QUALITY DECLINING at iteration {self.iteration_count}: "
                f"avg_depth={findings['babble_trend']['avg_depth']:.1f}"
            )
        
        if findings["rating_trend"]["consistently_bad"]:
            findings["interventions"].append("skip_seed")
            observer_logger.warning(
                f"CONSISTENTLY BAD RATINGS at iteration {self.iteration_count}: "
                f"bad_ratio={findings['rating_trend']['bad_ratio']:.2f}"
            )
        
        if findings["interventions"]:
            self.last_intervention = self.iteration_count
        
        observer_logger.info(
            f"Observer check at iteration {self.iteration_count}: "
            f"interventions={findings['interventions']}"
        )
        
        return findings

    def _check_stagnation(self):
        """Check if the Teacher is producing repetitive responses."""
        if len(self.recent_teacher_responses) < 10:
            return {"is_stagnant": False, "reason": "not enough data"}
        
        # Check overlap between recent responses
        recent = list(self.recent_teacher_responses)[-10:]
        overlap_count = 0
        
        for i in range(len(recent)):
            for j in range(i + 1, len(recent)):
                # Character-level overlap ratio
                a, b = recent[i], recent[j]
                if not a or not b:
                    continue
                common = sum(1 for c1, c2 in zip(a, b) if c1 == c2)
                ratio = common / max(len(a), len(b), 1)
                if ratio > 0.8:
                    overlap_count += 1
        
        total_pairs = len(recent) * (len(recent) - 1) // 2
        overlap_ratio = overlap_count / max(total_pairs, 1)
        
        is_stagnant = overlap_ratio > 0.3  # More than 30% of pairs are >80% similar
        
        return {
            "is_stagnant": is_stagnant,
            "overlap_ratio": overlap_ratio,
            "reason": f"overlap_ratio={overlap_ratio:.2f}" if is_stagnant else "ok",
        }

    def _check_babble_trend(self):
        """Check if Unison's suffix depth is declining (getting worse)."""
        if len(self.recent_babble_depths) < 10:
            return {"declining": False, "avg_depth": 0}
        
        depths = list(self.recent_babble_depths)
        first_half = depths[:len(depths) // 2]
        second_half = depths[len(depths) // 2:]
        
        avg_first = sum(first_half) / max(len(first_half), 1)
        avg_second = sum(second_half) / max(len(second_half), 1)
        avg_overall = sum(depths) / len(depths)
        
        # Declining if second half is significantly worse than first half
        declining = avg_second < avg_first * 0.7 and avg_second < 3
        
        return {
            "declining": declining,
            "avg_depth": avg_overall,
            "first_half_avg": avg_first,
            "second_half_avg": avg_second,
        }

    def _check_rating_trend(self):
        """Check if ratings are consistently bad."""
        if len(self.recent_ratings) < 10:
            return {"consistently_bad": False, "bad_ratio": 0}
        
        recent = list(self.recent_ratings)[-10:]
        bad_count = sum(1 for s, t in recent if s == "bad" and t == "bad")
        bad_ratio = bad_count / len(recent)
        
        return {
            "consistently_bad": bad_ratio > 0.7,
            "bad_ratio": bad_ratio,
        }

    def evaluate_response_quality(self, prompt, response):
        """
        Use the observer model to independently evaluate a response.
        Returns a quality assessment string.
        """
        try:
            eval_prompt = (
                f"Rate this AI response on a scale of 1-5 for naturalness and relevance. "
                f"Reply with just a number and one sentence.\n\n"
                f"User said: {prompt}\n"
                f"AI responded: {response[:500]}"
            )
            
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": eval_prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "temperature": 0.3},
                },
                timeout=30,
            )
            
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return "observer_error"
            
        except Exception as e:
            observer_logger.error(f"Observer evaluation failed: {e}")
            return "observer_error"
