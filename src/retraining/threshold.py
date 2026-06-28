"""
Adaptive drift threshold for retraining decisions.

The base threshold is fixed (env var DRIFT_THRESHOLD).  After each pipeline
run, the observed drift score is appended to a rolling history file.  If
recent scores are consistently high the threshold is lowered to make the
system more sensitive; if they are consistently low it is raised slightly.

Adaptive formula
----------------
    recent_mean   = mean of the last ``window`` recorded scores
    delta         = recent_mean - base_threshold
    adaptive      = base_threshold + sensitivity * delta
    clamped       = clamp(adaptive, base * 0.5, base * 2.0)

Setting ``sensitivity=0`` disables adaptation (pure static threshold).

Environment variables
---------------------
DRIFT_THRESHOLD          Base threshold (default: 0.3)
DRIFT_HISTORY_PATH       Path to rolling history JSON file
DRIFT_HISTORY_WINDOW     Number of past scores to consider (default: 30)
DRIFT_SENSITIVITY        Adaptation sensitivity 0–1 (default: 0.5)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from loguru import logger

BASE_THRESHOLD  = float(os.getenv("DRIFT_THRESHOLD",        "0.3"))
HISTORY_PATH    = os.getenv("DRIFT_HISTORY_PATH",   "data/state/drift_history.json")
HISTORY_WINDOW  = int(os.getenv("DRIFT_HISTORY_WINDOW",     "30"))
SENSITIVITY     = float(os.getenv("DRIFT_SENSITIVITY",       "0.5"))

_DT_FMT = "%Y-%m-%dT%H:%M:%SZ"


class ThresholdManager:
    def __init__(
        self,
        base_threshold: float = BASE_THRESHOLD,
        history_path:   str   = HISTORY_PATH,
        window:         int   = HISTORY_WINDOW,
        sensitivity:    float = SENSITIVITY,
    ):
        self.base      = base_threshold
        self.path      = Path(history_path)
        self.window    = window
        self.sensitivity = sensitivity

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read drift history: {exc}")
            return []

    def _save(self, history: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(history, f, indent=2)

    # ── public API ───────────────────────────────────────────────────────────

    def record_drift_score(
        self,
        score:     float,
        triggered: bool = False,
        dag_run_id: str | None = None,
    ) -> None:
        """Append a new score to the rolling history, trimming to ``window``."""
        history = self._load()
        history.append({
            "score":      score,
            "triggered":  triggered,
            "timestamp":  datetime.now(timezone.utc).strftime(_DT_FMT),
            "dag_run_id": dag_run_id,
        })
        # Keep only the most recent entries
        history = history[-self.window:]
        self._save(history)

    def get_recent_scores(self) -> list[float]:
        """Return raw score values from the rolling history."""
        return [e["score"] for e in self._load()]

    def get_adaptive_threshold(self) -> float:
        """
        Calculate the adaptive threshold from rolling history.
        Falls back to base when history is empty.
        """
        scores = self.get_recent_scores()
        if not scores or self.sensitivity == 0:
            return self.base

        recent_mean = mean(scores[-self.window:])
        delta       = recent_mean - self.base
        adaptive    = self.base + self.sensitivity * delta

        # Clamp: never go below half the base or above double the base
        low  = self.base * 0.5
        high = self.base * 2.0
        return max(low, min(high, adaptive))

    def should_retrain(self, current_score: float) -> tuple[bool, dict]:
        """
        Decide whether the current drift score warrants retraining.

        Returns
        -------
        (should_retrain, decision_info)
        decision_info keys: threshold_used, adaptive, base_threshold,
                            recent_mean, history_size
        """
        scores     = self.get_recent_scores()
        threshold  = self.get_adaptive_threshold()
        adaptive   = abs(threshold - self.base) > 1e-9
        recent_mean = mean(scores) if scores else None

        retrain = current_score > threshold

        info = {
            "threshold_used":  round(threshold, 4),
            "base_threshold":  self.base,
            "adaptive":        adaptive,
            "recent_mean":     round(recent_mean, 4) if recent_mean is not None else None,
            "history_size":    len(scores),
            "current_score":   round(current_score, 4),
        }
        return retrain, info
