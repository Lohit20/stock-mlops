"""
Retraining cooldown guard.

Prevents the pipeline from triggering back-to-back retraining runs when
drift is detected on consecutive days.  State is persisted as a JSON file
so it survives container restarts and is visible to all pipeline tasks.

Environment variables
---------------------
RETRAIN_COOLDOWN_HOURS   Hold-off period after a retrain (default: 168 = 7 days)
RETRAIN_STATE_PATH       Path to the JSON state file
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "168"))
STATE_PATH     = os.getenv("RETRAIN_STATE_PATH", "data/state/retraining_state.json")

_DT_FMT = "%Y-%m-%dT%H:%M:%SZ"


class CooldownManager:
    def __init__(
        self,
        state_path: str = STATE_PATH,
        cooldown_hours: int = COOLDOWN_HOURS,
    ):
        self.state_path    = Path(state_path)
        self.cooldown_hours = cooldown_hours

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self.state_path.exists():
            return {}
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not read retrain state: {exc}")
            return {}

    def _save(self, state: dict) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    # ── public API ───────────────────────────────────────────────────────────

    def get_last_retrain_time(self) -> datetime | None:
        """Return the UTC timestamp of the last recorded retraining trigger."""
        raw = self._load().get("last_retrain_at")
        if raw is None:
            return None
        return datetime.strptime(raw, _DT_FMT).replace(tzinfo=timezone.utc)

    def is_cooldown_active(self) -> bool:
        """
        Return True if we are still within the hold-off window since the
        last retraining run.  Returns False when no prior run is recorded
        (first-ever retrain is always allowed).
        """
        last = self.get_last_retrain_time()
        if last is None:
            return False
        now     = datetime.now(timezone.utc)
        elapsed = (now - last).total_seconds() / 3600
        return elapsed < self.cooldown_hours

    def hours_since_last_retrain(self) -> float | None:
        """Hours elapsed since last retrain, or None if never retrained."""
        last = self.get_last_retrain_time()
        if last is None:
            return None
        now = datetime.now(timezone.utc)
        return (now - last).total_seconds() / 3600

    def get_status(self) -> dict:
        """
        Return a status dict suitable for XCom / logging.

        Keys
        ----
        is_active            bool
        cooldown_hours       int
        hours_since_last     float | None
        hours_remaining      float | None
        next_retrain_at      str (ISO-8601) | None
        last_retrain_at      str | None
        last_drift_score     float | None
        last_dag_run_id      str | None
        """
        state           = self._load()
        hours_since     = self.hours_since_last_retrain()
        active          = self.is_cooldown_active()
        hours_remaining = None
        next_at         = None

        if active and hours_since is not None:
            hours_remaining = self.cooldown_hours - hours_since
            last = self.get_last_retrain_time()
            if last:
                from datetime import timedelta
                next_dt = last + timedelta(hours=self.cooldown_hours)
                next_at = next_dt.strftime(_DT_FMT)

        return {
            "is_active":        active,
            "cooldown_hours":   self.cooldown_hours,
            "hours_since_last": round(hours_since, 2) if hours_since is not None else None,
            "hours_remaining":  round(hours_remaining, 2) if hours_remaining is not None else None,
            "next_retrain_at":  next_at,
            "last_retrain_at":  state.get("last_retrain_at"),
            "last_drift_score": state.get("last_drift_score"),
            "last_dag_run_id":  state.get("last_dag_run_id"),
        }

    def record_retrain_triggered(
        self,
        drift_score: float | None = None,
        dag_run_id:  str   | None = None,
    ) -> None:
        """
        Record that a retraining run was triggered right now.
        Call this *before* firing TriggerDagRunOperator so the cooldown
        takes effect even if Airflow crashes between tasks.
        """
        now   = datetime.now(timezone.utc)
        state = self._load()
        state.update({
            "last_retrain_at": now.strftime(_DT_FMT),
            "last_drift_score": drift_score,
            "last_dag_run_id":  dag_run_id,
            "retrain_count":    state.get("retrain_count", 0) + 1,
        })
        self._save(state)
        logger.info(
            f"Cooldown: recorded retrain at {now.strftime(_DT_FMT)} "
            f"(drift={drift_score}, run={dag_run_id})"
        )

    def reset(self) -> None:
        """Wipe state — useful for testing or manual override."""
        if self.state_path.exists():
            self.state_path.unlink()
        logger.info("Cooldown state reset")
