"""
Retraining decision scheduler.

Combines the cooldown guard and adaptive threshold into a single
``evaluate_retraining_need`` call that the Airflow task uses.

Decision logic (in order)
--------------------------
1. If cooldown is active → skip retraining (return should_retrain=False).
2. Record the drift score in the rolling threshold history.
3. Ask the adaptive threshold whether the score is high enough.
4. If yes AND cooldown is not active → approve retraining.

The result dict is pushed to XCom by ``task_check_drift`` so downstream
tasks (branch, report, monitoring push) can access the full decision.
"""

import os
from loguru import logger

from src.retraining.cooldown   import CooldownManager, COOLDOWN_HOURS, STATE_PATH
from src.retraining.threshold  import ThresholdManager, BASE_THRESHOLD, HISTORY_PATH


def evaluate_retraining_need(
    drift_result:   dict,
    cooldown_hours: int   = COOLDOWN_HOURS,
    state_path:     str   = STATE_PATH,
    history_path:   str   = HISTORY_PATH,
    dag_run_id:     str | None = None,
) -> dict:
    """
    Evaluate whether the pipeline should trigger a retraining run.

    Parameters
    ----------
    drift_result:
        Dict returned by ``run_data_drift_report`` — must contain
        ``drift_score``, ``drift_detected``, ``needs_retraining``.
    cooldown_hours:
        Hold-off period between retraining runs.
    state_path:
        Cooldown state file path.
    history_path:
        Drift score history file path.
    dag_run_id:
        Airflow dag_run_id for traceability (optional).

    Returns
    -------
    dict with keys:
        should_retrain      bool
        reason              str — human-readable explanation
        drift_score         float
        drift_detected      bool
        threshold_used      float
        base_threshold      float
        adaptive            bool
        recent_mean         float | None
        history_size        int
        cooldown_active     bool
        cooldown_status     dict (from CooldownManager.get_status)
    """
    score    = float(drift_result.get("drift_score",    0.0) or 0.0)
    detected = bool(drift_result.get("drift_detected",  False))

    cooldown  = CooldownManager(state_path=state_path, cooldown_hours=cooldown_hours)
    threshold = ThresholdManager(history_path=history_path)

    # ── Step 1: cooldown guard ────────────────────────────────────────────────
    cooldown_status = cooldown.get_status()
    if cooldown_status["is_active"]:
        hrs = cooldown_status.get("hours_remaining", 0) or 0
        reason = (
            f"Cooldown active — {hrs:.1f} h remaining "
            f"(last retrain: {cooldown_status.get('last_retrain_at')})"
        )
        logger.info(f"Retraining skipped: {reason}")

        # Still record the score for threshold adaptation
        threshold.record_drift_score(score, triggered=False, dag_run_id=dag_run_id)
        _, threshold_info = threshold.should_retrain(score)

        return {
            "should_retrain":   False,
            "reason":           reason,
            "drift_score":      score,
            "drift_detected":   detected,
            "cooldown_active":  True,
            "cooldown_status":  cooldown_status,
            **threshold_info,
        }

    # ── Step 2: record score and get threshold decision ───────────────────────
    threshold.record_drift_score(score, triggered=False, dag_run_id=dag_run_id)
    should, threshold_info = threshold.should_retrain(score)

    # ── Step 3: build result ──────────────────────────────────────────────────
    if should:
        reason = (
            f"Drift score {score:.3f} exceeds threshold "
            f"{threshold_info['threshold_used']:.3f}"
            + (" (adaptive)" if threshold_info["adaptive"] else "")
        )
    else:
        reason = (
            f"Drift score {score:.3f} below threshold "
            f"{threshold_info['threshold_used']:.3f} — no retraining needed"
        )

    logger.info(
        f"Retraining decision: should_retrain={should} | {reason} | "
        f"cooldown_active=False"
    )

    return {
        "should_retrain":  should,
        "reason":          reason,
        "drift_score":     score,
        "drift_detected":  detected,
        "cooldown_active": False,
        "cooldown_status": cooldown_status,
        **threshold_info,
    }


def record_retrain_triggered(
    drift_score:  float | None = None,
    dag_run_id:   str   | None = None,
    state_path:   str          = STATE_PATH,
    history_path: str          = HISTORY_PATH,
) -> None:
    """
    Called when the pipeline actually fires a retraining run.

    - Marks the cooldown start time.
    - Updates the history entry to ``triggered=True``.
    """
    cooldown  = CooldownManager(state_path=state_path, cooldown_hours=COOLDOWN_HOURS)
    threshold = ThresholdManager(history_path=history_path)

    cooldown.record_retrain_triggered(drift_score=drift_score, dag_run_id=dag_run_id)

    # Back-patch the most-recent history entry to triggered=True
    history = threshold._load()
    if history:
        history[-1]["triggered"] = True
        threshold._save(history)

    logger.info(
        f"Retrain recorded: drift_score={drift_score} dag_run_id={dag_run_id}"
    )
