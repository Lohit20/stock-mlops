"""
Shared regression metrics used by all 4 model trainers.
All metrics compare y_true vs y_pred where shape is (n_samples, horizon).
"""

import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute MSE, MAE, RMSE, MAPE and directional accuracy.

    Args:
        y_true: (n_samples, horizon) or (n_samples,)
        y_pred: same shape as y_true

    Returns:
        Dict of float metrics — all serialisable for MLflow logging.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100)

    # Directional accuracy — did we predict up/down correctly at each step?
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        actual_dir = np.sign(np.diff(y_true, axis=-1))
        pred_dir   = np.sign(np.diff(y_pred, axis=-1))
        dir_acc    = float(np.mean(actual_dir == pred_dir))
    else:
        dir_acc = float("nan")

    return {
        "mse":                  mse,
        "mae":                  mae,
        "rmse":                 rmse,
        "mape":                 mape,
        "directional_accuracy": dir_acc,
    }
