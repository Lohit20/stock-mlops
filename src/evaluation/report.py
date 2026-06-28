"""
Comparison report artifacts for the Model Comparison MLflow run.

Functions:
  comparison_bar_chart()    — side-by-side bar chart of RMSE / MAPE / dir-acc
  per_symbol_winner_chart() — grid showing which model wins per symbol
  generate_model_card()     — JSON model card for the winning model
  log_all_artifacts()       — logs all charts + card to the active MLflow run
"""

import io
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger


# ── Chart: aggregated metric comparison ───────────────────────────────────────

def comparison_bar_chart(agg_df: pd.DataFrame, artifact_path: str = "plots"):
    """
    Side-by-side bar chart of RMSE, MAPE and directional accuracy per model.
    Saved as PNG and logged to the active MLflow run.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    metrics  = ["rmse", "mape", "directional_accuracy"]
    titles   = ["RMSE (lower = better)", "MAPE % (lower = better)",
                 "Directional Accuracy (higher = better)"]
    models   = agg_df["model_type"].tolist()
    n_models = len(models)
    x        = np.arange(n_models)
    width    = 0.6

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]

    for ax, metric, title in zip(axes, metrics, titles):
        vals = agg_df[metric].values if metric in agg_df.columns else np.zeros(n_models)
        bars = ax.bar(x, vals, width=width,
                      color=colors[:n_models], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Model Comparison — Held-Out Test Set", fontsize=13, y=1.02)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        tmp_path = tmp.name
    plt.close(fig)

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug("Comparison bar chart logged")


# ── Chart: per-symbol winner heatmap ─────────────────────────────────────────

def per_symbol_winner_chart(per_symbol_df: pd.DataFrame, artifact_path: str = "plots"):
    """
    Heatmap grid: rows = symbols, columns = models, cell = RMSE.
    Winning cell (lowest RMSE) highlighted in green.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    if per_symbol_df.empty:
        return

    pivot = per_symbol_df.pivot_table(
        index="symbol", columns="model_type", values="rmse", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2),
                                    max(4, len(pivot.index) * 0.6)),
                           tight_layout=True)

    data   = pivot.values
    n_rows, n_cols = data.shape

    # Color map: green = best, red = worst per row
    row_min = np.nanmin(data, axis=1, keepdims=True)
    normed  = (data - row_min) / (np.nanmax(data, axis=1, keepdims=True) - row_min + 1e-9)

    im = ax.imshow(normed, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Relative RMSE (green = best)")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(pivot.columns.tolist(), fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.set_title("Per-Symbol RMSE — lower is better", fontsize=12, fontweight="bold")

    # Annotate cells with actual RMSE
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        tmp_path = tmp.name
    plt.close(fig)

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug("Per-symbol winner chart logged")


# ── Chart: per-symbol win count ───────────────────────────────────────────────

def win_count_chart(per_symbol_df: pd.DataFrame, artifact_path: str = "plots"):
    """
    Bar chart showing how many symbols each model wins (lowest RMSE).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mlflow

    if per_symbol_df.empty:
        return

    pivot = per_symbol_df.pivot_table(
        index="symbol", columns="model_type", values="rmse", aggfunc="mean"
    )
    winners = pivot.idxmin(axis=1).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    colors = ["steelblue", "darkorange", "seagreen", "firebrick"]
    ax.bar(winners.index, winners.values,
           color=colors[:len(winners)], edgecolor="white")
    ax.set_title("Win Count per Model (lowest RMSE per symbol)", fontsize=11,
                 fontweight="bold")
    ax.set_ylabel("Number of symbols won")
    ax.grid(axis="y", alpha=0.3)

    for i, (model, count) in enumerate(winners.items()):
        ax.text(i, count + 0.05, str(count), ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
        tmp_path = tmp.name
    plt.close(fig)

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug("Win-count chart logged")


# ── Per-symbol metrics table ──────────────────────────────────────────────────

def log_per_symbol_table(per_symbol_df: pd.DataFrame, artifact_path: str = "metrics"):
    """Save the full per-symbol comparison CSV as an artifact."""
    import mlflow

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        per_symbol_df.sort_values(["model_type", "symbol"]).to_csv(tmp, index=False)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.debug("Per-symbol comparison table logged")


# ── Model card ────────────────────────────────────────────────────────────────

def generate_model_card(
    best_info:     dict,
    agg_df:        pd.DataFrame,
    per_symbol_df: pd.DataFrame,
) -> dict:
    """
    Generate a structured model card JSON for the winning model.

    Schema mirrors the Hugging Face Model Card format for interoperability.
    """
    model_type = best_info.get("model_type", "Unknown")

    # Per-symbol wins for the best model
    if not per_symbol_df.empty and "model_type" in per_symbol_df.columns:
        pivot    = per_symbol_df.pivot_table(
            index="symbol", columns="model_type", values="rmse", aggfunc="mean"
        )
        n_wins   = int((pivot.idxmin(axis=1) == model_type).sum()) if model_type in pivot.columns else 0
        n_total  = len(pivot)
    else:
        n_wins, n_total = 0, 0

    # Best model row in aggregated df
    best_row = agg_df[agg_df["model_type"] == model_type]
    avg_rmse = float(best_row["rmse"].values[0])  if not best_row.empty else None
    avg_mape = float(best_row["mape"].values[0])  if not best_row.empty else None
    avg_da   = float(best_row["directional_accuracy"].values[0]) if not best_row.empty else None

    card = {
        "model_id":    f"stock_price_forecaster_{model_type.lower()}",
        "model_type":  model_type,
        "run_id":      best_info.get("run_id", ""),
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name":      "NASDAQ stock prices",
            "source":    "yfinance",
            "features":  [
                "close", "returns", "volatility_7", "rsi_14",
                "sma_7", "sma_30", "ema_7", "ema_30", "bb_width",
            ],
        },
        "evaluation": {
            "horizon_days":     30,
            "context_days":     90,
            "avg_rmse":         avg_rmse,
            "avg_mape_pct":     avg_mape,
            "avg_dir_accuracy": avg_da,
            "symbols_won":      n_wins,
            "symbols_total":    n_total,
        },
        "intended_use":     "30-day closing price forecasting for NASDAQ equities",
        "limitations":      [
            "Does not incorporate news, earnings, or macro events",
            "Trained on limited symbol set — performance may vary on unseen tickers",
            "Financial forecasting is inherently uncertain; not investment advice",
        ],
        "registered_model": f"models:/stock_price_forecaster_{model_type.lower()}/Production",
    }

    return card


def log_model_card(card: dict, artifact_path: str = "model_card"):
    """Log model card JSON as an MLflow artifact."""
    import mlflow

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        json.dump(card, tmp, indent=2)
        tmp_path = tmp.name

    mlflow.log_artifact(tmp_path, artifact_path=artifact_path)
    os.unlink(tmp_path)
    logger.info("Model card logged to MLflow")
    return card


# ── Aggregate logger ──────────────────────────────────────────────────────────

def log_all_artifacts(
    per_symbol_df: pd.DataFrame,
    agg_df:        pd.DataFrame,
    best_info:     dict,
):
    """Log all comparison artifacts to the currently active MLflow run."""
    comparison_bar_chart(agg_df)
    if not per_symbol_df.empty:
        per_symbol_winner_chart(per_symbol_df)
        win_count_chart(per_symbol_df)
        log_per_symbol_table(per_symbol_df)
    card = generate_model_card(best_info, agg_df, per_symbol_df)
    log_model_card(card)
    return card
