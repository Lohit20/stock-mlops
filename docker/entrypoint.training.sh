#!/bin/sh
# Training container entrypoint.
# Selects the training module based on the MODEL_TYPE environment variable.
# Usage:  docker run -e MODEL_TYPE=arm training-image
#         docker run -e MODEL_TYPE=lstm training-image

set -e

MODEL_TYPE="${MODEL_TYPE:-lstm}"

echo "Starting training for model_type=${MODEL_TYPE}"

case "$MODEL_TYPE" in
  lstm)
    exec python -m src.training.train_lstm "$@"
    ;;
  tft)
    exec python -m src.training.train_tft "$@"
    ;;
  timesfm)
    exec python -m src.training.train_timesfm "$@"
    ;;
  arm)
    exec python -m src.training.train_arm "$@"
    ;;
  all)
    echo "Training all models sequentially"
    python -m src.training.train_lstm   && \
    python -m src.training.train_arm    && \
    python -m src.training.train_timesfm
    # TFT last (heaviest)
    exec python -m src.training.train_tft "$@"
    ;;
  *)
    echo "Unknown MODEL_TYPE='${MODEL_TYPE}'. Valid: lstm | tft | timesfm | arm | all" >&2
    exit 1
    ;;
esac
