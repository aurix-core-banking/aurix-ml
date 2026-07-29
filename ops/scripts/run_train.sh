#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MLOPS_DIR"
mkdir -p models artifacts
echo "Running training pipeline..."
python -m pipelines.train_pipeline --config config/config.yaml --model-dir models
echo "Training finished. Model saved to $MLOPS_DIR/models"
