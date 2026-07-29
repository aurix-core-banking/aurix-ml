#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MLOPS_DIR"
export AUREUS_MODEL_PATH="${MLOPS_DIR}/models/fraud_detection_model.pkl"
if [ ! -f "$AUREUS_MODEL_PATH" ]; then
  echo "Model not found at $AUREUS_MODEL_PATH. Run run_train.sh first."
  exit 1
fi
echo "Starting ML serving on http://0.0.0.0:8000"
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
