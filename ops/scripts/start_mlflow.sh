#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MLOPS_DIR"
docker-compose up -d mlflow
echo "MLflow UI: http://localhost:5000"
