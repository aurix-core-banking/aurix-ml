#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MLOPS_DIR"
mkdir -p artifacts
python -m monitoring.drift_detection --config config/config.yaml --output artifacts/drift_report.json
