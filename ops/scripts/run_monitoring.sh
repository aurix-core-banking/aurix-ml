#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLOPS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MLOPS_DIR"
mkdir -p artifacts

echo "==> 1/4 Data drift check"
python3 -m monitoring.drift_detection --config config/config.yaml --output artifacts/drift_report.json

echo "==> 2/4 Model drift check"
python3 -m monitoring.model_drift --config config/config.yaml --output artifacts/model_drift_report.json

echo "==> 3/4 Exportando métricas para Prometheus"
python3 -m monitoring.prometheus_exporter --config config/config.yaml --once

echo "==> 4/4 Avaliando alertas e retreino automático"
python3 - <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, '.')
from monitoring.alerting import alertar_por_arquivos
from monitoring.retraining import ler_relatorios, ha_drift

base = Path('.').resolve()
relatorios = ler_relatorios(base)

alertar_por_arquivos(
    base / 'artifacts' / 'drift_report.json',
    base / 'artifacts' / 'model_drift_report.json',
    retrain_triggered=False,
)

if ha_drift(relatorios):
    print("Drift detectado — acionando retreino automático")
    from monitoring.retraining import main as retrain_main
    sys.argv = ['retraining', '--config', 'config/config.yaml']
    retrain_main()
else:
    print("Nenhum drift — retreino não acionado")
EOF

echo "==> Monitoramento concluído"
