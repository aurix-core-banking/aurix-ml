"""Exporter Prometheus das métricas de drift para visualização no Grafana.

Expõe os scores de data drift e model drift como gauges Prometheus em
/var/tmp/aurix_ml_drift.prom (formato textfile do node_exporter) e também via
HTTP próprio (porta 9101).

Uso:
    python -m monitoring.prometheus_exporter --config config/config.yaml
"""

import argparse
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, Any, Optional

from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

METRICAS: Dict[str, Gauge] = {}

def _get_gauge(nome: str, descricao: str, rotulo: str = "") -> Gauge:
    chave = f"{nome}|{rotulo}"
    if chave not in METRICAS:
        if rotulo:
            METRICAS[chave] = Gauge(nome, descricao, [rotulo])
        else:
            METRICAS[chave] = Gauge(nome, descricao)
    return METRICAS[chave]


def carregar_relatorios(base_dir: Path) -> Dict[str, Any]:
    data_drift = base_dir / "artifacts" / "drift_report.json"
    model_drift = base_dir / "artifacts" / "model_drift_report.json"
    dados = {}
    if data_drift.exists():
        with open(data_drift) as f:
            dados["data_drift"] = json.load(f)
    if model_drift.exists():
        with open(model_drift) as f:
            dados["model_drift"] = json.load(f)
    return dados


def exportar_para_prometheus(base_dir: Path) -> None:
    """Atualiza os gauges Prometheus a partir dos relatórios de drift."""
    dados = carregar_relatorios(base_dir)
    data = dados.get("data_drift", {})
    model = dados.get("model_drift", {})

    g_overall = _get_gauge("aurix_ml_data_drift_score", "Score de data drift (PSI normalizado)")
    g_overall.set(float(data.get("overall_score", 0.0)))
    g_overall_detected = _get_gauge("aurix_ml_data_drift_detected", "Data drift detectado (1/0)")
    g_overall_detected.set(1.0 if data.get("drift_detected") else 0.0)

    g_model = _get_gauge("aurix_ml_model_degradation", "Degradação de performance do modelo (model drift)")
    g_model.set(float(model.get("overall_degradation", 0.0)))
    g_model_detected = _get_gauge("aurix_ml_model_drift_detected", "Model drift detectado (1/0)")
    g_model_detected.set(1.0 if model.get("drift_detected") else 0.0)

    for col, feat in data.get("features", {}).items():
        g_feat = _get_gauge("aurix_ml_feature_drift_score", "Score de drift por feature", "feature")
        g_feat.labels(feature=col).set(float(feat.get("drift_score", 0.0)))

    for metrica, vals in model.get("metricas", {}).items():
        g_met = _get_gauge("aurix_ml_metric_degradation", "Degradação por métrica", "metrica")
        g_met.labels(metrica=metrica).set(float(vals.get("degradacao", 0.0)))

    # Persiste no formato textfile do node_exporter
    textfile_dir = Path(os.environ.get("PROMETHEUS_TEXTFILE_DIR", str(base_dir / "artifacts")))
    textfile_dir.mkdir(parents=True, exist_ok=True)
    path = textfile_dir / "aurix_ml_drift.prom"
    with open(path, "wb") as f:
        f.write(generate_latest())
    logger.info("Métricas de drift exportadas para %s", path)


def _setup_handler(base_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/metrics":
                exportar_para_prometheus(base_dir)
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.end_headers()
                self.wfile.write(generate_latest())
            elif self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):  # noqa: A003
            logger.info("HTTP %s", args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Exporter Prometheus de métricas de drift")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DRIFT_EXPORTER_PORT", "9101")))
    parser.add_argument("--once", action="store_true", help="Exporta uma única vez e sai")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    exportar_para_prometheus(base)

    if args.once:
        return

    server = HTTPServer(("0.0.0.0", args.port), _setup_handler(base))
    logger.info("Drift exporter listening on :%d/metrics", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
