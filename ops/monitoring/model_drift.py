"""Model drift: monitora métricas de performance do modelo ao longo do tempo.

Compara métricas de performance (AUC, precisão, recall, F1) da janela atual
contra a baseline armazenada em artifacts/model_baseline_metrics.json. Registra
as métricas históricas no MLflow para tracking de performance.

Uso:
    python -m monitoring.model_drift --config config/config.yaml \
        --output artifacts/model_drift_report.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import argparse
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

from fraud_detection_model import FraudDetectionModel, generate_sample_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_baseline_metrics(model_path: Path, n_samples: int = 5000, target_column: str = "is_fraud") -> Dict[str, float]:
    """Treina/reavalia modelo e gera métricas baseline."""
    if model_path.exists():
        model = FraudDetectionModel()
        model.load_model(str(model_path))
    else:
        model = FraudDetectionModel()
        df = generate_sample_data(n_samples=n_samples)
        model.train(df, target_column=target_column)

    df = generate_sample_data(n_samples=n_samples)
    X = model.prepare_features(df)
    X_scaled = model.scaler.transform(X)
    y_true = (df["valor"] > df["valor"].quantile(0.99)).astype(int)
    y_pred = model.random_forest.predict(X_scaled)
    y_prob = model.random_forest.predict_proba(X_scaled)[:, 1]

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precisao": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def compute_current_metrics(model_path: Path, n_samples: int = 1000, target_column: str = "is_fraud") -> Dict[str, float]:
    """Calcula métricas de performance da janela atual."""
    return build_baseline_metrics(model_path, n_samples=n_samples, target_column=target_column)


def detect_model_drift(
    baseline: Dict[str, float],
    current: Dict[str, float],
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Compara métricas atuais com a baseline e avalia degradação (model drift)."""
    metricas = ["auc", "precisao", "recall", "f1", "acuracia"]
    report: Dict[str, Any] = {"drift_detected": False, "metricas": {}, "overall_degradation": 0.0}
    degradacoes: list = []

    for metrica in metricas:
        base = baseline.get(metrica, 0.0)
        atual = current.get(metrica, 0.0)
        # Degradação relativa (queda = drift). Pior caso: -inf -> satura em 1.
        if base <= 0:
            degradacao = 0.0
        else:
            degradacao = min(1.0, max(0.0, (base - atual) / base))
        report["metricas"][metrica] = {
            "baseline": round(base, 4),
            "atual": round(atual, 4),
            "degradacao": round(degradacao, 4),
            "drifted": degradacao >= threshold,
        }
        degradacoes.append(degradacao)

    if degradacoes:
        report["overall_degradation"] = round(float(np.mean(degradacoes)), 4)
        report["drift_detected"] = report["overall_degradation"] >= threshold

    return report


def log_mlflow(report: Dict[str, Any], use_mlflow: bool = True) -> Optional[str]:
    """Registra as métricas de model drift no MLflow."""
    if not use_mlflow:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("aurix-ml-drift")
        with mlflow.start_run(run_name=f"model_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_metric("overall_degradation", report["overall_degradation"])
            for metrica, vals in report.get("metricas", {}).items():
                mlflow.log_metric(f"baseline_{metrica}", vals["baseline"])
                mlflow.log_metric(f"atual_{metrica}", vals["atual"])
                mlflow.log_metric(f"degradacao_{metrica}", vals["degradacao"])
            return run.info.run_id
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow indisponível, continuando sem tracking: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Detecção de model drift")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-path", help="Path do modelo (.pkl)")
    parser.add_argument("--baseline-file", help="Path das métricas baseline JSON")
    parser.add_argument("--baseline-size", type=int, default=5000)
    parser.add_argument("--current-size", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--output", help="Path para salvar o relatório")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = base / args.config
    threshold = args.threshold
    if config_path.exists():
        config = load_config(config_path)
        threshold = config.get("monitoring", {}).get("model_drift_threshold", args.threshold)
        if args.model_path is None:
            args.model_path = config.get("serving", {}).get("model_path")

    model_path = Path(args.model_path) if args.model_path else base.parent / "models" / "fraud_detection_model.pkl"
    if not model_path.is_absolute():
        model_path = base.parent / model_path

    if args.baseline_file and Path(args.baseline_file).exists():
        with open(args.baseline_file) as f:
            baseline = json.load(f)
        logger.info("Loaded baseline from %s", args.baseline_file)
    else:
        logger.info("Building baseline metrics (n=%s)...", args.baseline_size)
        baseline = build_baseline_metrics(model_path, n_samples=args.baseline_size)
        baseline_path = base / "artifacts" / "model_baseline_metrics.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(baseline_path, "w") as f:
            json.dump(baseline, f, indent=2)
        logger.info("Saved baseline to %s", baseline_path)

    current = compute_current_metrics(model_path, n_samples=args.current_size)
    report = detect_model_drift(baseline, current, threshold=threshold)
    report["timestamp"] = datetime.utcnow().isoformat() + "Z"
    report["threshold"] = threshold
    report["mlflow_run_id"] = log_mlflow(report, use_mlflow=not args.no_mlflow)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote model drift report to %s", args.output)

    logger.info("Model drift detected: %s (overall_degradation=%.4f)", report["drift_detected"], report["overall_degradation"])
    return report


if __name__ == "__main__":
    main()
