"""Detecção de data drift (features de entrada vs baseline).

Compara a distribuição das features atuais com a baseline (referência) usando
PSI (Population Stability Index) para features numéricas e categóricas. Suporta
registro dos scores de drift no MLflow.

Uso:
    python -m monitoring.drift_detection --config config/config.yaml \
        --output artifacts/drift_report.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import argparse
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from fraud_detection_model import generate_sample_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EPSILON = 1e-6
NUMERIC_DEFAULT = ["valor", "score_risco", "latitude", "longitude", "tempo_processamento_ms"]
CATEGORICAL_DEFAULT = ["tipo_transacao", "status", "canal", "cidade", "estado"]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _compute_psi(esperado: np.ndarray, atual: np.ndarray) -> float:
    """Calcula o PSI entre duas distribuições de proporções."""
    esp = np.clip(np.asarray(esperado, dtype=float), EPSILON, None)
    atu = np.clip(np.asarray(atual, dtype=float), EPSILON, None)
    esp = esp / esp.sum()
    atu = atu / atu.sum()
    return float(np.sum((atu - esp) * np.log(atu / esp)))


def compute_reference_stats(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    n_bins: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Calcula a estatística de referência (baseline) por feature."""
    numeric_cols = numeric_cols or [c for c in NUMERIC_DEFAULT if c in df.columns]
    categorical_cols = categorical_cols or [c for c in CATEGORICAL_DEFAULT if c in df.columns]
    stats: Dict[str, Dict[str, Any]] = {"tipo": "referencia", "features": {}}

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < n_bins:
            continue
        bins = np.quantile(data, np.linspace(0, 1, n_bins + 1))
        bins = np.unique(bins)
        if len(bins) < 2:
            continue
        counts, _ = np.histogram(data, bins=bins)
        probs = counts / counts.sum()
        stats["features"][col] = {
            "tipo": "numerica",
            "bins": bins.tolist(),
            "proporcoes": probs.tolist(),
            "mean": float(data.mean()),
            "std": float(data.std()) if data.std() > 0 else 1e-6,
        }

    for col in categorical_cols:
        if col not in df.columns:
            continue
        probs = df[col].value_counts(normalize=True)
        stats["features"][col] = {
            "tipo": "categorica",
            "categorias": probs.index.astype(str).tolist(),
            "proporcoes": probs.values.tolist(),
        }

    return stats


def detect_drift(
    reference_stats: Dict[str, Dict[str, Any]],
    current_df: pd.DataFrame,
    threshold: float = 0.15,
) -> Dict[str, Any]:
    """Detecta drift comparando a distribuição atual com a referência."""
    features = reference_stats.get("features", {})
    report: Dict[str, Any] = {"drift_detected": False, "features": {}, "overall_score": 0.0}
    scores: List[float] = []

    for col, ref in features.items():
        if col not in current_df.columns:
            continue
        if ref["tipo"] == "numerica":
            curr = current_df[col].dropna()
            if len(curr) < 10:
                continue
            edges = np.array(ref["bins"])
            clipped = np.clip(curr, edges[0], edges[-1])
            counts, _ = np.histogram(clipped, bins=edges)
            probs = counts / counts.sum()
            psi = _compute_psi(ref["proporcoes"], probs)
        else:
            counts = current_df[col].value_counts()
            categorias = ref["categorias"]
            atuais = np.array([counts.get(c, 0) for c in categorias], dtype=float)
            extras = sum(v for k, v in counts.items() if k not in set(categorias))
            atuais = np.append(atuais, float(extras))
            esperados = np.append(np.array(ref["proporcoes"]), 0.0)
            psi = _compute_psi(esperados, atuais)

        drift_score = min(1.0, psi / 0.25)
        report["features"][col] = {
            "drift_score": round(drift_score, 4),
            "psi": round(psi, 4),
            "drifted": drift_score >= threshold,
        }
        scores.append(drift_score)

    if scores:
        report["overall_score"] = round(float(np.max(scores)), 4)
        report["drift_detected"] = report["overall_score"] >= threshold

    return report


def log_mlflow(report: Dict[str, Any], use_mlflow: bool = True) -> Optional[str]:
    """Registra os scores de drift no MLflow."""
    if not use_mlflow:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("aurix-ml-drift")
        with mlflow.start_run(run_name=f"data_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            mlflow.log_metric("overall_drift", report["overall_score"])
            for col, feat in report.get("features", {}).items():
                mlflow.log_metric(f"drift_{col}", feat["drift_score"])
            return run.info.run_id
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow indisponível, continuando sem tracking: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Detecção de data drift")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--reference", help="Path para stats de referência JSON")
    parser.add_argument("--reference-data", type=int, default=5000, help="Tamanho da amostra de referência")
    parser.add_argument("--current-size", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output", help="Path para salvar o relatório de drift")
    parser.add_argument("--no-mlflow", action="store_true", help="Desabilita MLflow")
    parser.add_argument("--current-data", help="Path CSV/Parquet com dados atuais")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = base / args.config
    threshold = args.threshold
    if config_path.exists():
        config = load_config(config_path)
        threshold = config.get("monitoring", {}).get("drift_threshold", args.threshold)

    if args.reference and Path(args.reference).exists():
        with open(args.reference) as f:
            reference_stats = json.load(f)
        logger.info("Loaded reference stats from %s", args.reference)
    else:
        logger.info("Building reference stats from sample (n=%s)", args.reference_data)
        ref_df = generate_sample_data(n_samples=args.reference_data)
        reference_stats = compute_reference_stats(ref_df)
        ref_path = base / "artifacts" / "reference_stats.json"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(reference_stats, f, indent=2)
        logger.info("Saved reference stats to %s", ref_path)

    if args.current_data:
        if str(args.current_data).endswith(".csv"):
            current_df = pd.read_csv(args.current_data)
        else:
            current_df = pd.read_parquet(args.current_data)
        logger.info("Loaded current data from %s", args.current_data)
    else:
        current_df = generate_sample_data(n_samples=args.current_size)

    report = detect_drift(reference_stats, current_df, threshold=threshold)
    report["timestamp"] = datetime.utcnow().isoformat() + "Z"
    report["threshold"] = threshold
    report["mlflow_run_id"] = log_mlflow(report, use_mlflow=not args.no_mlflow)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote drift report to %s", args.output)

    logger.info("Drift detected: %s (overall_score=%.4f)", report["drift_detected"], report["overall_score"])
    return report


if __name__ == "__main__":
    main()
