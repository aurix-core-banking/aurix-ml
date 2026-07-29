import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import argparse
import yaml
import logging
from datetime import datetime

import mlflow
import mlflow.sklearn

from fraud_detection_model import FraudDetectionModel, generate_sample_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_training(config: dict, model_dir: Path, use_mlflow: bool = True) -> dict:
    train_cfg = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})
    sample_size = train_cfg.get("sample_size", 5000)
    target_col = train_cfg.get("target_column", "is_fraud")

    logger.info("Generating sample data (size=%s)...", sample_size)
    df = generate_sample_data(n_samples=sample_size)

    if use_mlflow and mlflow_cfg.get("tracking_uri"):
        try:
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "aurix-fraud-detection"))
        except Exception as e:
            logger.warning("MLflow not available, continuing without: %s", e)
            use_mlflow = False

    run_id = None
    if use_mlflow:
        with mlflow.start_run(run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            run_id = run.info.run_id
            model = FraudDetectionModel()
            model.train(df, target_column=target_col)

            mlflow.log_param("sample_size", sample_size)
            mlflow.log_param("target_column", target_col)

            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "fraud_detection_model.pkl"
            model.save_model(str(model_path))
            mlflow.log_artifact(str(model_path))

            if mlflow_cfg.get("registry_model_name"):
                mlflow.sklearn.log_model(
                    model.random_forest,
                    "model",
                    registered_model_name=mlflow_cfg["registry_model_name"],
                )
    else:
        model = FraudDetectionModel()
        model.train(df, target_column=target_col)
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "fraud_detection_model.pkl"
        model.save_model(str(model_path))

    metrics = {}
    if use_mlflow and run_id:
        metrics["mlflow_run_id"] = run_id
    metrics["model_path"] = str(model_dir / "fraud_detection_model.pkl")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--model-dir", default="models", help="Directory to save model")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = base / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    config = load_config(config_path)
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = base / args.model_dir

    metrics = run_training(config, model_dir, use_mlflow=not args.no_mlflow)
    logger.info("Training finished: %s", metrics)


if __name__ == "__main__":
    main()
