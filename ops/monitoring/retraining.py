"""Retreino automático quando drift for detectado (com MLflow model registry).

- Lê os relatórios de data drift e model drift (artifacts/).
- Se qualquer um ultrapassar o threshold, dispara o pipeline de treino
  (pipelines.train_pipeline) e promove o modelo de staging para production no
  MLflow Model Registry.

Uso:
    python -m monitoring.retraining --config config/config.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def ler_relatorios(base_dir: Path) -> Dict[str, Any]:
    relatorios: Dict[str, Any] = {}
    for nome in ("drift_report.json", "model_drift_report.json"):
        caminho = base_dir / "artifacts" / nome
        if caminho.exists():
            with open(caminho) as f:
                relatorios[nome] = json.load(f)
    return relatorios


def ha_drift(relatorios: Dict[str, Any]) -> bool:
    return any(r.get("drift_detected", False) for r in relatorios.values())


def promover_modelo_mlflow(modelo: str, stage: str = "Production") -> None:
    """Promove o modelo no MLflow Model Registry para o estágio informado."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri("http://localhost:5000")
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{modelo}'")
        if not versions:
            logger.warning("Modelo %s não registrado no MLflow — nenhuma promoção.", modelo)
            return
        # Mais recente
        latest = sorted(versions, key=lambda v: v.creation_timestamp or 0, reverse=True)[0]
        client.transition_model_version_stage(
            name=modelo, version=latest.version, stage=stage, archive_existing_versions=True
        )
        logger.info("Modelo %s (versão %s) promovido para %s", modelo, latest.version, stage)
    except Exception as e:  # noqa: BLE001
        logger.warning("MLflow indisponível, sem promoção no registry: %s", e)


def disparar_retreino(base_dir: Path, config: dict, use_mlflow: bool = True) -> Dict[str, Any]:
    """Dispara o pipeline de treino e promove o modelo no registry."""
    treino = config.get("training", {})
    mlflow_cfg = config.get("mlflow", {})
    modelo = mlflow_cfg.get("registry_model_name", treino.get("model_name", "fraud_detection"))

    resultado: Dict[str, Any] = {
        "retrain_triggered": True,
        "modelo": modelo,
        "inicio": datetime.now().isoformat(),
    }

    # 1. Rodar pipeline de treino
    try:
        cmd = [
            sys.executable,
            "-m",
            "pipelines.train_pipeline",
            "--config",
            str(base_dir / "config" / "config.yaml"),
            "--model-dir",
            str(base_dir.parent / "models"),
        ]
        if not use_mlflow:
            cmd.append("--no-mlflow")
        proc = subprocess.run(cmd, cwd=str(base_dir), capture_output=True, text=True, timeout=3600)
        resultado["treino_ok"] = proc.returncode == 0
        resultado["saida"] = proc.stdout[-2000:] if proc.stdout else ""
        if proc.returncode != 0:
            logger.error("Pipeline de treino falhou:\n%s", proc.stderr[-2000:])
    except Exception as e:  # noqa: BLE001
        logger.error("Falha ao disparar retreino: %s", e)
        resultado["treino_ok"] = False
        resultado["erro"] = str(e)

    # 2. Promover para production
    if use_mlflow and resultado.get("treino_ok"):
        promover_modelo_mlflow(modelo)

    resultado["fim"] = datetime.now().isoformat()
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Retreino automático por drift")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true", help="Dispara retreino mesmo sem drift")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    config_path = base / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    config = load_config(config_path)

    relatorios = ler_relatorios(base)
    if not ha_drift(relatorios) and not args.force:
        logger.info("Nenhum drift detectado — retreino não acionado.")
        return {"retrain_triggered": False}

    resultado = disparar_retreino(base, config, use_mlflow=not args.no_mlflow)
    logger.info("Resultado do retreino: %s", resultado)
    return resultado


if __name__ == "__main__":
    main()
