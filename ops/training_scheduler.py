"""
Aurix ML — Agendador de Treinamento Automático (APScheduler).

Jobs agendados:
  - Diário (02:00): Feature engineering (reprocessa features do Feature Store)
  - Semanal (dom, 03:00): Retreino dos modelos (credit, fraud, segmentation)
  - Mensal (dia 1, 04:00): Hyperparameter tuning (GridSearch/Optuna)

Uso:
    python -m aurix.ml.ops.training_scheduler
    # ou
    python aurix-ml/ops/training_scheduler.py
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("aurix.ml.scheduler")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OPS_DIR = BASE_DIR / "ops"
PIPELINES_DIR = BASE_DIR / "pipelines"


# ═══════════════════════════════════════════════════════════
# Utilitários
# ═══════════════════════════════════════════════════════════

def _executar_comando(cmd: list, descricao: str, timeout: int = 3600) -> Dict[str, Any]:
    """Executa um subprocesso e retorna o resultado."""
    logger.info("Iniciando: %s → %s", descricao, " ".join(cmd))
    inicio = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BASE_DIR),
        )
        duracao = time.time() - inicio
        resultado = {
            "descricao": descricao,
            "sucesso": proc.returncode == 0,
            "duracao_seg": round(duracao, 2),
            "saida": proc.stdout[-2000:] if proc.stdout else "",
            "erros": proc.stderr[-1000:] if proc.stderr else "",
            "timestamp": datetime.now().isoformat(),
        }
        if proc.returncode == 0:
            logger.info("Concluído: %s (%.1fs)", descricao, duracao)
        else:
            logger.error("Falhou: %s (rc=%d)\n%s", descricao, proc.returncode, proc.stderr[-500:])
        return resultado
    except subprocess.TimeoutExpired:
        logger.error("Timeout: %s (%ds)", descricao, timeout)
        return {"descricao": descricao, "sucesso": False, "erro": "timeout", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error("Erro: %s — %s", descricao, e)
        return {"descricao": descricao, "sucesso": False, "erro": str(e), "timestamp": datetime.now().isoformat()}


# ═══════════════════════════════════════════════════════════
# Jobs de Treinamento
# ═══════════════════════════════════════════════════════════

def job_feature_engineering():
    """Diário às 02:00 — Reprocessa features do Feature Store.

    Executa pipelines de feature engineering que atualizam as
    features materializadas no Feature Store (Feast).
    """
    logger.info("═══ Feature Engineering Diário ═══")

    resultados = []

    # 1. Rebuild credit features
    resultados.append(_executar_comando(
        [sys.executable, "-m", "pipelines.feature_engineering", "--target", "credit"],
        "Feature Engineering — Credit Features",
    ))

    # 2. Rebuild fraud features
    resultados.append(_executar_comando(
        [sys.executable, "-m", "pipelines.feature_engineering", "--target", "fraud"],
        "Feature Engineering — Fraud Features",
    ))

    # 3. Rebuild customer features
    resultados.append(_executar_comando(
        [sys.executable, "-m", "pipelines.feature_engineering", "--target", "customer"],
        "Feature Engineering — Customer Features",
    ))

    sucesso = all(r["sucesso"] for r in resultados)
    logger.info("Feature Engineering concluído: %s", "OK" if sucesso else "FALHA")
    return resultados


def job_retreino_semanal():
    """Semanal (dom, 03:00) — Retreina todos os modelos.

    Executa o pipeline completo de treino para:
    - credit_risk_model
    - fraud_detection_model
    - customer_segmentation_model
    """
    logger.info("═══ Retreino Semanal ═══")

    resultados = []

    # Treinar todos os modelos via train_models.py
    resultados.append(_executar_comando(
        [sys.executable, str(MODELS_DIR / "train_models.py")],
        "Treino Geral — Todos os Modelos",
        timeout=7200,
    ))

    # Treinar pipeline de crédito especificamente
    resultados.append(_executar_comando(
        [sys.executable, str(MODELS_DIR / "train_credit_pipeline.py")],
        "Treino Pipeline de Crédito",
        timeout=3600,
    ))

    # Verificar drift após treino
    resultados.append(_executar_comando(
        [sys.executable, str(OPS_DIR / "monitoring" / "drift_detection.py")],
        "Verificação de Drift Pós-Treino",
    ))

    sucesso = all(r["sucesso"] for r in resultados)
    logger.info("Retreino semanal concluído: %s", "OK" if sucesso else "FALHA")
    return resultados


def job_hyperparameter_tuning():
    """Mensal (dia 1, 04:00) — Hyperparameter tuning.

    Executa busca de hiperparâmetros com Optuna/GridSearch
    para os modelos principais.
    """
    logger.info("═══ Hyperparameter Tuning Mensal ═══")

    resultados = []

    # Tuning do modelo de crédito
    resultados.append(_executar_comando(
        [sys.executable, str(OPS_DIR / "pipelines" / "train_pipeline.py"),
         "--config", "config/config.yaml", "--tune"],
        "Hyperparameter Tuning — Credit Risk",
        timeout=14400,
    ))

    # Tuning do modelo de fraude
    resultados.append(_executar_comando(
        [sys.executable, str(MODELS_DIR / "fraud_detection_model.py"), "--tune"],
        "Hyperparameter Tuning — Fraud Detection",
        timeout=14400,
    ))

    sucesso = all(r["sucesso"] for r in resultados)
    logger.info("Hyperparameter tuning concluído: %s", "OK" if sucesso else "FALHA")
    return resultados


def job_drift_check():
    """Diário às 06:00 — Verificação de drift.

    Roda detecção de data drift e model drift. Se drift significativo
    for detectado, dispara retreino.
    """
    logger.info("═══ Verificação de Drift ═══")

    resultado = _executar_comando(
        [sys.executable, str(OPS_DIR / "monitoring" / "drift_detection.py")],
        "Drift Detection",
    )

    # Se drift detectado, disparar retreino
    if resultado["sucesso"] and "drift_detected: true" in resultado.get("saida", "").lower():
        logger.warning("Drift detectado — disparando retreino emergencial")
        _executar_comando(
            [sys.executable, str(OPS_DIR / "monitoring" / "retraining.py"), "--force"],
            "Retreino Emergencial (Drift Detectado)",
            timeout=7200,
        )

    return resultado


# ═══════════════════════════════════════════════════════════
# Listener de eventos
# ═══════════════════════════════════════════════════════════

def listener_eventos(event):
    """Listener para eventos do APScheduler."""
    if event.exception:
        logger.error(
            "Job %s falhou: %s",
            event.job_id,
            event.exception,
        )
    else:
        logger.info(
            "Job %s concluído com sucesso (execução %s)",
            event.job_id,
            event.scheduled_run_time,
        )


# ═══════════════════════════════════════════════════════════
# Agendador principal
# ═══════════════════════════════════════════════════════════

def criar_agendador() -> BlockingScheduler:
    """Cria e configura o agendador com todos os jobs."""
    scheduler = BlockingScheduler(timezone="America/Sao_Paulo")

    # Adicionar listener de eventos
    scheduler.add_listener(listener_eventos, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)

    # ── Diário: Feature Engineering (02:00) ──
    scheduler.add_job(
        job_feature_engineering,
        CronTrigger(hour=2, minute=0),
        id="feature_engineering_diario",
        name="Feature Engineering Diário",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    # ── Diário: Drift Check (06:00) ──
    scheduler.add_job(
        job_drift_check,
        CronTrigger(hour=6, minute=0),
        id="drift_check_diario",
        name="Verificação de Drift Diária",
        misfire_grace_time=3600,
        coalesce=True,
        max_instances=1,
    )

    # ── Semanal: Retreino (dom, 03:00) ──
    scheduler.add_job(
        job_retreino_semanal,
        CronTrigger(day_of_week="sun", hour=3, minute=0),
        id="retreino_semanal",
        name="Retreino Semanal",
        misfire_grace_time=7200,
        coalesce=True,
        max_instances=1,
    )

    # ── Mensal: Hyperparameter Tuning (dia 1, 04:00) ──
    scheduler.add_job(
        job_hyperparameter_tuning,
        CronTrigger(day=1, hour=4, minute=0),
        id="hp_tuning_mensal",
        name="Hyperparameter Tuning Mensal",
        misfire_grace_time=14400,
        coalesce=True,
        max_instances=1,
    )

    logger.info("Agendador de treinamento Aurix ML configurado:")
    logger.info("  → Feature Engineering:  diário às 02:00")
    logger.info("  → Drift Check:          diário às 06:00")
    logger.info("  → Retreino:             semanal (dom) às 03:00")
    logger.info("  → HP Tuning:            mensal (dia 1) às 04:00")

    return scheduler


def main():
    """Ponto de entrada principal."""
    scheduler = criar_agendador()

    try:
        logger.info("Iniciando agendador de treinamento Aurix ML...")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Agendador encerrado.")
        scheduler.shutdown()


if __name__ == "__main__":
    main()
