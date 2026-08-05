"""Alertas automáticos quando o drift ultrapassa o threshold (Slack/e-mail)."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

log = logging.getLogger(__name__)

SLACK_WEBHOOK_ENV = "ML_SLACK_WEBHOOK_URL"
EMAIL_TO_ENV = "ML_ALERTAS_EMAIL_TO"


def _webhook_url() -> str:
    return os.environ.get(SLACK_WEBHOOK_ENV, "").strip()


def _destinatarios_email() -> List[str]:
    raw = os.environ.get(EMAIL_TO_ENV, "").strip()
    return [e.strip() for e in raw.split(",") if e.strip()]


def notificar_slack(texto: str) -> bool:
    """Envia mensagem ao Slack via webhook."""
    url = _webhook_url()
    if not url or requests is None:
        log.info("Slack não configurado (%s). Mensagem: %s", SLACK_WEBHOOK_ENV, texto)
        return False
    try:
        resp = requests.post(url, json={"text": texto}, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Falha ao notificar Slack: %s", e)
        return False


def notificar_email(assunto: str, corpo: str) -> bool:
    """Envia e-mail via configuração SMTP do Airflow, se configurado."""
    para = _destinatarios_email()
    if not para:
        return False
    try:
        from airflow.utils.email import send_email

        send_email(to=para, subject=assunto, html_content=f"<pre>{corpo}</pre>")
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("Falha ao notificar e-mail: %s", e)
        return False


def _formatar_reporte(data_drift: Dict[str, Any], model_drift: Dict[str, Any]) -> str:
    linhas = [":rotating_light: *Drift detectado nos modelos Aurix*"]
    if data_drift:
        linhas.append(
            f"- *Data drift*: score {data_drift.get('overall_score', 0.0):.4f} "
            f"(threshold {data_drift.get('threshold', 0.0)})"
        )
        drifted = [
            f"{col} ({v['drift_score']:.2f})"
            for col, v in data_drift.get("features", {}).items()
            if v.get("drifted")
        ]
        if drifted:
            linhas.append(f"  - Features com drift: {', '.join(drifted)}")
    if model_drift:
        linhas.append(
            f"- *Model drift*: degradação {model_drift.get('overall_degradation', 0.0):.4f} "
            f"(threshold {model_drift.get('threshold', 0.0)})"
        )
        degraded = [
            f"{col} ({v['degradacao']:.2f})"
            for col, v in model_drift.get("metricas", {}).items()
            if v.get("drifted")
        ]
        if degraded:
            linhas.append(f"  - Métricas degradadas: {', '.join(degraded)}")
    return "\n".join(linhas)


def avaliar_e_alertar(
    data_drift: Dict[str, Any],
    model_drift: Dict[str, Any],
    retrain_triggered: bool = False,
) -> Dict[str, Any]:
    """Avalia relatórios e dispara alertas se qualquer drift ultrapassar threshold."""
    data_drifted = data_drift.get("drift_detected", False)
    model_drifted = model_drift.get("drift_detected", False)

    if not (data_drifted or model_drifted):
        log.info("Nenhum drift detectado — sem alertas.")
        return {"alerted": False}

    texto = _formatar_reporte(data_drift if data_drifted else {}, model_drift if model_drifted else {})
    if retrain_triggered:
        texto += "\n- Retreino automático acionado."

    ok_slack = notificar_slack(texto)
    ok_email = notificar_email("[Aurix ML] Drift detectado", texto)
    log.warning("Alerta de drift disparado (slack=%s, email=%s)", ok_slack, ok_email)
    return {"alerted": True, "slack": ok_slack, "email": ok_email}


def alertar_por_arquivos(data_drift_path: Path, model_drift_path: Path, retrain_triggered: bool = False) -> Dict[str, Any]:
    """Carrega relatórios JSON e avalia alertas."""
    data_drift = {}
    model_drift = {}
    if data_drift_path and Path(data_drift_path).exists():
        with open(data_drift_path) as f:
            data_drift = json.load(f)
    if model_drift_path and Path(model_drift_path).exists():
        with open(model_drift_path) as f:
            model_drift = json.load(f)
    return avaliar_e_alertar(data_drift, model_drift, retrain_triggered=retrain_triggered)
