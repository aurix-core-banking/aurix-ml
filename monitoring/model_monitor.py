"""
Monitor de Modelo — trackeia predicoes, monitora degradacao de metricas, alertas.

Funcionalidades:
- Registro de predicoes com timestamp
- Monitoramento de AUC/precision/recall ao longo do tempo
- Alertas quando metricas caem abaixo de threshold
- Dados para dashboard Grafana (exportacao Prometheus)

Uso:
    python -m monitoring.model_monitor --config ../ops/config/config.yaml
    python -m monitoring.model_monitor --check-decay --window-days 7
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Metricas minima para considerar validas
MIN_AMOSTRAS = 100

# Thresholds de alerta
DEFAULT_THRESHOLDS = {
    "auc_minimo": 0.8,
    "precision_minimo": 0.7,
    "recall_minimo": 0.6,
    "f1_minimo": 0.65,
    "drift_psi_alerta": 0.2,
}


class ModelMonitor:
    """Monitor de performance do modelo de credito."""

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        metrics_dir: Optional[str] = None,
    ):
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.metrics_dir = Path(metrics_dir) if metrics_dir else Path("metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Historico de metricas (em memoria)
        self.historico_metricas: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self.historico_predicoes: Deque[Dict[str, Any]] = deque(maxlen=50000)

        # Carrega historico existente
        self._carregar_historico()

    def _carregar_historico(self) -> None:
        """Carrega historico de metricas de arquivos JSONL."""
        for arquivo in sorted(self.metrics_dir.glob("metrics_*.jsonl")):
            try:
                with open(arquivo) as f:
                    for linha in f:
                        if linha.strip():
                            self.historico_metricas.append(json.loads(linha))
            except Exception as e:
                logger.warning("Erro ao carregar %s: %s", arquivo, e)

    def registrar_predicao(
        self,
        id_cliente: int,
        probabilidade: float,
        score: int,
        nivel_risco: str,
        modelo_versao: str = "2.0.0",
    ) -> None:
        """Registra uma predicao individual."""
        registro = {
            "timestamp": datetime.now().isoformat(),
            "id_cliente": id_cliente,
            "probabilidade": round(probabilidade, 6),
            "score": score,
            "nivel_risco": nivel_risco,
            "modelo_versao": modelo_versao,
        }
        self.historico_predicoes.append(registro)

        # Salva incrementalmente
        arquivo = self.metrics_dir / f"predicoes_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(arquivo, "a") as f:
            f.write(json.dumps(registro) + "\n")

    def registrar_metricas(
        self,
        auc_roc: float,
        precision: float,
        recall: float,
        f1: float,
        ks_statistic: float,
        n_amostras: int,
        modelo_versao: str = "2.0.0",
    ) -> Dict[str, Any]:
        """Registra metricas de performance e verifica degradacao."""
        metricas = {
            "timestamp": datetime.now().isoformat(),
            "auc_roc": round(auc_roc, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "ks_statistic": round(ks_statistic, 6),
            "n_amostras": n_amostras,
            "modelo_versao": modelo_versao,
        }

        # Verifica alertas
        alertas = self._verificar_alertas(metricas)
        metricas["alertas"] = alertas
        metricas["alerta_ativo"] = len(alertas) > 0

        # Salva
        self.historico_metricas.append(metricas)
        arquivo = self.metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H')}.jsonl"
        with open(arquivo, "a") as f:
            f.write(json.dumps(metricas) + "\n")

        if alertas:
            logger.warning("Alertas de degradacao: %s", alertas)

        return metricas

    def _verificar_alertas(self, metricas: Dict[str, Any]) -> List[str]:
        """Verifica se metricas estao abaixo dos thresholds."""
        alertas = []
        if metricas["auc_roc"] < self.thresholds["auc_minimo"]:
            alertas.append(f"AUC-ROC {metricas['auc_roc']:.4f} < {self.thresholds['auc_minimo']}")
        if metricas["precision"] < self.thresholds["precision_minimo"]:
            alertas.append(f"Precision {metricas['precision']:.4f} < {self.thresholds['precision_minimo']}")
        if metricas["recall"] < self.thresholds["recall_minimo"]:
            alertas.append(f"Recall {metricas['recall']:.4f} < {self.thresholds['recall_minimo']}")
        if metricas["f1"] < self.thresholds["f1_minimo"]:
            alertas.append(f"F1 {metricas['f1']:.4f} < {self.thresholds['f1_minimo']}")
        return alertas

    def verificar_degradacao(
        self, window_days: int = 7,
    ) -> Dict[str, Any]:
        """Verifica se houve degradacao de metricas na janela."""
        if not self.historico_metricas:
            return {"degradacao": False, "mensagem": "Sem historico de metricas"}

        cutoff = (datetime.now() - timedelta(days=window_days)).isoformat()
        metricas_janela = [
            m for m in self.historico_metricas
            if m.get("timestamp", "") >= cutoff
        ]

        if len(metricas_janela) < 2:
            return {"degradacao": False, "mensagem": "Dados insuficientes na janela"}

        # Compara primeira e ultima metrica da janela
        primeira = metricas_janela[0]
        ultima = metricas_janela[-1]

        degradacoes = {}
        for metrica in ["auc_roc", "precision", "recall", "f1", "ks_statistic"]:
            base = primeira.get(metrica, 0)
            atual = ultima.get(metrica, 0)
            if base > 0:
                delta = (atual - base) / base
                degradacoes[metrica] = {
                    "baseline": base,
                    "atual": atual,
                    "delta_percentual": round(delta * 100, 2),
                }

        degradacao_detectada = any(
            d["delta_percentual"] < -10 for d in degradacoes.values()
        )

        return {
            "degradacao": degradacao_detectada,
            "window_days": window_days,
            "n_amostras_janela": len(metricas_janela),
            "metricas": degradacoes,
            "ultima_metrica": ultima,
        }

    def distribuicao_predicoes(self, window_hours: int = 24) -> Dict[str, Any]:
        """Analisa a distribuicao de predicoes na janela."""
        cutoff = (datetime.now() - timedelta(hours=window_hours)).isoformat()
        predicoes = [
            p for p in self.historico_predicoes
            if p.get("timestamp", "") >= cutoff
        ]

        if not predicoes:
            return {"n_predicoes": 0}

        probabilidades = [p["probabilidade"] for p in predicoes]
        niveis = [p["nivel_risco"] for p in predicoes]

        return {
            "n_predicoes": len(predicoes),
            "probabilidade_media": round(float(np.mean(probabilidades)), 4),
            "probabilidade_std": round(float(np.std(probabilidades)), 4),
            "probabilidade_min": round(float(np.min(probabilidades)), 4),
            "probabilidade_max": round(float(np.max(probabilidades)), 4),
            "distribuicao_niveis": {
                nivel: niveis.count(nivel) / len(niveis)
                for nivel in set(niveis)
            },
        }

    def dados_grafana(self) -> Dict[str, Any]:
        """Gera dados formatados para dashboard Grafana."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metricas_historicas": list(self.historico_metricas)[-24:],
            "distribuicao_predicoes": self.distribuicao_predicoes(),
            "degradacao": self.verificar_degradacao(),
        }

    def gerar_relatorio(self) -> Dict[str, Any]:
        """Gera relatorio completo de monitoramento."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_metricas_registradas": len(self.historico_metricas),
            "total_predicoes_registradas": len(self.historico_predicoes),
            "degradacao_7d": self.verificar_degradacao(window_days=7),
            "degradacao_30d": self.verificar_degradacao(window_days=30),
            "distribuicao_predicoes_24h": self.distribuicao_predicoes(window_hours=24),
            "ultima_metrica": self.historico_metricas[-1] if self.historico_metricas else None,
            "thresholds": self.thresholds,
        }


def main():
    parser = argparse.ArgumentParser(description="Monitor de modelo de credito")
    parser.add_argument("--config", default="../ops/config/config.yaml")
    parser.add_argument("--check-decay", action="store_true", help="Verifica degradacao")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--output", help="Path para salvar relatorio")
    args = parser.parse_args()

    monitor = ModelMonitor(metrics_dir=args.metrics_dir)

    if args.check_decay:
        resultado = monitor.verificar_degradacao(args.window_days)
        print(json.dumps(resultado, indent=2, default=str))
    else:
        relatorio = monitor.gerar_relatorio()
        print(json.dumps(relatorio, indent=2, default=str))

    if args.output:
        relatorio = monitor.gerar_relatorio()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(relatorio, f, indent=2, default=str)
        logger.info("Relatorio salvo em %s", args.output)


if __name__ == "__main__":
    main()
