# Copyright (c) 2026 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitários do monitoramento de modelos (drift, alertas, MLflow)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

OPS_DIR = Path(__file__).resolve().parents[1] / "ops"
sys.path.insert(0, str(OPS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from monitoring.drift_detection import _compute_psi, compute_reference_stats, detect_drift
from monitoring.model_drift import detect_model_drift
from monitoring.alerting import avaliar_e_alertar


class TestPSI:
    def test_psi_identico_é_zero(self):
        probs = np.array([0.2, 0.3, 0.5])
        assert _compute_psi(probs, probs) == pytest.approx(0.0, abs=1e-6)

    def test_psi_distribuições_diferentes_é_positivo(self):
        esp = np.array([0.1, 0.4, 0.5])
        atu = np.array([0.5, 0.4, 0.1])
        assert _compute_psi(esp, atu) > 0.5

    def test_psi_trata_zeros_com_epsilon(self):
        esp = np.array([1.0, 0.0])
        atu = np.array([0.9, 0.1])
        assert _compute_psi(esp, atu) > 0


class TestDataDrift:
    @pytest.fixture()
    def dados_referencia(self):
        return generate_sample_data(2000)

    def test_reference_stats_contem_features(self, dados_referencia):
        stats = compute_reference_stats(dados_referencia)
        assert "features" in stats
        assert "valor" in stats["features"]
        assert "tipo_transacao" in stats["features"]

    def test_detect_drift_mesmos_dados_sem_drift(self, dados_referencia):
        stats = compute_reference_stats(dados_referencia)
        report = detect_drift(stats, dados_referencia.head(500))
        assert report["drift_detected"] is False
        assert report["overall_score"] < 0.15

    def test_detect_drift_shift_radical_detecta(self):
        base = generate_sample_data(2000)
        stats = compute_reference_stats(base)
        shifted = base.copy()
        shifted["valor"] = base["valor"] * 50 + 1e6
        report = detect_drift(stats, shifted)
        assert report["drift_detected"] is True


def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    from fraud_detection_model import generate_sample_data as gen

    return gen(n_samples=n_samples)


class TestModelDrift:
    def test_modelo_sem_degradacao_sem_drift(self):
        baseline = {"auc": 0.9, "precisao": 0.8, "recall": 0.7, "f1": 0.74, "acuracia": 0.85}
        current = {"auc": 0.88, "precisao": 0.78, "recall": 0.68, "f1": 0.72, "acuracia": 0.84}
        report = detect_model_drift(baseline, current, threshold=0.1)
        assert report["drift_detected"] is False

    def test_modelo_degradado_detecta_drift(self):
        baseline = {"auc": 0.9, "precisao": 0.8, "recall": 0.7, "f1": 0.74, "acuracia": 0.85}
        current = {"auc": 0.5, "precisao": 0.4, "recall": 0.3, "f1": 0.34, "acuracia": 0.6}
        report = detect_model_drift(baseline, current, threshold=0.1)
        assert report["drift_detected"] is True
        assert report["overall_degradation"] > 0.2

    def test_degradacao_saturada_em_1(self):
        baseline = {"auc": 0.8, "precisao": 0.8, "recall": 0.7, "f1": 0.74, "acuracia": 0.85}
        current = {"auc": 0.0, "precisao": 0.0, "recall": 0.0, "f1": 0.0, "acuracia": 0.0}
        report = detect_model_drift(baseline, current, threshold=0.1)
        assert report["overall_degradation"] == pytest.approx(1.0)


class TestAlerting:
    def test_sem_drift_não_alerta(self, monkeypatch):
        chamado = {"slack": False, "email": False}

        def _fake_slack(texto):
            chamado["slack"] = True
            return True

        def _fake_email(assunto, corpo):
            chamado["email"] = True
            return True

        monkeypatch.setattr("monitoring.alerting.notificar_slack", _fake_slack)
        monkeypatch.setattr("monitoring.alerting.notificar_email", _fake_email)
        resultado = avaliar_e_alertar({"drift_detected": False}, {"drift_detected": False})
        assert resultado["alerted"] is False
        assert not chamado["slack"]
        assert not chamado["email"]

    def test_com_data_drift_alerta(self, monkeypatch):
        chamado = {"slack": False, "email": False}

        def _fake_slack(texto):
            chamado["slack"] = True
            assert "Data drift" in texto
            return True

        def _fake_email(assunto, corpo):
            chamado["email"] = True
            return True

        monkeypatch.setattr("monitoring.alerting.notificar_slack", _fake_slack)
        monkeypatch.setattr("monitoring.alerting.notificar_email", _fake_email)
        resultado = avaliar_e_alertar(
            {"drift_detected": True, "overall_score": 0.3, "threshold": 0.15, "features": {}},
            {"drift_detected": False},
        )
        assert resultado["alerted"] is True
        assert chamado["slack"]

    def test_com_model_drift_alerta(self, monkeypatch):
        chamado = {"slack": False}

        def _fake_slack(texto):
            chamado["slack"] = True
            assert "Model drift" in texto
            return True

        monkeypatch.setattr("monitoring.alerting.notificar_slack", _fake_slack)
        monkeypatch.setattr("monitoring.alerting.notificar_email", lambda a, b: True)
        resultado = avaliar_e_alertar(
            {"drift_detected": False},
            {"drift_detected": True, "overall_degradation": 0.4, "threshold": 0.1, "metricas": {}},
        )
        assert resultado["alerted"] is True
        assert chamado["slack"]


class TestPrometheusExporter:
    def test_carrega_relatorios_inexistentes(self):
        from monitoring.prometheus_exporter import carregar_relatorios

        tmp = Path("/tmp/opencode/relatorios_inexistentes")
        tmp.mkdir(parents=True, exist_ok=True)
        dados = carregar_relatorios(tmp)
        assert dados == {}

    def test_exporta_sem_erro(self, tmp_path, monkeypatch):
        from monitoring import prometheus_exporter

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "drift_report.json").write_text(
            json.dumps({"overall_score": 0.3, "drift_detected": True, "features": {}})
        )
        (artifacts / "model_drift_report.json").write_text(
            json.dumps({"overall_degradation": 0.2, "drift_detected": False, "metricas": {}})
        )
        prometheus_exporter.exportar_para_prometheus(tmp_path)
        prom = artifacts / "aurix_ml_drift.prom"
        assert prom.exists()
        conteudo = prom.read_text()
        assert "aurix_ml_data_drift_score" in conteudo


class TestRetraining:
    def test_ha_drift_false_sem_relatorios(self):
        from monitoring.retraining import ler_relatorios, ha_drift

        tmp = Path("/tmp/opencode/relatorios_vazios")
        tmp.mkdir(parents=True, exist_ok=True)
        assert ha_drift(ler_relatorios(tmp)) is False

    def test_ha_drift_true_quando_modelo_degradado(self, tmp_path):
        from monitoring.retraining import ler_relatorios, ha_drift

        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "drift_report.json").write_text(json.dumps({"drift_detected": False}))
        (artifacts / "model_drift_report.json").write_text(json.dumps({"drift_detected": True}))
        assert ha_drift(ler_relatorios(tmp_path)) is True

    def test_disparar_retreino_executa_pipeline(self, tmp_path, monkeypatch):
        from monitoring.retraining import disparar_retreino

        chamado = {"cmd": None}

        def _fake_subprocess(cmd, **kwargs):
            chamado["cmd"] = cmd
            return type("Proc", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()

        monkeypatch.setattr("monitoring.retraining.subprocess.run", _fake_subprocess)
        resultado = disparar_retreino(
            tmp_path, {"training": {"model_name": "fraude"}, "mlflow": {}}, use_mlflow=False
        )
        assert resultado["retrain_triggered"] is True
        assert resultado["treino_ok"] is True
        assert "-m" in chamado["cmd"]
        assert "pipelines.train_pipeline" in chamado["cmd"]

    def test_main_force_dispara_mesmo_sem_drift(self, monkeypatch):
        from monitoring import retraining

        retraining.ler_relatorios = lambda base: {}
        retraining.ha_drift = lambda r: False
        retraining.disparar_retreino = lambda base, config, use_mlflow: {"retrain_triggered": True}
        monkeypatch.setattr(sys, "argv", ["retraining", "--force"])
        resultado = retraining.main()
        assert resultado["retrain_triggered"] is True
