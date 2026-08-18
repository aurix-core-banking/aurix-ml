# Copyright (c) 2026 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitarios do detector de drift."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MONITORING_DIR = Path(__file__).resolve().parents[1] / "monitoring"
sys.path.insert(0, str(MONITORING_DIR))

from drift_detector import DriftDetector, gerar_dados_drift


class TestDriftDetector:
    """Testes do DriftDetector."""

    @pytest.fixture()
    def detector(self):
        return DriftDetector()

    @pytest.fixture()
    def dados_ref(self):
        return gerar_dados_drift(n_samples=2000, shift=0.0, seed=42)

    @pytest.fixture()
    def dados_atual_sem_drift(self):
        return gerar_dados_drift(n_samples=2000, shift=0.0, seed=99)

    @pytest.fixture()
    def dados_com_drift(self):
        return gerar_dados_drift(n_samples=2000, shift=3.0, seed=42)

    # ------------------------------------------------------------------
    # PSI
    # ------------------------------------------------------------------
    def test_psi_distribuicao_identica(self, detector):
        dados = np.random.normal(0, 1, 5000)
        psi = detector.calcular_psi(dados, dados)
        assert psi == pytest.approx(0.0, abs=0.01)

    def test_psi_distribuicoes_diferentes(self, detector):
        ref = np.random.normal(0, 1, 5000)
        cur = np.random.normal(3, 1, 5000)
        psi = detector.calcular_psi(ref, cur)
        assert psi > 0.5

    def test_psi_com_nan(self, detector):
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0] * 200)
        cur = np.array([1.1, 2.1, 3.1, 3.1, 4.1] * 200)
        psi = detector.calcular_psi(ref, cur)
        assert isinstance(psi, float)

    def test_psi_poucos_dados_retorna_zero(self, detector):
        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])
        psi = detector.calcular_psi(ref, cur)
        assert psi == 0.0

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------
    def test_ks_distribuicoes_iguais(self, detector):
        dados = np.random.normal(0, 1, 1000)
        resultado = detector.teste_ks(dados, dados)
        assert resultado["p_value"] > 0.05

    def test_ks_distribuicoes_diferentes(self, detector):
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(5, 1, 1000)
        resultado = detector.teste_ks(ref, cur)
        assert resultado["p_value"] < 0.01
        assert resultado["statistic"] > 0.3

    def test_ks_poucos_dados(self, detector):
        resultado = detector.teste_ks(np.array([1.0]), np.array([2.0]))
        assert resultado["statistic"] == 0.0
        assert resultado["p_value"] == 1.0

    # ------------------------------------------------------------------
    # Chi-squared
    # ------------------------------------------------------------------
    def test_chi2_distribuicoes_iguais(self, detector):
        cats = ["A", "B", "C", "D"]
        ref = pd.Series(np.random.choice(cats, 1000))
        cur = pd.Series(np.random.choice(cats, 1000))
        resultado = detector.teste_chi2(ref, cur)
        assert resultado["p_value"] > 0.05
        assert resultado["drift_detectado"] is False

    def test_chi2_distribuicoes_diferentes(self, detector):
        ref = pd.Series(["A"] * 900 + ["B"] * 100)
        cur = pd.Series(["A"] * 100 + ["B"] * 900)
        resultado = detector.teste_chi2(ref, cur)
        assert resultado["drift_detectado"] is True

    # ------------------------------------------------------------------
    # Feature drift
    # ------------------------------------------------------------------
    def test_feature_drift_sem_drift(self, detector, dados_ref):
        dados_cur = gerar_dados_drift(n_samples=2000, shift=0.0, seed=99)
        relatorio = detector.detectar_feature_drift(dados_ref, dados_cur)
        assert relatorio["drift_detectado"] is False

    def test_feature_drift_com_drift(self, detector, dados_ref, dados_com_drift):
        relatorio = detector.detectar_feature_drift(dados_ref, dados_com_drift)
        assert relatorio["drift_detectado"] is True
        assert relatorio["nivel"] in ("alerta", "critico")

    def test_feature_drift_psi_geral(self, detector, dados_ref, dados_com_drift):
        relatorio = detector.detectar_feature_drift(dados_ref, dados_com_drift)
        assert relatorio["psi_geral"] > 0.0

    # ------------------------------------------------------------------
    # Prediction drift
    # ------------------------------------------------------------------
    def test_prediction_drift_sem_drift(self, detector):
        pred_ref = np.random.beta(2, 5, 1000)
        pred_cur = np.random.beta(2, 5, 1000)
        relatorio = detector.detectar_prediction_drift(pred_ref, pred_cur)
        assert relatorio["drift_detectado"] is False

    def test_prediction_drift_com_drift(self, detector):
        pred_ref = np.random.beta(2, 5, 1000)
        pred_cur = np.random.beta(5, 2, 1000)
        relatorio = detector.detectar_prediction_drift(pred_ref, pred_cur)
        assert relatorio["drift_detectado"] is True

    # ------------------------------------------------------------------
    # Concept drift
    # ------------------------------------------------------------------
    def test_concept_drift_sem_drift(self, detector):
        y_ref = np.random.binomial(1, 0.15, 1000)
        y_cur = np.random.binomial(1, 0.15, 1000)
        pred_ref = np.random.beta(2, 5, 1000)
        pred_cur = np.random.beta(2, 5, 1000)
        relatorio = detector.detectar_concept_drift(y_ref, y_cur, pred_ref, pred_cur)
        assert relatorio["drift_detectado"] is False

    # ------------------------------------------------------------------
    # Relatorio completo
    # ------------------------------------------------------------------
    def test_relatorio_completo(self, detector, dados_ref, dados_com_drift):
        relatorio = detector.gerar_relatorio_completo(dados_ref, dados_com_drift)
        assert "feature_drift" in relatorio
        assert "nivel_geral" in relatorio
        assert relatorio["nivel_geral"] in ("ok", "atencao", "alerta", "critico")

    def test_relatorio_com_prediction_drift(self, detector, dados_ref, dados_com_drift):
        pred_ref = np.random.beta(2, 5, 2000)
        pred_cur = np.random.beta(5, 2, 2000)
        relatorio = detector.gerar_relatorio_completo(
            dados_ref, dados_com_drift, pred_ref, pred_cur,
        )
        assert "prediction_drift" in relatorio

    def test_gerar_dados_drift(self):
        df = gerar_dados_drift(n_samples=500, shift=2.0, seed=42)
        assert len(df) == 500
        assert "renda_mensal" in df.columns
        assert "score_bureau" in df.columns
