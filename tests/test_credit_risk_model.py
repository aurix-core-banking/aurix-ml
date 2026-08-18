# Copyright (c) 2026 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitarios do modelo de risco de credito v2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
sys.path.insert(0, str(MODELS_DIR))

from credit_risk_model_v2 import (
    ALL_FEATURES,
    CreditRiskModelV2,
    generate_credit_data_v2,
)


class TestCreditRiskModelV2:
    """Testes do CreditRiskModelV2."""

    @pytest.fixture()
    def df_dados(self):
        return generate_credit_data_v2(n_samples=1000, seed=42)

    @pytest.fixture()
    def modelo_treinado(self, df_dados):
        model = CreditRiskModelV2(random_state=42)
        model.train(df_dados, target_column="inadimplente")
        return model

    def test_geracao_dados_sinteticos(self, df_dados):
        assert len(df_dados) == 1000
        assert "inadimplente" in df_dados.columns
        assert "id_cliente" in df_dados.columns
        assert df_dados["inadimplente"].mean() > 0.05
        assert df_dados["inadimplente"].mean() < 0.5

    def test_todas_features_presentes(self, df_dados):
        model = CreditRiskModelV2()
        X = model.prepare_features(df_dados)
        assert list(X.columns) == ALL_FEATURES

    def test_features_derivadas(self, df_dados):
        model = CreditRiskModelV2()
        X = model.prepare_features(df_dados)
        assert "saldo_ratio" in X.columns
        assert "transacao_frequency" in X.columns
        assert "risk_score" in X.columns
        assert "comprometimento_renda" in X.columns
        assert "parcela_renda_ratio" in X.columns

    def test_treino_retorna_metricas(self, modelo_treinado, df_dados):
        metricas = modelo_treinado.train(df_dados)
        assert "auc_roc" in metricas
        assert "ks_statistic" in metricas
        assert "precision" in metricas
        assert "recall" in metricas
        assert "f1" in metricas

    def test_modelo_treinado_prediz(self, modelo_treinado, df_dados):
        proba = modelo_treinado.predict_proba(df_dados)
        assert len(proba) == len(df_dados)
        assert all(0 <= p <= 1 for p in proba)

    def test_score_fico(self, modelo_treinado, df_dados):
        scores = modelo_treinado.predict_score(df_dados)
        assert all(300 <= s <= 850 for s in scores)

    def test_risk_levels(self, modelo_treinado, df_dados):
        niveis = modelo_treinado.predict_risk_level(df_dados)
        assert all(n in ("A", "B", "C", "D") for n in niveis)
        assert len(niveis) == len(df_dados)

    def test_predict_with_explanation(self, modelo_treinado, df_dados):
        proba, niveis, shap_vals = modelo_treinado.predict_with_explanation(df_dados.head(10))
        assert len(proba) == 10
        assert len(niveis) == 10
        assert shap_vals is not None

    def test_explain(self, modelo_treinado, df_dados):
        explicacao = modelo_treinado.explain(df_dados.head(10))
        assert explicacao is not None
        assert len(explicacao) > 0
        assert all(isinstance(v, float) for v in explicacao.values())

    def test_feature_importances(self, modelo_treinado):
        importancias = modelo_treinado.feature_importances()
        assert len(importancias) == len(ALL_FEATURES)
        assert all(0 <= v <= 1 for v in importancias.values())

    def test_salvar_e_carregar(self, modelo_treinado, tmp_path):
        path = str(tmp_path / "modelo_teste.pkl")
        modelo_treinado.save_model(path)
        assert Path(path).exists()
        assert Path(path).with_suffix(".json").exists()

        loaded = CreditRiskModelV2.load_model(path)
        assert loaded.is_trained
        assert loaded.metadata["versao"] == "2.0.0"

        # Previsao identica
        df = generate_credit_data_v2(n_samples=5, seed=99)
        proba_orig = modelo_treinado.predict_proba(df)
        proba_loaded = loaded.predict_proba(df)
        np.testing.assert_array_almost_equal(proba_orig, proba_loaded, decimal=5)

    def test_sem_treino_erro(self):
        model = CreditRiskModelV2()
        with pytest.raises(ValueError, match="nao foi treinado"):
            model.predict_proba(pd.DataFrame({"x": [1]}))

    def test_auc_alto_dados_sinteticos(self, df_dados):
        model = CreditRiskModelV2(random_state=42)
        metricas = model.train(df_dados, test_size=0.2)
        assert metricas["auc_roc"] > 0.7

    def test_ks_positivo(self, modelo_treinado, df_dados):
        metricas = modelo_treinado.train(df_dados, test_size=0.2)
        assert metricas["ks_statistic"] > 0.1
