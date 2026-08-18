# Copyright (c) 2026 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitarios do pipeline de feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINES_DIR = Path(__file__).resolve().parents[1] / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from feature_engineering import (
    calcular_features_derivadas,
    executar_pipeline,
    gerar_dados_sinteticos,
    join_features,
)


class TestGeracaoDados:
    def test_gerar_dados_sinteticos(self):
        df = gerar_dados_sinteticos(n_samples=500)
        assert len(df) == 500
        assert "renda_mensal" in df.columns
        assert "inadimplente" in df.columns

    def test_gerar_dados_diferentes_seeds(self):
        df1 = gerar_dados_sinteticos(n_samples=100, seed=42)
        df2 = gerar_dados_sinteticos(n_samples=100, seed=99)
        assert not df1["renda_mensal"].equals(df2["renda_mensal"])


class TestFeaturesDerivadas:
    @pytest.fixture()
    def df_base(self):
        return gerar_dados_sinteticos(n_samples=200)

    def test_saldo_ratio(self, df_base):
        df = calcular_features_derivadas(df_base)
        assert "saldo_ratio" in df.columns
        assert df["saldo_ratio"].notna().all()

    def test_transacao_frequency(self, df_base):
        df = calcular_features_derivadas(df_base)
        assert "transacao_frequency" in df.columns
        assert (df["transacao_frequency"] >= 0).all()

    def test_risk_score(self, df_base):
        df = calcular_features_derivadas(df_base)
        assert "risk_score" in df.columns
        assert (df["risk_score"] >= 0).all()
        assert (df["risk_score"] <= 1).all()

    def test_comprometimento_renda(self, df_base):
        df = calcular_features_derivadas(df_base)
        assert "comprometimento_renda" in df.columns

    def test_parcela_renda_ratio(self, df_base):
        df = calcular_features_derivadas(df_base)
        assert "parcela_renda_ratio" in df.columns

    def test_nao_modifica_df_original(self, df_base):
        colunas_originais = list(df_base.columns)
        calcular_features_derivadas(df_base)
        assert list(df_base.columns) == colunas_originais


class TestJoinFeatures:
    def test_join_contas_transacoes(self):
        df_contas = gerar_dados_sinteticos(n_samples=100)
        df_transacoes = pd.DataFrame({
            "id_cliente": df_contas["id_cliente"].values,
            "total_transacoes_90d": np.random.poisson(10, 100),
            "valor_medio_transacao": np.random.exponential(500, 100),
            "qtd_pix": np.random.poisson(3, 100),
        })
        df = join_features(df_contas, df_transacoes)
        assert "total_transacoes_90d" in df.columns
        assert len(df) == 100

    def test_join_sem_transacoes(self):
        df_contas = gerar_dados_sinteticos(n_samples=50)
        df_transacoes = pd.DataFrame()
        df = join_features(df_contas, df_transacoes)
        assert len(df) == 50
        assert list(df.columns) == list(df_contas.columns)


class TestExecutarPipeline:
    def test_dry_run(self, tmp_path):
        df = executar_pipeline(
            dry_run=True, n_samples=200, repo_path=str(tmp_path),
        )
        assert len(df) == 200
        assert "event_timestamp" in df.columns
        assert "risk_score" in df.columns

    def test_dry_run_salva_parquet(self, tmp_path):
        executar_pipeline(dry_run=True, n_samples=100, repo_path=str(tmp_path))
        features_dir = tmp_path / "data" / "features"
        assert features_dir.exists()
        parquets = list(features_dir.glob("*.parquet"))
        assert len(parquets) >= 1
