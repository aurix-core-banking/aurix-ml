"""
AUREUS ML - Modelo de Risco de Crédito (XGBoost).

Substitui o stub anterior (`CreditScoringModel`) por um modelo de
probabilidade de inadimplência treinado com XGBoost e features do core
bancário: renda, score de bureau (SPC/Receita), comprometimento de renda,
histórico transacional e de atraso, relacionamento com o banco etc.

Compatível com o fluxo de treino unificado em ``train_models.py`` e com o
pipeline de treino com MLflow em ``ops/pipelines/train_credit_pipeline.py``.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# Lista estável de features usadas pelo modelo (ordem fixa para previsão).
FEATURE_COLUMNS: List[str] = [
    "renda_mensal",
    "renda_log",
    "renda_per_capita",
    "idade",
    "idade_squared",
    "escolaridade_numeric",
    "is_married",
    "tempo_conta_meses",
    "saldo_medio_12m",
    "saldo_renda_ratio",
    "comprometimento_renda",
    "score_bureau",
    "numero_operacoes_credito",
    "total_financiado",
    "valor_parcela",
    "parcela_renda_ratio",
    "atrasos_hist",
    "consultas_ultimo_6m",
    "is_clt",
    "is_pj",
    "is_servidor",
    "possui_imovel",
    "possui_veiculo",
    "is_capital",
]


class CreditRiskModel:
    """Modelo de probabilidade de inadimplência com XGBoost."""

    DEFAULT_PARAMS: Dict[str, Any] = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "tree_method": "hist",
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.is_trained = False
        self.metadata: Dict[str, Any] = {
            "algoritmo": "XGBoost",
            "random_state": random_state,
            "features": list(FEATURE_COLUMNS),
        }

    # ------------------------------------------------------------------
    # Engenharia de features
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera as features do core bancário a partir do DataFrame bruto."""
        df = df.copy()

        # Renda
        df["renda_log"] = np.log1p(df["renda_mensal"].fillna(0))
        df["renda_per_capita"] = df["renda_mensal"] / (df["pessoas_residencia"] + 1)

        # Idade
        df["idade_squared"] = df["idade"] ** 2

        # Escolaridade
        df["escolaridade_numeric"] = df["escolaridade"].map({
            "FUNDAMENTAL": 1,
            "MEDIO": 2,
            "SUPERIOR": 3,
            "POS_GRADUACAO": 4,
            "MESTRADO": 5,
            "DOUTORADO": 6,
        }).fillna(0)

        # Estado civil
        df["is_married"] = (df["estado_civil"] == "CASADO").astype(int)

        # Relacionamento
        df["tempo_conta_meses"] = (
            datetime.now() - pd.to_datetime(df["data_abertura"])
        ).dt.days / 30

        # Saldo / renda
        df["saldo_renda_ratio"] = df["saldo_atual"] / (df["renda_mensal"] + 1)

        # Comprometimento da renda (dívidas / renda)
        df["comprometimento_renda"] = df["total_dividas"] / (df["renda_mensal"] + 1)

        # Parcela / renda
        df["parcela_renda_ratio"] = df["valor_parcela"] / (df["renda_mensal"] + 1)

        # Tipo de empregador (one-hot)
        df["is_clt"] = (df["tipo_empregador"] == "CLT").astype(int)
        df["is_pj"] = (df["tipo_empregador"] == "PJ").astype(int)
        df["is_servidor"] = (df["tipo_empregador"] == "SERV_PUBLICO").astype(int)

        # Localização
        capitais = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Brasília"]
        df["is_capital"] = df["cidade"].isin(capitais).astype(int)

        features = df[self.feature_columns]
        return features.fillna(0)

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, target_column: str = "inadimplente") -> "CreditRiskModel":
        """Treina o XGBoost para classificar inadimplência."""
        X = self.prepare_features(df)
        y = df[target_column].astype(int).values

        model = xgb.XGBClassifier(
            **self.params,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X, y)

        self.model = model
        self.is_trained = True
        self.metadata["target_column"] = target_column
        self.metadata["n_amostras_treino"] = int(len(df))
        self.metadata["treinado_em"] = datetime.now().isoformat()
        return self

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade de inadimplência (0..1) por cliente."""
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo ainda não foi treinado!")
        X = self.prepare_features(df)
        return self.model.predict_proba(X)[:, 1]

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Converte a probabilidade em score estilo FICO (300-850)."""
        proba = self.predict_proba(df)
        score = 850 - 550 * proba
        return np.clip(score, 300, 850).astype(int)

    def predict_default(self, df: pd.DataFrame, limiar: float = 0.5) -> np.ndarray:
        """Classificação binária de inadimplência dado o limiar."""
        return (self.predict_proba(df) >= limiar).astype(int)

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> str:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            joblib.dump({"modelo": self.model, "metadata": self.metadata}, f)
        with open(Path(path).with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_model(cls, path: str) -> "CreditRiskModel":
        with open(path, "rb") as f:
            dados = joblib.load(f)
        instancia = cls()
        instancia.model = dados["modelo"]
        instancia.metadata = dados.get("metadata", {})
        instancia.feature_columns = instancia.metadata.get("features", FEATURE_COLUMNS)
        instancia.is_trained = instancia.model is not None
        return instancia

    def feature_importances(self) -> Dict[str, float]:
        """Importância das features (gain) como dict ordenado."""
        if self.model is None:
            return {}
        importancias = self.model.feature_importances_
        return dict(
            sorted(
                zip(self.feature_columns, importancias),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )


# --------------------------------------------------------------------------
# Geração de dados de exemplo com sinal aprendível (AUC alto)
# --------------------------------------------------------------------------
def generate_credit_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Gera amostra sintética de clientes com inadimplência dependente das features.

    A probabilidade de inadimplência é função de uma credibilidade latente,
    para que o XGBoost aprenda padrões reais (AUC >> 0.5).
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    renda = np.round(np.maximum(rng.exponential(6000, n), 500), 2)
    idade = rng.integers(18, 80, n)
    escolaridade = rng.choice(
        ["FUNDAMENTAL", "MEDIO", "SUPERIOR", "POS_GRADUACAO", "MESTRADO"],
        n, p=[0.08, 0.32, 0.45, 0.12, 0.03],
    )
    estado_civil = rng.choice(["SOLTEIRO", "CASADO", "DIVORCIADO", "VIUVO"], n, p=[0.4, 0.4, 0.15, 0.05])
    pessoas_residencia = rng.integers(1, 7, n)
    tipo_empregador = rng.choice(["CLT", "PJ", "SERV_PUBLICO", "DESEMPREGADO"], n, p=[0.5, 0.25, 0.15, 0.1])
    cidades = ["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador", "Brasília", "Campinas", "Porto Alegre"]
    cidade = rng.choice(cidades, n)

    data_abertura = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 2200, n), unit="D")

    # Score de bureau: fortemente relacionado à credibilidade
    score_bureau = np.clip(rng.normal(600, 90, n), 0, 1000).astype(int)

    # Histórico de atraso (quanto maior, pior)
    atrasos_hist = np.clip(
        rng.poisson(np.maximum(2.0 - score_bureau / 400.0, 0.1), n), 0, 30
    ).astype(int)

    # Nº de consultas recentes ao bureau
    consultas = np.clip(
        rng.poisson(np.maximum(3.0 - score_bureau / 350.0, 0.1), n), 0, 40
    ).astype(int)

    total_dividas = np.round(np.maximum(rng.exponential(18000, n) * (1 + atrasos_hist / 20.0), 0), 2)
    total_financiado = np.round(np.maximum(rng.exponential(40000, n), 0), 2)
    valor_parcela = np.round(np.maximum(total_financiado * rng.uniform(0.02, 0.05, n), 0), 2)

    saldo_medio = np.round(np.maximum(rng.exponential(12000, n), 0), 2)
    saldo_atual = np.round(np.maximum(saldo_medio * rng.uniform(0.4, 1.3, n), 0), 2)
    numero_operacoes_credito = rng.integers(0, 8, n)

    possui_imovel = rng.binomial(1, 0.35 + score_bureau / 2000.0, n)
    possui_veiculo = rng.binomial(1, 0.4 + score_bureau / 3000.0, n)

    comprometimento = total_dividas / np.maximum(renda, 1.0)
    mapa_escolaridade = np.array(
        [{"FUNDAMENTAL": -0.6, "MEDIO": 0.0, "SUPERIOR": 0.6,
          "POS_GRADUACAO": 1.2, "MESTRADO": 1.8}.get(e, 0.0) for e in escolaridade]
    )
    mapa_empregador = np.array(
        [{"CLT": 0.4, "PJ": 0.5, "SERV_PUBLICO": 1.0, "DESEMPREGADO": -1.5}.get(e, 0.0)
         for e in tipo_empregador]
    )

    # Credibilidade latente — quanto maior, menor a chance de inadimplência.
    # p_default = sigmoid(-credibilidade), recentrada para ~12-16% de default.
    credibilidade = (
        + score_bureau / 180.0
        - atrasos_hist * 0.4
        - consultas * 0.15
        - comprometimento / 60.0
        + mapa_escolaridade
        + mapa_empregador
        + np.log1p(renda) / 8.0
        + possui_imovel * 0.35
        + possui_veiculo * 0.25
        - 2.3
        + rng.normal(0, 0.8, n)
    )
    p_default = 1 / (1 + np.exp(credibilidade))
    inadimplente = rng.binomial(1, p_default, n)

    return pd.DataFrame({
        "id_cliente": np.arange(1, n + 1),
        "renda_mensal": renda,
        "idade": idade,
        "pessoas_residencia": pessoas_residencia,
        "escolaridade": escolaridade,
        "estado_civil": estado_civil,
        "tipo_empregador": tipo_empregador,
        "cidade": cidade,
        "data_abertura": data_abertura,
        "score_bureau": score_bureau,
        "atrasos_hist": atrasos_hist,
        "consultas_ultimo_6m": consultas,
        "total_dividas": total_dividas,
        "total_financiado": total_financiado,
        "valor_parcela": valor_parcela,
        "saldo_medio_12m": saldo_medio,
        "saldo_atual": saldo_atual,
        "numero_operacoes_credito": numero_operacoes_credito,
        "possui_imovel": possui_imovel,
        "possui_veiculo": possui_veiculo,
        "inadimplente": inadimplente,
    })


# Compatibilidade com o stub anterior: o modelo legado de scoring continua
# disponível por este nome (RandomForest), enquanto o real é ``CreditRiskModel``.
class CreditScoringModel(CreditRiskModel):
    """Alias do modelo de risco para compatibilidade com código legado."""


__all__ = ["CreditRiskModel", "CreditScoringModel", "generate_credit_data", "FEATURE_COLUMNS"]
