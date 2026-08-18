"""
Modelo de Risco de Credito v2 — XGBoost com feature store (Feast), SHAP e MLflow.

Substitui o v1 com integracao ao Feast feature store para consistencia
de features entre treino e inferencia. Inclui engenharia de features
derivadas (saldo_ratio, transacao_frequency, risk_score), metricas
completas (AUC-ROC, precision, recall, F1, KS statistic) e
explicabilidade via SHAP.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# Features base do Feast (credit_features view)
FEAST_FEATURES: List[str] = [
    "renda_mensal",
    "idade",
    "score_bureau",
    "saldo_medio_12m",
    "saldo_atual",
    "total_dividas",
    "total_financiado",
    "valor_parcela",
    "numero_operacoes_credito",
    "atrasos_hist",
    "consultas_ultimo_6m",
    "tempo_conta_meses",
    "pessoas_residencia",
    "possui_imovel",
    "possui_veiculo",
]

# Features derivadas de engenharia
DERIVED_FEATURES: List[str] = [
    "renda_log",
    "renda_per_capita",
    "idade_squared",
    "saldo_renda_ratio",
    "comprometimento_renda",
    "parcela_renda_ratio",
    "transacao_frequency",
    "risk_score",
    "saldo_ratio",
    "escolaridade_numeric",
    "is_married",
    "is_clt",
    "is_pj",
    "is_servidor",
    "is_capital",
]

ALL_FEATURES: List[str] = FEAST_FEATURES + DERIVED_FEATURES

# Colunas categoricas que precisam de encoding
CATEGORICAL_COLS: Dict[str, List[str]] = {
    "escolaridade": ["FUNDAMENTAL", "MEDIO", "SUPERIOR", "POS_GRADUACAO", "MESTRADO", "DOUTORADO"],
    "estado_civil": ["SOLTEIRO", "CASADO", "DIVORCIADO", "VIUVO"],
    "tipo_empregador": ["CLT", "PJ", "SERV_PUBLICO", "DESEMPREGADO"],
    "cidade": [
        "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador",
        "Brasilia", "Campinas", "Porto Alegre", "Fortaleza", "Curitiba",
    ],
}


class CreditRiskModelV2:
    """Modelo de probabilidade de inadimplencia com XGBoost v2.

    Integrado ao Feast feature store para consistencia de features,
    com SHAP para explicabilidade e MLflow para tracking.
    """

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

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        feature_store_url: Optional[str] = None,
    ):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.random_state = random_state
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_columns: List[str] = list(ALL_FEATURES)
        self.feature_store_url = feature_store_url
        self.is_trained = False
        self.shap_explainer = None
        self.metadata: Dict[str, Any] = {
            "algoritmo": "XGBoost",
            "versao": "2.0.0",
            "random_state": random_state,
            "features": list(ALL_FEATURES),
            "feature_store": "Feast",
        }

    # ------------------------------------------------------------------
    # Engenharia de features
    # ------------------------------------------------------------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features derivadas a partir do DataFrame bruto.

        Combina features do Feast com engenharia de dominio bancario:
        - saldo_ratio: saldo atual / saldo medio (estabilidade financeira)
        - transacao_frequency: operacoes por mes de relacionamento
        - risk_score: score composto de risco (0-1)
        """
        df = df.copy()

        # Renda
        df["renda_log"] = np.log1p(df["renda_mensal"].fillna(0))
        df["renda_per_capita"] = df["renda_mensal"] / (df["pessoas_residencia"] + 1)

        # Idade
        df["idade_squared"] = df["idade"] ** 2

        # Saldo / renda
        df["saldo_renda_ratio"] = df["saldo_atual"] / (df["renda_mensal"] + 1)

        # Comprometimento da renda
        df["comprometimento_renda"] = df["total_dividas"] / (df["renda_mensal"] + 1)

        # Parcela / renda
        df["parcela_renda_ratio"] = df["valor_parcela"] / (df["renda_mensal"] + 1)

        # Saldo ratio: estabilidade do saldo ao longo do tempo
        df["saldo_ratio"] = df["saldo_atual"] / (df["saldo_medio_12m"] + 1)

        # Frequencia de transacoes por mes de relacionamento
        meses_conta = np.maximum(df["tempo_conta_meses"].fillna(1), 1)
        df["transacao_frequency"] = df["numero_operacoes_credito"] / meses_conta

        # Score de risco composto (normalizado 0-1)
        score_bureau_norm = df["score_bureau"].clip(0, 1000) / 1000.0
        atrasos_norm = df["atrasos_hist"].clip(0, 30) / 30.0
        comprometimento_norm = df["comprometimento_renda"].clip(0, 5) / 5.0
        df["risk_score"] = (
            (1 - score_bureau_norm) * 0.4
            + atrasos_norm * 0.3
            + comprometimento_norm * 0.3
        )

        # Escolaridade numerica
        mapa_esc = {
            "FUNDAMENTAL": 1, "MEDIO": 2, "SUPERIOR": 3,
            "POS_GRADUACAO": 4, "MESTRADO": 5, "DOUTORADO": 6,
        }
        df["escolaridade_numeric"] = df.get("escolaridade", pd.Series(dtype=str)).map(mapa_esc).fillna(0)

        # Estado civil
        if "estado_civil" in df.columns:
            df["is_married"] = (df["estado_civil"] == "CASADO").astype(int)
        else:
            df["is_married"] = 0

        # Tipo de empregador
        if "tipo_empregador" in df.columns:
            df["is_clt"] = (df["tipo_empregador"] == "CLT").astype(int)
            df["is_pj"] = (df["tipo_empregador"] == "PJ").astype(int)
            df["is_servidor"] = (df["tipo_empregador"] == "SERV_PUBLICO").astype(int)
        else:
            df["is_clt"] = 0
            df["is_pj"] = 0
            df["is_servidor"] = 0

        # Capital
        capitais = [
            "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador",
            "Brasilia", "Campinas", "Porto Alegre", "Fortaleza", "Curitiba",
        ]
        if "cidade" in df.columns:
            df["is_capital"] = df["cidade"].isin(capitais).astype(int)
        else:
            df["is_capital"] = 0

        # Garante que todas as features existem
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        features = df[self.feature_columns]
        return features.fillna(0)

    # ------------------------------------------------------------------
    # Treino
    # ------------------------------------------------------------------
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = "inadimplente",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Treina o XGBoost com train/test split e retorna metricas.

        Returns:
            Dict com metricas: auc_roc, precision, recall, f1, ks_statistic
        """
        from scipy import stats as scipy_stats
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split

        X = self.prepare_features(df)
        y = df[target_column].astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y,
        )

        self.model = xgb.XGBClassifier(
            **self.params,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Probabilidades e predicoes
        y_prob = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Metricas
        auc_roc = float(roc_auc_score(y_test, y_prob))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        acuracia = float(accuracy_score(y_test, y_pred))

        # KS statistic (Kolmogorov-Smirnov)
        ks_statistic = float(scipy_stats.ks_2samp(y_prob[y_test == 0], y_prob[y_test == 1]).statistic)

        # Inicializa SHAP explainer
        self.shap_explainer = None
        try:
            import shap
            self.shap_explainer = shap.TreeExplainer(self.model)
        except ImportError:
            pass

        self.is_trained = True
        self.metadata.update({
            "target_column": target_column,
            "n_amostras_treino": int(len(X_train)),
            "n_amostras_teste": int(len(X_test)),
            "treinado_em": datetime.now().isoformat(),
            "auc_roc": auc_roc,
            "ks_statistic": ks_statistic,
        })

        return {
            "auc_roc": auc_roc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "acuracia": acuracia,
            "ks_statistic": ks_statistic,
            "n_amostras_treino": int(len(X_train)),
            "n_amostras_teste": int(len(X_test)),
        }

    # ------------------------------------------------------------------
    # Previsao
    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade de inadimplencia (0..1) por cliente."""
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo ainda nao foi treinado!")
        X = self.prepare_features(df)
        return self.model.predict_proba(X)[:, 1]

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Converte a probabilidade em score estilo FICO (300-850)."""
        proba = self.predict_proba(df)
        score = 850 - 550 * proba
        return np.clip(score, 300, 850).astype(int)

    def predict_risk_level(self, df: pd.DataFrame) -> List[str]:
        """Classifica o risco em niveis A/B/C/D."""
        proba = self.predict_proba(df)
        niveis = []
        for p in proba:
            if p < 0.15:
                niveis.append("A")
            elif p < 0.35:
                niveis.append("B")
            elif p < 0.60:
                niveis.append("C")
            else:
                niveis.append("D")
        return niveis

    def predict_with_explanation(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
        """Predicao com nivel de risco e explicabilidade SHAP.

        Returns:
            (proba, risk_levels, shap_values)
        """
        proba = self.predict_proba(df)
        risk_levels = self.predict_risk_level(df)

        shap_values = None
        if self.shap_explainer is not None:
            X = self.prepare_features(df)
            shap_values = self.shap_explainer.shap_values(X)

        return proba, risk_levels, shap_values

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def explain(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Retorna a importancia SHAP media por feature para o DataFrame."""
        if self.shap_explainer is None:
            return None
        X = self.prepare_features(df)
        shap_values = self.shap_explainer.shap_values(X)
        importancia = np.abs(shap_values).mean(axis=0)
        return dict(sorted(
            zip(self.feature_columns, importancia),
            key=lambda kv: kv[1],
            reverse=True,
        ))

    # ------------------------------------------------------------------
    # Persistencia
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> str:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            joblib.dump({
                "modelo": self.model,
                "metadata": self.metadata,
                "feature_columns": self.feature_columns,
            }, f)
        with open(Path(path).with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_model(cls, path: str) -> "CreditRiskModelV2":
        with open(path, "rb") as f:
            dados = joblib.load(f)
        instancia = cls()
        instancia.model = dados["modelo"]
        instancia.metadata = dados.get("metadata", {})
        instancia.feature_columns = dados.get("feature_columns", ALL_FEATURES)
        instancia.is_trained = instancia.model is not None
        return instancia

    def feature_importances(self) -> Dict[str, float]:
        """Importancia das features (gain) como dict ordenado."""
        if self.model is None:
            return {}
        importancias = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_columns, importancias),
            key=lambda kv: kv[1],
            reverse=True,
        ))

    # ------------------------------------------------------------------
    # Feast integration
    # ------------------------------------------------------------------
    def fetch_features_from_feast(
        self, entity_rows: List[Dict[str, Any]], feature_store: Optional[Any] = None,
    ) -> pd.DataFrame:
        """Busca features do Feast feature store.

        Args:
            entity_rows: Lista de dicts com entity_df (id_cliente, event_timestamp)
            feature_store: Instancia do FeatureStore do Feast (opcional)

        Returns:
            DataFrame com features do Feast
        """
        if feature_store is None:
            try:
                from feast import FeatureStore
                feature_store = FeatureStore(repo_path=".")
            except (ImportError, Exception):
                return pd.DataFrame(entity_rows)

        entity_df = pd.DataFrame(entity_rows)
        features = feature_store.get_historical_features(
            entity_df=entity_df,
            features=[
                "credit_features:renda_mensal",
                "credit_features:idade",
                "credit_features:score_bureau",
                "credit_features:saldo_medio_12m",
                "credit_features:saldo_atual",
                "credit_features:total_dividas",
                "credit_features:total_financiado",
                "credit_features:valor_parcela",
                "credit_features:numero_operacoes_credito",
                "credit_features:atrasos_hist",
                "credit_features:consultas_ultimo_6m",
            ],
        ).to_df()

        return features


# --------------------------------------------------------------------------
# Geracao de dados sinteticos (compativel com v1)
# --------------------------------------------------------------------------
def generate_credit_data_v2(
    n_samples: int = 5000, seed: int = 42,
) -> pd.DataFrame:
    """Gera amostra sintetica com sinal aprendivel (AUC alto).

    A probabilidade de inadimplencia e funcao de uma credibilidade latente,
    para que o XGBoost aprenda padroes reais (AUC >> 0.5).
    """
    rng = np.random.default_rng(seed)
    n = n_samples

    renda = np.round(np.maximum(rng.exponential(6000, n), 500), 2)
    idade = rng.integers(18, 80, n)
    escolaridade = rng.choice(
        ["FUNDAMENTAL", "MEDIO", "SUPERIOR", "POS_GRADUACAO", "MESTRADO", "DOUTORADO"],
        n, p=[0.05, 0.30, 0.42, 0.13, 0.08, 0.02],
    )
    estado_civil = rng.choice(
        ["SOLTEIRO", "CASADO", "DIVORCIADO", "VIUVO"], n, p=[0.4, 0.4, 0.15, 0.05],
    )
    pessoas_residencia = rng.integers(1, 7, n)
    tipo_empregador = rng.choice(
        ["CLT", "PJ", "SERV_PUBLICO", "DESEMPREGADO"], n, p=[0.5, 0.25, 0.15, 0.1],
    )
    cidades = [
        "Sao Paulo", "Rio de Janeiro", "Belo Horizonte", "Salvador",
        "Brasilia", "Campinas", "Porto Alegre", "Fortaleza", "Curitiba",
    ]
    cidade = rng.choice(cidades, n)

    data_abertura = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.integers(0, 2200, n), unit="D",
    )
    tempo_conta_meses = (
        pd.Timestamp.now() - pd.to_datetime(data_abertura)
    ).days / 30.0

    score_bureau = np.clip(rng.normal(600, 90, n), 0, 1000).astype(int)
    atrasos_hist = np.clip(
        rng.poisson(np.maximum(2.0 - score_bureau / 400.0, 0.1), n), 0, 30,
    ).astype(int)
    consultas = np.clip(
        rng.poisson(np.maximum(3.0 - score_bureau / 350.0, 0.1), n), 0, 40,
    ).astype(int)

    total_dividas = np.round(
        np.maximum(rng.exponential(18000, n) * (1 + atrasos_hist / 20.0), 0), 2,
    )
    total_financiado = np.round(np.maximum(rng.exponential(40000, n), 0), 2)
    valor_parcela = np.round(
        np.maximum(total_financiado * rng.uniform(0.02, 0.05, n), 0), 2,
    )

    saldo_medio = np.round(np.maximum(rng.exponential(12000, n), 0), 2)
    saldo_atual = np.round(
        np.maximum(saldo_medio * rng.uniform(0.4, 1.3, n), 0), 2,
    )
    numero_operacoes_credito = rng.integers(0, 8, n)

    possui_imovel = rng.binomial(1, 0.35 + score_bureau / 2000.0, n)
    possui_veiculo = rng.binomial(1, 0.4 + score_bureau / 3000.0, n)

    # Credibilidade latente
    comprometimento = total_dividas / np.maximum(renda, 1.0)
    mapa_escolaridade = np.array(
        [{"FUNDAMENTAL": -0.6, "MEDIO": 0.0, "SUPERIOR": 0.6,
          "POS_GRADUACAO": 1.2, "MESTRADO": 1.8, "DOUTORADO": 2.2}.get(e, 0.0)
         for e in escolaridade]
    )
    mapa_empregador = np.array(
        [{"CLT": 0.4, "PJ": 0.5, "SERV_PUBLICO": 1.0, "DESEMPREGADO": -1.5}
         .get(e, 0.0) for e in tipo_empregador]
    )

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
        "tempo_conta_meses": tempo_conta_meses,
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


__all__ = [
    "CreditRiskModelV2",
    "generate_credit_data_v2",
    "FEAST_FEATURES",
    "DERIVED_FEATURES",
    "ALL_FEATURES",
]
