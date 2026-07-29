"""
AUREUS ML - Previsao de Inadimplencia
Modelo para prever probabilidade de inadimplencia (default) do cliente
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from typing import Dict, Any, List
import warnings
warnings.filterwarnings("ignore")


class DefaultPredictionModel:
    """Previsao de probabilidade de inadimplencia."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            learning_rate=0.1,
        )
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "data_ref" not in df.columns:
            df["data_ref"] = pd.Timestamp.now()
        df["dias_atraso_medio"] = df.get("dias_atraso_medio", np.random.randint(0, 60, len(df)))
        df["valor_pendente_renda"] = (df.get("valor_pendente", 0) / (df.get("renda_mensal", 1) + 1))
        df["num_operacoes_12m"] = df.get("num_operacoes_12m", np.random.randint(0, 50, len(df)))
        df["pct_pagamentos_atraso"] = df.get("pct_pagamentos_atraso", np.random.uniform(0, 0.5, len(df)))
        df["limite_utilizado"] = df.get("limite_utilizado", np.random.uniform(0, 1, len(df)))
        df["tempo_relacionamento_meses"] = df.get("tempo_relacionamento_meses", np.random.randint(1, 120, len(df)))
        df["idade"] = df.get("idade", np.random.randint(22, 70, len(df)))
        df["renda_log"] = np.log1p(df.get("renda_mensal", 5000))
        cols = [
            "dias_atraso_medio",
            "valor_pendente_renda",
            "num_operacoes_12m",
            "pct_pagamentos_atraso",
            "limite_utilizado",
            "tempo_relacionamento_meses",
            "idade",
            "renda_log",
        ]
        self.feature_columns = [c for c in cols if c in df.columns]
        return df[self.feature_columns].fillna(0)

    def train(self, df: pd.DataFrame, target_column: str = "inadimplente"):
        if target_column not in df.columns:
            df[target_column] = (
                (df.get("dias_atraso_medio", 0) > 30)
                | (df.get("pct_pagamentos_atraso", 0) > 0.2)
            ).astype(int)
        X = self.prepare_features(df)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)
        self.model.fit(X_train_s, y_train)
        probs = self.model.predict_proba(X_test_s)[:, 1]
        print("DefaultPrediction - AUC: %.4f" % roc_auc_score(y_test, probs))
        print(classification_report(y_test, self.model.predict(X_test_s)))
        self.is_trained = True

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modelo nao treinado.")
        X = self.prepare_features(df)
        X_s = self.scaler.transform(X)
        return self.model.predict_proba(X_s)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modelo nao treinado.")
        X = self.prepare_features(df)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def save_model(self, path: str):
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "is_trained": self.is_trained,
            },
            path,
        )

    def load_model(self, path: str):
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self.feature_columns = d["feature_columns"]
        self.is_trained = d["is_trained"]


def generate_default_data(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    df = pd.DataFrame({
        "dias_atraso_medio": np.random.exponential(15, n_samples),
        "valor_pendente": np.random.exponential(2000, n_samples),
        "renda_mensal": np.random.exponential(6000, n_samples),
        "num_operacoes_12m": np.random.poisson(20, n_samples),
        "pct_pagamentos_atraso": np.random.beta(1, 5, n_samples),
        "limite_utilizado": np.random.uniform(0, 1, n_samples),
        "tempo_relacionamento_meses": np.random.randint(6, 120, n_samples),
        "idade": np.random.randint(22, 70, n_samples),
    })
    df["inadimplente"] = ((df["dias_atraso_medio"] > 30) | (df["pct_pagamentos_atraso"] > 0.25)).astype(int)
    return df
