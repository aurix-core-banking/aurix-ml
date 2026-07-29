"""
AUREUS ML - Segmentacao de Clientes
Modelo K-Means para segmentar clientes (marketing e ofertas)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")


class CustomerSegmentationModel:
    """Segmentacao de clientes por comportamento e perfil."""

    def __init__(self, n_segments: int = 4):
        self.n_segments = n_segments
        self.model = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.segment_names: Dict[int, str] = {}
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["renda_log"] = np.log1p(df.get("renda_mensal", 5000))
        df["transacoes_mes"] = df.get("transacoes_ultimo_mes", 0)
        df["valor_medio"] = (df.get("valor_total_transacoes", 0) / (df.get("transacoes_ultimo_mes", 1) + 1))
        df["saldo_renda"] = (df.get("saldo_atual", 0) / (df.get("renda_mensal", 1) + 1))
        df["idade_norm"] = (df.get("idade", 40) - 20) / 50.0
        cols = ["renda_log", "transacoes_mes", "valor_medio", "saldo_renda", "idade_norm"]
        self.feature_columns = [c for c in cols if c in df.columns]
        return df[self.feature_columns].fillna(0)

    def fit(self, df: pd.DataFrame):
        X = self.prepare_features(df)
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s)
        score = silhouette_score(X_s, self.model.labels_)
        print("CustomerSegmentation - Silhouette: %.4f" % score)
        for i in range(self.n_segments):
            self.segment_names[i] = "segment_%d" % i
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modelo nao treinado.")
        X = self.prepare_features(df)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)

    def get_segment_names(self) -> Dict[int, str]:
        return self.segment_names

    def save_model(self, path: str):
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "segment_names": self.segment_names,
                "n_segments": self.n_segments,
                "is_trained": self.is_trained,
            },
            path,
        )

    def load_model(self, path: str):
        d = joblib.load(path)
        self.model = d["model"]
        self.scaler = d["scaler"]
        self.feature_columns = d["feature_columns"]
        self.segment_names = d.get("segment_names", {})
        self.n_segments = d.get("n_segments", 4)
        self.is_trained = d["is_trained"]


def generate_segmentation_data(n_samples: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "renda_mensal": np.random.exponential(5000, n_samples),
        "transacoes_ultimo_mes": np.random.poisson(25, n_samples),
        "valor_total_transacoes": np.random.exponential(15000, n_samples),
        "saldo_atual": np.random.exponential(3000, n_samples),
        "idade": np.random.randint(22, 65, n_samples),
    })
