"""
Aurix ML — Explicabilidade com SHAP.

Fornece explicações globais e locais para modelos de crédito e fraude:
  - Resumo global (summary plot)
  - Waterfall (instância individual)
  - Force plot (contribuição por feature)
  - Dependence plot (relação feature vs SHAP value)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("aurix.ml.explainability.shap")


class SHAPExplainer:
    """Wrapper para explicabilidade SHAP em modelos Aurix."""

    def __init__(self, modelo, feature_columns: List[str], tipo: str = "tree"):
        """
        Args:
            modelo: modelo treinado (XGBoost, RandomForest, etc.)
            feature_columns: nomes das features de entrada
            tipo: 'tree' (TreeExplainer), 'kernel' (KernelExplainer), 'linear'
        """
        self.modelo = modelo
        self.feature_columns = feature_columns
        self.tipo = tipo
        self._explainer = None
        self._valores = None

    def _criar_explainer(self, X_background: Optional[np.ndarray] = None):
        """Cria o explainer SHAP conforme o tipo de modelo."""
        import shap

        if self.tipo == "tree":
            self._explainer = shap.TreeExplainer(self.modelo)
        elif self.tipo == "kernel":
            background = shap.sample(X_background, 100) if X_background is not None else X_background
            self._explainer = shap.KernelExplainer(
                self.modelo.predict_proba if hasattr(self.modelo, "predict_proba") else self.modelo.predict,
                background,
            )
        elif self.tipo == "linear":
            self._explainer = shap.LinearExplainer(self.modelo, X_background)
        else:
            raise ValueError(f"Tipo de explainer não suportado: {self.tipo}")

        logger.info("Explainer SHAP (%s) criado", self.tipo)

    def explicar_local(self, X: np.ndarray) -> Dict[str, Any]:
        """Gera explicação SHAP para uma instância.

        Args:
            X: array 2D com uma ou mais instâncias

        Returns:
            dict com shap_values, feature_importance, base_value
        """
        import shap

        if self._explainer is None:
            self._criar_explainer(X)

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # classe positiva

        valores_instancia = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base_value = float(self._explainer.expected_value[1]) if isinstance(
            self._explainer.expected_value, (list, np.ndarray)
        ) else float(self._explainer.expected_value)

        feature_importance = {
            feat: float(val)
            for feat, val in zip(self.feature_columns, valores_instancia)
        }

        # Ordenar por magnitude absoluta
        feature_importance_ordenado = dict(
            sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return {
            "shap_values": valores_instancia.tolist(),
            "feature_importance": feature_importance_ordenado,
            "base_value": base_value,
            "prediction": float(base_value + sum(valores_instancia)),
        }

    def explicar_global(self, X: np.ndarray) -> Dict[str, Any]:
        """Gera explicação SHAP global (resumo de todas as instâncias).

        Args:
            X: array 2D com múltiplas instâncias

        Returns:
            dict com importance_global, mean_abs_shap, shap_values_matrix
        """
        import shap

        if self._explainer is None:
            self._criar_explainer(X)

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_abs = np.mean(np.abs(shap_values), axis=0)

        importance_global = {
            feat: float(val)
            for feat, val in zip(self.feature_columns, mean_abs)
        }
        importance_global = dict(sorted(importance_global.items(), key=lambda x: x[1], reverse=True))

        return {
            "importance_global": importance_global,
            "mean_abs_shap": mean_abs.tolist(),
            "shap_values_matrix": shap_values.tolist(),
            "n_instancias": len(X),
        }

    def gerar_grafico_resumo(self, X: np.ndarray, caminho: str):
        """Gera gráfico de resumo SHAP (summary plot)."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._explainer is None:
            self._criar_explainer(X)

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.feature_columns,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Gráfico de resumo salvo em %s", caminho)

    def gerar_grafico_waterfall(self, X_instancia: np.ndarray, caminho: str):
        """Gera gráfico waterfall para uma instância."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._explainer is None:
            self._criar_explainer(X_instancia)

        shap_values = self._explainer.shap_values(X_instancia)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        valor = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        base = float(self._explainer.expected_value[1]) if isinstance(
            self._explainer.expected_value, (list, np.ndarray)
        ) else float(self._explainer.expected_value)

        explicacao = shap.Explanation(
            values=valor,
            base_values=base,
            data=X_instancia[0],
            feature_names=self.feature_columns,
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(explicacao, show=False)
        plt.tight_layout()
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Gráfico waterfall salvo em %s", caminho)

    def gerar_grafico_force(self, X_instancia: np.ndarray, caminho: str):
        """Gera gráfico de força (force plot) para uma instância."""
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._explainer is None:
            self._criar_explainer(X_instancia)

        shap_values = self._explainer.shap_values(X_instancia)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base = float(self._explainer.expected_value[1]) if isinstance(
            self._explainer.expected_value, (list, np.ndarray)
        ) else float(self._explainer.expected_value)

        shap.force_plot(
            base,
            shap_values[0] if len(shap_values.shape) > 1 else shap_values,
            X_instancia[0],
            feature_names=self.feature_columns,
            matplotlib=True,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(caminho, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Gráfico force plot salvo em %s", caminho)
