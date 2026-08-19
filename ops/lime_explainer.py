"""
Aurix ML — Explicabilidade com LIME.

LIME (Local Interpretable Model-agnostic Explanations) para dados tabulares.
Alternativa ao SHAP quando o modelo é caixa-preta ou quando se deseja
explicação apenas local.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("aurix.ml.explainability.lime")


class LIMEExplainer:
    """Wrapper para explicabilidade LIME em modelos Aurix."""

    def __init__(
        self,
        modelo,
        feature_columns: List[str],
        X_treino: Optional[np.ndarray] = None,
        classe_positiva: int = 1,
    ):
        """
        Args:
            modelo: modelo treinado
            feature_columns: nomes das features
            X_treino: dados de treino para calibrar o LIME
            classe_positiva: índice da classe positiva (default: 1)
        """
        self.modelo = modelo
        self.feature_columns = feature_columns
        self.X_treino = X_treino
        self.classe_positiva = classe_positiva
        self._explainer = None

    def _criar_explainer(self):
        """Cria o explainer LIME para dados tabulares."""
        from lime.lime_tabular import LimeTabularExplainer

        predict_fn = (
            self.modelo.predict_proba
            if hasattr(self.modelo, "predict_proba")
            else self.modelo.predict
        )

        self._explainer = LimeTabularExplainer(
            training_data=self.X_treino if self.X_treino is not None else np.zeros((100, len(self.feature_columns))),
            feature_names=self.feature_columns,
            class_names=["negativo", "positivo"],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("Explainer LIME criado para %d features", len(self.feature_columns))

    def explicar_instancia(
        self,
        X: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000,
    ) -> Dict[str, Any]:
        """Gera explicação LIME para uma instância.

        Args:
            X: array 1D ou 2D com a instância (shape: [n_features] ou [1, n_features])
            num_features: número de features na explicação
            num_samples: número de amostras para perturbação

        Returns:
            dict com feature_importance, prediction, probabilities
        """
        if self._explainer is None:
            self._criar_explainer()

        instancia = X.flatten() if X.ndim > 1 else X

        predict_fn = (
            self.modelo.predict_proba
            if hasattr(self.modelo, "predict_proba")
            else self.modelo.predict
        )

        explicacao = self._explainer.explain_instance(
            instancia,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
        )

        # Extrair pesos das features
        pesos = explicacao.as_list()
        feature_importance = {
            feat.split(" ")[0] if " " in feat else feat: float(peso)
            for feat, peso in pesos
        }

        probabilidades = None
        try:
            pred = predict_fn(instancia.reshape(1, -1))
            if pred.ndim > 1 and pred.shape[1] >= 2:
                probabilidades = {
                    "negativo": float(pred[0][0]),
                    "positivo": float(pred[0][1]),
                }
        except Exception:
            pass

        return {
            "feature_importance": feature_importance,
            "prediction": int(explicacao.local_pred[0]) if hasattr(explicacao, "local_pred") else None,
            "probabilities": probabilidades,
            "score": float(explicacao.score) if hasattr(explicacao, "score") else None,
            "intercept": float(explicacao.intercept[1]) if hasattr(explicacao, "intercept") and len(explicacao.intercept) > 1 else 0.0,
        }

    def explicar_batch(
        self,
        X: np.ndarray,
        num_features: int = 10,
    ) -> List[Dict[str, Any]]:
        """Explica múltiplas instâncias.

        Args:
            X: array 2D com múltiplas instâncias
            num_features: features por explicação

        Returns:
            lista de dicts com feature_importance por instância
        """
        resultados = []
        for i in range(len(X)):
            try:
                resultado = self.explicar_instancia(X[i], num_features=num_features)
                resultado["indice"] = i
                resultados.append(resultado)
            except Exception as e:
                logger.warning("Falha ao explicar instância %d: %s", i, e)
                resultados.append({"indice": i, "erro": str(e)})
        return resultados

    def importancia_media(
        self,
        X: np.ndarray,
        num_features: int = 10,
    ) -> Dict[str, float]:
        """Calcula importância média das features em múltiplas instâncias.

        Returns:
            dict com média absoluta dos pesos LIME por feature
        """
        resultados = self.explicar_batch(X, num_features=num_features)

        acumulador = {feat: 0.0 for feat in self.feature_columns}
        contagem = 0

        for r in resultados:
            if "feature_importance" in r:
                contagem += 1
                for feat, peso in r["feature_importance"].items():
                    # Mapear nome truncado de volta ao nome completo
                    for col in self.feature_columns:
                        if col.startswith(feat) or feat.startswith(col):
                            acumulador[col] += abs(peso)
                            break

        if contagem > 0:
            return {k: v / contagem for k, v in sorted(acumulador.items(), key=lambda x: x[1], reverse=True)}
        return acumulador
