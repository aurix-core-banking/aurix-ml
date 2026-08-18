"""
Servico de Credit Scoring — FastAPI com integracao Feast e SHAP.

Endpoints:
- POST /predict: recebe customer_id, busca features no Feast, retorna risk_score
- GET /health: health check do servico
- GET /metrics: metricas Prometheus

Saida:
- risk_score (300-850)
- risk_level (A/B/C/D)
- shap_explanation (top features que influenciaram a decisao)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Adiciona paths para imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Metricas Prometheus
REQUEST_COUNT = Counter("aurix_credit_scoring_requests_total", "Total de requisicoes", ["endpoint", "status"])
LATENCY = Histogram("aurix_credit_scoring_seconds", "Latencia de scoring", ["endpoint"])

# Variaveis globais
_modelo = None
_feature_store = None

MODEL_PATH = os.environ.get(
    "AUREUS_CREDIT_MODEL_PATH", "models/credit_risk_v2.pkl",
)
FEAST_REPO_PATH = os.environ.get("FEAST_REPO_PATH", ".")


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    customer_id: int = Field(..., description="ID do cliente no core bancario")
    event_timestamp: Optional[str] = Field(None, description="Timestamp da consulta (ISO format)")


class SHAPFeature(BaseModel):
    feature: str
    valor: float
    impacto: float


class PredictResponse(BaseModel):
    customer_id: int
    risk_score: int = Field(..., description="Score de risco 300-850")
    risk_level: str = Field(..., description="Nivel de risco: A, B, C ou D")
    default_probability: float = Field(..., description="Probabilidade de inadimplencia (0-1)")
    shap_explanation: Optional[List[SHAPFeature]] = Field(None, description="Top 10 features SHAP")
    model_version: str = "2.0.0"


class HealthResponse(BaseModel):
    status: str
    modelo_carregado: bool
    feast_configurado: bool
    model_version: str


# ------------------------------------------------------------------
# App FastAPI
# ------------------------------------------------------------------
app = FastAPI(
    title="Aurix Credit Scoring Service",
    description="API de score de credito com Feast feature store e SHAP",
    version="2.0.0",
)


def _carregar_modelo() -> None:
    """Carrega o modelo de credito do disco."""
    global _modelo
    path = Path(MODEL_PATH)
    if not path.is_absolute():
        base = Path(__file__).resolve().parent
        path = base.parent / path if not (base.parent / path).exists() else base.parent.parent / path
    if not path.exists():
        logger.warning("Modelo nao encontrado em %s", path)
        return

    try:
        import joblib
        dados = joblib.load(path)
        if isinstance(dados, dict) and "model" in dados:
            _modelo = dados
        else:
            from credit_risk_model_v2 import CreditRiskModelV2
            _modelo = CreditRiskModelV2.load_model(str(path))
        logger.info("Modelo carregado de %s", path)
    except Exception as e:
        logger.exception("Falha ao carregar modelo: %s", e)


def _buscar_features_feast(
    customer_id: int, event_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Busca features do Feast feature store.

    Se Feast nao estiver disponivel, retorna dict vazio (fallback para dados default).
    """
    try:
        from feast import FeatureStore
        import pandas as pd

        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        ts = event_timestamp or pd.Timestamp.now().isoformat()

        entity_df = pd.DataFrame({
            "id_cliente": [customer_id],
            "event_timestamp": [ts],
        })

        features = store.get_historical_features(
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
                "credit_features:tempo_conta_meses",
                "credit_features:pessoas_residencia",
                "credit_features:possui_imovel",
                "credit_features:possui_veiculo",
            ],
        ).to_df()

        if not features.empty:
            return features.iloc[0].to_dict()
    except Exception as e:
        logger.debug("Feast indisponivel: %s", e)

    return {}


def _construir_df_cliente(
    customer_id: int, feast_features: Dict[str, Any],
):
    """Constrói DataFrame com features do Feast + defaults."""
    import pandas as pd

    padrao = {
        "id_cliente": customer_id,
        "renda_mensal": 5000.0,
        "idade": 35,
        "pessoas_residencia": 2,
        "escolaridade": "SUPERIOR",
        "estado_civil": "SOLTEIRO",
        "tipo_empregador": "CLT",
        "cidade": "Sao Paulo",
        "data_abertura": "2020-01-01",
        "tempo_conta_meses": 60.0,
        "score_bureau": 600,
        "atrasos_hist": 0,
        "consultas_ultimo_6m": 0,
        "total_dividas": 0.0,
        "total_financiado": 0.0,
        "valor_parcela": 0.0,
        "saldo_medio_12m": 10000.0,
        "saldo_atual": 12000.0,
        "numero_operacoes_credito": 3,
        "possui_imovel": 0,
        "possui_veiculo": 0,
    }

    # Atualiza com features do Feast
    for k, v in feast_features.items():
        if k in padrao and v is not None and not (isinstance(v, float) and np.isnan(v)):
            padrao[k] = v

    return pd.DataFrame([padrao])


def _calcular_shap_explanation(
    model_data: Any, df,
) -> Optional[List[SHAPFeature]]:
    """Calcula explicabilidade SHAP para a predicao."""
    try:
        import shap

        if isinstance(model_data, dict) and "model" in model_data:
            modelo = model_data["model"]
            feature_columns = model_data.get("feature_columns", [])
        else:
            modelo = model_data.model
            feature_columns = model_data.feature_columns

        explainer = shap.TreeExplainer(modelo)
        X = model_data.prepare_features(df) if hasattr(model_data, "prepare_features") else df[feature_columns]
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        importancia = np.abs(shap_values[0]).tolist()
        features = feature_columns[:len(importancia)]

        # Top 10 features
        pares = sorted(zip(features, importancia), key=lambda x: x[1], reverse=True)[:10]
        return [SHAPFeature(feature=f, valor=round(float(v), 6), impacto=round(float(v), 6))
                for f, v in pares]
    except Exception as e:
        logger.debug("SHAP indisponivel: %s", e)
        return None


def _classificar_risco(probabilidade: float) -> str:
    """Classifica o risco em A/B/C/D baseado na probabilidade."""
    if probabilidade < 0.15:
        return "A"
    elif probabilidade < 0.35:
        return "B"
    elif probabilidade < 0.60:
        return "C"
    else:
        return "D"


import logging

logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup():
    _carregar_modelo()


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check do servico."""
    return HealthResponse(
        status="healthy" if _modelo is not None else "degraded",
        modelo_carregado=_modelo is not None,
        feast_configurado=Path(FEAST_REPO_PATH / "feature_store.yaml").exists()
        if isinstance(FEAST_REPO_PATH, (str, Path)) else False,
        model_version="2.0.0",
    )


@app.get("/metrics")
def metrics():
    """Metricas Prometheus."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse)
@LATENCY.labels(endpoint="predict").time()
def predict(req: PredictRequest):
    """Score de credito para um cliente.

    Busca features do Feast, calcula probabilidade de inadimplencia,
    converte para score (300-850), classifica risco (A/B/C/D) e
    gera explicabilidade SHAP.
    """
    REQUEST_COUNT.labels(endpoint="predict", status="started").inc()

    if _modelo is None:
        REQUEST_COUNT.labels(endpoint="predict", status="model_not_loaded").inc()
        raise HTTPException(
            status_code=503,
            detail="Modelo nao carregado. Execute o pipeline de treino primeiro.",
        )

    # Busca features do Feast
    feast_features = _buscar_features_feast(req.customer_id, req.event_timestamp)

    # Constroi DataFrame
    df = _construir_df_cliente(req.customer_id, feast_features)

    # Predicao
    try:
        if isinstance(_modelo, dict):
            import joblib
            modelo = _modelo["model"]
            feature_columns = _modelo.get("feature_columns", [])
            from credit_risk_model_v2 import CreditRiskModelV2
            wrapper = CreditRiskModelV2()
            wrapper.model = modelo
            wrapper.feature_columns = feature_columns
            wrapper.is_trained = True

            proba = float(wrapper.predict_proba(df)[0])
            score = int(wrapper.predict_score(df)[0])
        else:
            proba = float(_modelo.predict_proba(df)[0])
            score = int(_modelo.predict_score(df)[0])
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        logger.exception("Erro na predicao")
        raise HTTPException(status_code=422, detail=str(e)) from e

    risk_level = _classificar_risco(proba)
    shap_explanation = _calcular_shap_explanation(_modelo, df)

    REQUEST_COUNT.labels(endpoint="predict", status="success").inc()

    return PredictResponse(
        customer_id=req.customer_id,
        risk_score=score,
        risk_level=risk_level,
        default_probability=round(proba, 6),
        shap_explanation=shap_explanation,
        model_version="2.0.0",
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8210"))
    uvicorn.run(app, host=host, port=port)
