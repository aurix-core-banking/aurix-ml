import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import os
import yaml
import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("AUREUS_MODEL_PATH", "models/fraud_detection_model.pkl")
CREDIT_MODEL_PATH = os.environ.get("AUREUS_CREDIT_MODEL_PATH", "models/credit_risk_model.pkl")
model_data = None
credit_model_data = None

REQUEST_COUNT = Counter("aurix_ml_requests_total", "Total prediction requests", ["model"])
LATENCY = Histogram("aurix_ml_predict_seconds", "Prediction latency")


class PredictRequest(BaseModel):
    transactions: list = Field(..., description="List of transaction records for prediction")


class PredictResponse(BaseModel):
    predictions: list
    scores: list
    model_version: str = "1.0.0"


class ScoreRequest(BaseModel):
    cliente: dict = Field(..., description="Dados do cliente (renda, score_bureau, etc.) para score de crédito")


class ScoreResponse(BaseModel):
    score: int
    default_probability: float
    risk_level: str
    decision: str
    model_version: str = "1.0.0"
    regime: str = "r3"


def _resolver_caminho_modelo(caminho: str) -> Path:
    """Resolve o caminho do artefato para onde o treino realmente grava.

    Treinos via ``ops`` gravam em ``ops/models``; treinos diretos em
    ``models/`` gravam em ``aurix-ml/models``. Aceita ambos mais caminho
    absoluto (env var) para nao quebrar o serving conforme o pipeline usado.
    """
    path = Path(caminho)
    if path.is_absolute():
        return path
    base = Path(__file__).resolve().parent  # ops/serving
    ops_models = base.parent / path         # ops/models/<nome>
    if ops_models.exists():
        return ops_models
    return base.parent.parent / path        # aurix-ml/models/<nome>


def _carregar_fraude():
    global model_data
    path = _resolver_caminho_modelo(MODEL_PATH)
    if not path.exists():
        logger.warning("Model file not found at %s, serving will return mock until model is available", path)
        model_data = None
        return
    model_data = joblib.load(path)
    logger.info("Model loaded from %s", path)


def _carregar_credito():
    global credit_model_data
    path = _resolver_caminho_modelo(CREDIT_MODEL_PATH)
    if not path.exists():
        logger.warning("Modelo de credito nao encontrado em %s — /score retornara 503", path)
        credit_model_data = None
        return
    try:
        from credit_risk_model import CreditRiskModel
        credit_model_data = CreditRiskModel.load_model(str(path))
        logger.info("Modelo de credito carregado de %s", path)
    except Exception as e:
        logger.exception("Falha ao carregar modelo de credito de %s", path)
        credit_model_data = None


def load_model():
    _carregar_fraude()
    _carregar_credito()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    pass


app = FastAPI(title="AUREUS ML Serving", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_data is not None,
        "credit_model_loaded": credit_model_data is not None,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _construir_df_cliente(cliente: dict):
    """Constroi DataFrame com os campos raiz esperados pelo CreditRiskModel."""
    import pandas as pd

    padrao = {
        "id_cliente": 0,
        "renda_mensal": 3000.0,
        "idade": 35,
        "pessoas_residencia": 1,
        "escolaridade": "MEDIO",
        "estado_civil": "SOLTEIRO",
        "tipo_empregador": "CLT",
        "cidade": "Campinas",
        "data_abertura": "2020-01-01",
        "score_bureau": 600,
        "atrasos_hist": 0,
        "consultas_ultimo_6m": 0,
        "total_dividas": 0.0,
        "total_financiado": 0.0,
        "valor_parcela": 0.0,
        "saldo_medio_12m": 0.0,
        "saldo_atual": 0.0,
        "numero_operacoes_credito": 0,
        "possui_imovel": 0,
        "possui_veiculo": 0,
    }
    linha = dict(padrao)
    linha.update({k: v for k, v in cliente.items() if k in padrao})
    return pd.DataFrame([linha])


@app.post("/predict", response_model=PredictResponse)
@LATENCY.time()
def predict(req: PredictRequest):
    REQUEST_COUNT.labels(model="fraud_detection").inc()
    if not req.transactions:
        raise HTTPException(status_code=400, detail="transactions list is required")

    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first and set AUREUS_MODEL_PATH.",
        )

    import pandas as pd
    from fraud_detection_model import FraudDetectionModel

    df = pd.DataFrame(req.transactions)
    model = FraudDetectionModel()
    model.isolation_forest = model_data["isolation_forest"]
    model.random_forest = model_data["random_forest"]
    model.scaler = model_data["scaler"]
    model.label_encoders = model_data.get("label_encoders", {})
    model.feature_columns = model_data["feature_columns"]
    model.is_trained = True

    try:
        result = model.predict(df)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=str(e)) from e

    return PredictResponse(
        predictions=result["combined"]["predictions"],
        scores=[float(x) for x in result["combined"]["scores"]],
    )


@app.post("/predict/credit", response_model=ScoreResponse)
@LATENCY.time()
def predict_credit(req: ScoreRequest):
    """Score de credito para um unico cliente (mesmo contrato de /score)."""
    REQUEST_COUNT.labels(model="credit_risk").inc()
    return _computar_score(req)


@app.post("/score", response_model=ScoreResponse)
@LATENCY.time()
def score(req: ScoreRequest):
    REQUEST_COUNT.labels(model="credit_risk").inc()
    return _computar_score(req)


def _computar_score(req: ScoreRequest) -> ScoreResponse:
    if credit_model_data is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo de credito nao carregado. Execute models/train_credit_pipeline.py "
                   "e configure AUREUS_CREDIT_MODEL_PATH.",
        )

    try:
        df = _construir_df_cliente(req.cliente)
        proba = float(credit_model_data.predict_proba(df)[0])
        score_val = int(credit_model_data.predict_score(df)[0])
    except Exception as e:
        logger.exception("Credit scoring failed")
        raise HTTPException(status_code=422, detail=str(e)) from e

    if score_val >= 700:
        risk_level = "LOW"
        decision = "APROVADO"
    elif score_val >= 500:
        risk_level = "MEDIUM"
        decision = "REVISAO"
    else:
        risk_level = "HIGH"
        decision = "REJEITADO"

    metadata = getattr(credit_model_data, "metadata", {}) or {}
    return ScoreResponse(
        score=score_val,
        default_probability=round(proba, 6),
        risk_level=risk_level,
        decision=decision,
        model_version=str(metadata.get("model_version", "1.0.0")),
        regime=str(metadata.get("regime", "r3")),
    )


@app.post("/predict/batch")
@LATENCY.time()
def predict_batch(req: PredictRequest):
    return predict(req)
