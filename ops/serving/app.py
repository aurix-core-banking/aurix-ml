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
model_data = None

REQUEST_COUNT = Counter("aurix_ml_requests_total", "Total prediction requests", ["model"])
LATENCY = Histogram("aurix_ml_predict_seconds", "Prediction latency")


class PredictRequest(BaseModel):
    transactions: list = Field(..., description="List of transaction records for prediction")


class PredictResponse(BaseModel):
    predictions: list
    scores: list
    model_version: str = "1.0"


def load_model():
    global model_data
    path = Path(MODEL_PATH)
    if not path.is_absolute():
        base = Path(__file__).resolve().parents[1]
        path = base.parent / path
    if not path.exists():
        logger.warning("Model file not found at %s, serving will return mock until model is available", path)
        model_data = None
        return
    model_data = joblib.load(path)
    logger.info("Model loaded from %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    pass


app = FastAPI(title="AUREUS ML Serving", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_data is not None}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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


@app.post("/predict/batch")
@LATENCY.time()
def predict_batch(req: PredictRequest):
    return predict(req)
