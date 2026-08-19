"""
Aurix ML — API de Explicabilidade (FastAPI).

Endpoint POST /explain que recebe um prediction_id e features,
retorna SHAP values, LIME explanation e feature importance.

Uso:
    uvicorn aurix.ml.ops.explainability_api:app --host 0.0.0.0 --port 8206
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aurix.ml.explainability.api")


# ═══════════════════════════════════════════════════════════
# Modelos de request/response
# ═══════════════════════════════════════════════════════════

class ExplicacaoRequest(BaseModel):
    prediction_id: str = Field(..., description="ID da predição a ser explicada")
    features: Dict[str, Any] = Field(..., description="Features da instância predita")
    modelo_tipo: str = Field(
        default="credit_risk",
        description="Tipo do modelo: credit_risk, fraud_detection, customer_segmentation",
    )
    metodo: str = Field(
        default="auto",
        description="Método de explicação: shap, lime, auto (tenta ambos)",
    )
    num_features: int = Field(default=10, ge=1, le=50, description="Número de features na explicação")
    incluir_graficos: bool = Field(default=False, description="Se deve gerar gráficos")


class FeatureImportance(BaseModel):
    feature: str
    shap_value: Optional[float] = None
    lime_value: Optional[float] = None
    peso_absoluto: float


class ExplicacaoResponse(BaseModel):
    prediction_id: str
    explanation_id: str
    metodo_utilizado: str
    shap: Optional[Dict[str, Any]] = None
    lime: Optional[Dict[str, Any]] = None
    feature_importance: List[FeatureImportance]
    prediction: Optional[Dict[str, Any]] = None
    tempo_processamento_ms: int


# ═══════════════════════════════════════════════════════════
# App FastAPI
# ═══════════════════════════════════════════════════════════

app = FastAPI(
    title="Aurix ML Explainability API",
    description="Explicabilidade de predições com SHAP e LIME",
    version="1.0.0",
)


# Cache de modelos em memória
_modelos_cache: Dict[str, Any] = {}
_feature_columns_cache: Dict[str, List[str]] = {}

# Feature columns padrão por modelo
FEATURE_COLUMNS_PADRAO = {
    "credit_risk": [
        "renda_mensal", "idade", "score_bureau", "atrasos_hist",
        "total_dividas", "total_financiado", "valor_parcela",
        "saldo_medio_12m", "saldo_atual", "numero_operacoes_credito",
        "possui_imovel", "possui_veiculo", "pessoas_residencia",
    ],
    "fraud_detection": [
        "valor", "hora", "dia_semana", "is_fim_de_semana",
        "is_horario_comercial", "transacoes_ultima_hora",
        "valor_acumulado_1h", "distancia_ultima_transacao_km",
        "qtd_dispositivos_30d", "is_novo_dispositivo",
    ],
    "customer_segmentation": [
        "idade", "tempo_como_cliente_dias", "qtd_contas", "saldo_total",
        "volume_mensal_transacoes", "frequencia_transacoes_semanal",
        "tem_emprestimo", "tem_cartao", "tem_investimento", "risco_score",
    ],
}


def _obter_modelo(modelo_tipo: str):
    """Carrega modelo do disco se necessário."""
    import joblib
    from pathlib import Path

    if modelo_tipo in _modelos_cache:
        return _modelos_cache[modelo_tipo]

    mapa_modelos = {
        "credit_risk": "credit_risk_model",
        "fraud_detection": "fraud_detection_model",
        "customer_segmentation": "customer_segmentation_model",
    }
    nome = mapa_modelos.get(modelo_tipo)
    if nome is None:
        raise ValueError(f"Tipo de modelo não suportado: {modelo_tipo}")

    base = Path(__file__).resolve().parents[2] / "models"
    ops = Path(__file__).resolve().parents[1] / "models"
    for raiz in (ops, base):
        pkl = raiz / f"{nome}.pkl"
        if pkl.exists():
            dados = joblib.load(pkl)
            _modelos_cache[modelo_tipo] = dados
            return dados

    raise FileNotFoundError(f"Modelo {nome}.pkl não encontrado")


def _construir_array_features(features: Dict[str, Any], feature_columns: List[str]):
    """Converte dict de features para array numpy na ordem correta."""
    import numpy as np
    valores = []
    for col in feature_columns:
        valores.append(float(features.get(col, 0.0)))
    return np.array([valores])


# ═══════════════════════════════════════════════════════════
# Endpoint POST /explain
# ═══════════════════════════════════════════════════════════

@app.post("/explain", response_model=ExplicacaoResponse)
async def explicar(req: ExplicacaoRequest):
    """Explica uma predição usando SHAP e/ou LIME."""
    inicio = time.time()
    explanation_id = str(uuid.uuid4())

    feature_columns = FEATURE_COLUMNS_PADRAO.get(req.modelo_tipo, [])
    if not feature_columns:
        raise HTTPException(status_code=400, detail=f"Tipo de modelo desconhecido: {req.modelo_tipo}")

    try:
        modelo_dados = _obter_modelo(req.modelo_tipo)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    X = _construir_array_features(req.features, feature_columns)

    resultado_shap = None
    resultado_lime = None
    metodo = req.metodo

    # ── SHAP ──
    if metodo in ("shap", "auto"):
        try:
            from aurix.ml.ops.shap_explainer import SHAPExplainer

            modelo_raw = _obter_modelo_raw(modelo_dados, req.modelo_tipo)
            shap_exp = SHAPExplainer(modelo_raw, feature_columns, tipo="tree")
            shap_exp._criar_explainer(X)
            resultado_shap = shap_exp.explicar_local(X)
            metodo = "shap"
        except Exception as e:
            logger.warning("SHAP falhou, tentando LIME: %s", e)
            if metodo == "shap":
                raise HTTPException(status_code=500, detail=f"SHAP falhou: {e}")

    # ── LIME ──
    if metodo in ("lime", "auto") and resultado_shap is None:
        try:
            from aurix.ml.ops.lime_explainer import LIMEExplainer

            modelo_raw = _obter_modelo_raw(modelo_dados, req.modelo_tipo)
            lime_exp = LIMEExplainer(modelo_raw, feature_columns)
            resultado_lime = lime_exp.explicar_instancia(X, num_features=req.num_features)
            metodo = "lime"
        except Exception as e:
            logger.warning("LIME falhou: %s", e)
            if metodo == "lime":
                raise HTTPException(status_code=500, detail=f"LIME falhou: {e}")

    # ── Feature Importance combinada ──
    importancia = []
    shap_vals = {}
    lime_vals = {}

    if resultado_shap and "feature_importance" in resultado_shap:
        shap_vals = resultado_shap["feature_importance"]
    if resultado_lime and "feature_importance" in resultado_lime:
        lime_vals = resultado_lime["feature_importance"]

    todas_features = set(list(shap_vals.keys()) + list(lime_vals.keys()) + feature_columns)
    for feat in todas_features:
        shap_v = shap_vals.get(feat)
        lime_v = lime_vals.get(feat)
        peso = max(abs(shap_v or 0), abs(lime_v or 0))
        importancia.append(FeatureImportance(
            feature=feat,
            shap_value=shap_v,
            lime_value=lime_v,
            peso_absoluto=peso,
        ))

    importancia.sort(key=lambda x: x.peso_absoluto, reverse=True)

    tempo_ms = int((time.time() - inicio) * 1000)

    return ExplicacaoResponse(
        prediction_id=req.prediction_id,
        explanation_id=explanation_id,
        metodo_utilizado=metodo,
        shap=resultado_shap,
        lime=resultado_lime,
        feature_importance=importancia[:req.num_features],
        prediction=resultado_shap.get("prediction") if resultado_shap else None,
        tempo_processamento_ms=tempo_ms,
    )


def _obter_modelo_raw(dados: dict, modelo_tipo: str):
    """Retorna o modelo raw do dict carregado do .pkl."""
    mapa_chaves = {
        "credit_risk": "modelo",
        "fraud_detection": "isolation_forest",
        "customer_segmentation": "modelo",
    }
    chave = mapa_chaves.get(modelo_tipo, "modelo")
    return dados.get(chave)


@app.get("/health")
async def health():
    return {"status": "healthy", "servico": "explainability_api"}


@app.get("/modelos")
async def listar_modelos():
    """Lista modelos disponíveis e seus feature columns."""
    return {
        "modelos": {
            k: {"feature_columns": v}
            for k, v in FEATURE_COLUMNS_PADRAO.items()
        }
    }
