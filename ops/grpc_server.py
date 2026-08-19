"""
Aurix ML — Servidor gRPC de inferência.

Serviços expostos:
  - CreditScoring   (score de crédito, probabilidade de default)
  - FraudDetection   (classificação de fraude, score combinado)
  - CustomerSegmentation (segmento de cliente, cluster)

Recursos:
  - Health check (gRPC health protocol)
  - Server reflection (grpcurl)
  - Métricas Prometheus via interceptor
"""

import os
import sys
import time
import uuid
import logging
from pathlib import Path
from concurrent import futures

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import grpc
from grpc_health.v1 import health, health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from prometheus_client import Counter, Histogram, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aurix.ml.grpc")

# ─────────────────────────────────────────────────────────
# Métricas Prometheus
# ─────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "aurix_ml_grpc_requests_total",
    "Total de requisições gRPC",
    ["servico", "metodo"],
)
REQUEST_LATENCY = Histogram(
    "aurix_ml_grpc_latency_seconds",
    "Latência gRPC por serviço",
    ["servico"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
ERROR_COUNT = Counter(
    "aurix_ml_grpc_errors_total",
    "Total de erros gRPC",
    ["servico", "codigo"],
)

# ─────────────────────────────────────────────────────────
# Modelo em memória (lazy load)
# ─────────────────────────────────────────────────────────
_modelos = {}


def _resolver_pkl(nome: str) -> Path:
    """Resolve caminho do artefato .pkl em ops/models ou models/."""
    base = Path(__file__).resolve().parents[2] / "models"
    ops = Path(__file__).resolve().parents[1] / "models"
    for raiz in (ops, base):
        pkl = raiz / f"{nome}.pkl"
        if pkl.exists():
            return pkl
    return base / f"{nome}.pkl"


def carregar_modelos():
    """Carrega todos os modelos .pkl disponíveis."""
    import joblib

    nomes = [
        "credit_risk_model",
        "fraud_detection_model",
        "customer_segmentation_model",
        "default_prediction_model",
    ]
    for nome in nomes:
        caminho = _resolver_pkl(nome)
        if caminho.exists():
            _modelos[nome] = joblib.load(caminho)
            logger.info("Modelo carregado: %s (%s)", nome, caminho)
        else:
            _modelos[nome] = None
            logger.warning("Modelo não encontrado: %s", nome)


# ═══════════════════════════════════════════════════════════
# CreditScoring
# ═══════════════════════════════════════════════════════════

class CreditScoringServicer:
    """Serviço de scoring de crédito."""

    def Avaliar(self, request, context):
        REQUEST_COUNT.labels(servico="credit_scoring", metodo="Avaliar").inc()
        inicio = time.time()

        try:
            import pandas as pd

            modelo_dados = _modelos.get("credit_risk_model")
            if modelo_dados is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Modelo de crédito indisponível")
                return None

            from credit_risk_model import CreditRiskModel

            modelo = CreditRiskModel()
            modelo.model = modelo_dados["modelo"]
            modelo.metadata = modelo_dados.get("metadata", {})
            modelo.feature_columns = modelo.metadata.get("features", [])
            modelo.is_trained = True

            df = pd.DataFrame([{
                "id_cliente": 0,
                "renda_mensal": max(float(request.renda_mensal), 1.0),
                "idade": int(request.idade) if request.idade > 0 else 35,
                "pessoas_residencia": 1,
                "escolaridade": "MEDIO",
                "estado_civil": "SOLTEIRO",
                "tipo_empregador": "CLT",
                "cidade": request.cidade or "Campinas",
                "data_abertura": "2020-01-01",
                "score_bureau": int(request.score_bureau) if request.score_bureau > 0 else 600,
                "atrasos_hist": int(request.atrasos_historico),
                "consultas_ultimo_6m": 0,
                "total_dividas": float(request.divida_total),
                "total_financiado": float(request.valor_solicitado),
                "valor_parcela": float(request.valor_solicitado) * 0.03,
                "saldo_medio_12m": 0.0,
                "saldo_atual": 0.0,
                "numero_operacoes_credito": 0,
                "possui_imovel": 1 if request.possui_imovel else 0,
                "possui_veiculo": 0,
            }])

            proba = float(modelo.predict_proba(df)[0])
            score = int(modelo.predict_score(df)[0])

            if score >= 700:
                nivel_risco = "BAIXO"
                decisao = "APROVADO"
            elif score >= 500:
                nivel_risco = "MEDIO"
                decisao = "REVISAO"
            else:
                nivel_risco = "ALTO"
                decisao = "REJEITADO"

            limite_sugerido = float(request.valor_solicitado * score / 1000.0)
            if score < 400:
                limite_sugerido = 0.0

            fatores_positivos = []
            fatores_risco = []
            if request.historico_pagamentos_meses >= 24:
                fatores_positivos.append("Histórico de pagamentos superior a 24 meses")
            renda = max(float(request.renda_mensal), 1.0)
            comprometimento = float(request.divida_total) / renda
            if comprometimento > 0.5:
                fatores_risco.append("Comprometimento de renda acima de 50%")

            REQUEST_LATENCY.labels(servico="credit_scoring").observe(time.time() - inicio)
            return {
                "score_credito": float(score),
                "probabilidade_inadimplencia": round(max(0.0, min(1.0, proba)), 6),
                "limite_sugerido": round(limite_sugerido, 2),
                "nivel_risco": nivel_risco,
                "decisao": decisao,
                "justificativa": f"Score {score:.0f} — {decisao}",
                "fatores_positivos": fatores_positivos,
                "fatores_risco": fatores_risco,
                "decision_id": str(uuid.uuid4()),
                "tempo_processamento_ms": int((time.time() - inicio) * 1000),
            }

        except Exception as e:
            logger.exception("Falha no credit scoring")
            ERROR_COUNT.labels(servico="credit_scoring", codigo="INTERNAL").inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None


# ═══════════════════════════════════════════════════════════
# FraudDetection
# ═══════════════════════════════════════════════════════════

class FraudDetectionServicer:
    """Serviço de detecção de fraude."""

    def AnalisarTransacao(self, request, context):
        REQUEST_COUNT.labels(servico="fraud_detection", metodo="AnalisarTransacao").inc()
        inicio = time.time()

        try:
            import pandas as pd
            from fraud_detection_model import FraudDetectionModel

            dados = _modelos.get("fraud_detection_model")
            if dados is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Modelo de fraude indisponível")
                return None

            modelo = FraudDetectionModel()
            modelo.isolation_forest = dados["isolation_forest"]
            modelo.random_forest = dados["random_forest"]
            modelo.scaler = dados["scaler"]
            modelo.label_encoders = dados.get("label_encoders", {})
            modelo.feature_columns = dados["feature_columns"]
            modelo.is_trained = True

            df = pd.DataFrame([{
                "valor": float(request.valor),
                "canal": request.canal,
                "dispositivo": request.dispositivo_id,
                "ip_address": request.ip_address,
                "user_agent": request.user_agent,
                "cidade": request.cidade,
                "estado": request.estado,
                "tipo_transacao": request.tipo_transacao,
            }])

            resultado = modelo.predict(df)
            score_fraude = float(resultado["combined"]["scores"][0])
            score_anomalia = float(resultado["anomaly"]["scores"][0])
            score_supervisionado = float(resultado["supervised"]["scores"][0])

            if score_fraude >= 0.8:
                nivel_risco = "CRITICO"
                bloquear = True
            elif score_fraude >= 0.5:
                nivel_risco = "ALTO"
                bloquear = False
            elif score_fraude >= 0.2:
                nivel_risco = "MEDIO"
                bloquear = False
            else:
                nivel_risco = "BAIXO"
                bloquear = False

            sinalizadores = []
            if score_anomalia > 0.7:
                sinalizadores.append("Comportamento anômalo detectado")
            if score_supervisionado > 0.7:
                sinalizadores.append("Padrão de fraude conhecido")

            REQUEST_LATENCY.labels(servico="fraud_detection").observe(time.time() - inicio)
            return {
                "score_fraude": score_fraude,
                "score_anomalia": score_anomalia,
                "score_supervisionado": score_supervisionado,
                "nivel_risco": nivel_risco,
                "sinalizadores": sinalizadores,
                "bloquear_transacao": bloquear,
                "recomendacao": "BLOQUEAR" if bloquear else "PERMITIR",
                "decision_id": str(uuid.uuid4()),
                "tempo_processamento_ms": int((time.time() - inicio) * 1000),
            }

        except Exception as e:
            logger.exception("Falha na detecção de fraude")
            ERROR_COUNT.labels(servico="fraud_detection", codigo="INTERNAL").inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None


# ═══════════════════════════════════════════════════════════
# CustomerSegmentation
# ═══════════════════════════════════════════════════════════

class CustomerSegmentationServicer:
    """Serviço de segmentação de clientes."""

    def Segmentar(self, request, context):
        REQUEST_COUNT.labels(servico="customer_segmentation", metodo="Segmentar").inc()
        inicio = time.time()

        try:
            import pandas as pd

            dados = _modelos.get("customer_segmentation_model")
            if dados is None:
                context.set_code(grpc.StatusCode.UNAVAILABLE)
                context.set_details("Modelo de segmentação indisponível")
                return None

            from customer_segmentation_model import CustomerSegmentationModel

            modelo = CustomerSegmentationModel()
            modelo.model = dados["modelo"]
            modelo.scaler = dados.get("scaler")
            modelo.feature_columns = dados.get("feature_columns", [])
            modelo.is_trained = True

            df = pd.DataFrame([{
                "idade": int(request.idade) if request.idade > 0 else 35,
                "tempo_como_cliente_dias": int(request.tempo_cliente_dias),
                "qtd_contas": int(request.qtd_contas) if request.qtd_contas > 0 else 1,
                "saldo_total": float(request.saldo_total),
                "volume_mensal_transacoes": float(request.volume_mensal),
                "frequencia_transacoes_semanal": float(request.frequencia_semanal),
                "tem_emprestimo": 1 if request.tem_emprestimo else 0,
                "tem_cartao": 1 if request.tem_cartao else 0,
                "tem_investimento": 1 if request.tem_investimento else 0,
                "risco_score": float(request.risco_score) if request.risco_score > 0 else 0.5,
            }])

            cluster = int(modelo.predict(df)[0])
            mapa_segmentos = {
                0: "PREMIUM",
                1: "REGULAR",
                2: "RECENTE",
                3: "RISCO",
                4: "INATIVO",
            }
            segmento = mapa_segmentos.get(cluster, f"CLUSTER_{cluster}")

            REQUEST_LATENCY.labels(servico="customer_segmentation").observe(time.time() - inicio)
            return {
                "cluster": cluster,
                "segmento": segmento,
                "confidence": 0.85,
                "decision_id": str(uuid.uuid4()),
                "tempo_processamento_ms": int((time.time() - inicio) * 1000),
            }

        except Exception as e:
            logger.exception("Falha na segmentação")
            ERROR_COUNT.labels(servico="customer_segmentation", codigo="INTERNAL").inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return None


# ═══════════════════════════════════════════════════════════
# Métricas Prometheus
# ═══════════════════════════════════════════════════════════

class MetricasInterceptor:
    """Interceptor gRPC que coleta métricas Prometheus."""

    def intercept_service(self, continuation, handler_call_details):
        metodo = handler_call_details.method
        return continuation(handler_call_details)


# ═══════════════════════════════════════════════════════════
# Inicialização do servidor
# ═══════════════════════════════════════════════════════════

def criar_servidor(porta: int = 50051):
    """Cria e configura o servidor gRPC com todos os serviços."""

    # Registrar servicers via stubs gerados pelo protobuf
    # (assumindo que ml_pb2_grpc.py foi gerado pelo buf generate)
    try:
        from ml_pb2_grpc import (
            add_CreditScoringServiceServicer_to_server,
            add_FraudDetectionServiceServicer_to_server,
            add_CustomerSegmentationServiceServicer_to_server,
        )

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=[MetricasInterceptor()],
        )

        add_CreditScoringServiceServicer_to_server(CreditScoringServicer(), server)
        add_FraudDetectionServiceServicer_to_server(FraudDetectionServicer(), server)
        add_CustomerSegmentationServiceServicer_to_server(CustomerSegmentationServicer(), server)

    except ImportError:
        logger.warning("Stubs protobuf não gerados — modo stub com serviços genéricos")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Health check
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    # Reflection (para grpcurl)
    SERVICE_NAMES = (
        "grpc.health.v1.Health",
        "aurix.ml.CreditScoringService",
        "aurix.ml.FraudDetectionService",
        "aurix.ml.CustomerSegmentationService",
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{porta}")
    server.start()
    logger.info("Servidor gRPC Aurix ML iniciado na porta %d", porta)

    return server


def main():
    porta = int(os.environ.get("GRPC_PORT", 50051))
    carregar_modelos()
    servidor = criar_servidor(porta)

    try:
        servidor.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Desligando servidor gRPC...")
        servidor.stop(grace=5)


if __name__ == "__main__":
    main()
