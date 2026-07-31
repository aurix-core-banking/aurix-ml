import sys
import os
import time
import uuid
import logging
from pathlib import Path
from concurrent import futures

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "models"))

import grpc
from prometheus_client import Counter, Histogram, generate_latest

from ml_pb2 import (
    FraudAnalysisRequest, FraudAnalysisResponse,
    CreditAnalysisRequest, CreditAnalysisResponse,
    ComplianceCheckRequest, ComplianceCheckResponse,
    GovernanceDecisionRequest, GovernanceDecisionResponse,
)
from ml_pb2_grpc import (
    FraudDetectionServiceServicer,
    CreditAnalysisServiceServicer,
    ComplianceCheckServiceServicer,
    GovernanceServiceServicer,
    add_FraudDetectionServiceServicer_to_server,
    add_CreditAnalysisServiceServicer_to_server,
    add_ComplianceCheckServiceServicer_to_server,
    add_GovernanceServiceServicer_to_server,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("aurix_ml_grpc_requests_total", "gRPC requests", ["service"])
LATENCY = Histogram("aurix_ml_grpc_latency_seconds", "gRPC latency", ["service"])

model_data = {}


def load_models():
    import joblib
    import pandas as pd

    base = Path(__file__).resolve().parents[2] / "models"
    for name in ["fraud_detection_model", "default_prediction_model", "customer_segmentation_model"]:
        pkl = base / f"{name}.pkl"
        if pkl.exists():
            model_data[name] = joblib.load(pkl)
            logger.info("Loaded %s", pkl)
        else:
            logger.warning("Model %s.pkl not found", name)
            model_data[name] = None


class FraudDetectionServicer(FraudDetectionServiceServicer):

    def AnalyzeTransaction(self, request, context):
        start = time.time()
        REQUEST_COUNT.labels(service="fraud").inc()
        decision_id = str(uuid.uuid4())

        try:
            import pandas as pd
            from fraud_detection_model import FraudDetectionModel

            if model_data.get("fraud_detection_model") is None:
                return FraudAnalysisResponse(
                    fraud_score=0.0, risk_level="UNAVAILABLE",
                    block_transaction=False,
                    recommendation="Model not loaded. Allowing by default.",
                    decision_id=decision_id,
                    processing_time_ms=int((time.time() - start) * 1000),
                )

            data = model_data["fraud_detection_model"]
            df = pd.DataFrame([{
                "valor": request.amount,
                "canal": request.channel,
                "dispositivo": request.device_id,
                "ip_address": request.ip_address,
                "user_agent": request.user_agent,
                "cidade": request.city,
                "estado": request.state,
                "tipo_transacao": request.transaction_type,
            }])

            model = FraudDetectionModel()
            model.isolation_forest = data["isolation_forest"]
            model.random_forest = data["random_forest"]
            model.scaler = data["scaler"]
            model.label_encoders = data.get("label_encoders", {})
            model.feature_columns = data["feature_columns"]
            model.is_trained = True

            result = model.predict(df)

            fraud_score = float(result["combined"]["scores"][0])
            anomaly = float(result["anomaly"]["scores"][0])
            supervised = float(result["supervised"]["scores"][0])

            if fraud_score >= 0.8:
                risk_level = "CRITICAL"
                block = True
            elif fraud_score >= 0.5:
                risk_level = "HIGH"
                block = False
            elif fraud_score >= 0.2:
                risk_level = "MEDIUM"
                block = False
            else:
                risk_level = "LOW"
                block = False

            red_flags = []
            if anomaly > 0.7:
                red_flags.append("Comportamento anômalo detectado")
            if supervised > 0.7:
                red_flags.append("Padrão de fraude conhecido")

            return FraudAnalysisResponse(
                fraud_score=fraud_score,
                anomaly_score=anomaly,
                supervised_score=supervised,
                risk_level=risk_level,
                red_flags=red_flags,
                block_transaction=block,
                recommendation="BLOQUEAR" if block else "PERMITIR",
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )

        except Exception as e:
            logger.exception("Fraud analysis failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return FraudAnalysisResponse(
                fraud_score=0.0, risk_level="ERROR",
                block_transaction=False,
                recommendation=f"Error: {str(e)}",
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )


class CreditAnalysisServicer(CreditAnalysisServiceServicer):

    def EvaluateCredit(self, request, context):
        start = time.time()
        REQUEST_COUNT.labels(service="credit").inc()
        decision_id = str(uuid.uuid4())

        try:
            if model_data.get("fraud_detection_model") is not None:
                from fraud_detection_model import CreditScoringModel
                scorer = CreditScoringModel()
                score = scorer.predict([[
                    float(request.monthly_income),
                    float(request.existing_debt),
                    float(request.requested_amount),
                    float(request.payment_history_months),
                ]])[0]
            else:
                score = 500.0

            if score >= 700:
                risk_level = "LOW"
                decision = "APPROVED"
            elif score >= 400:
                risk_level = "MEDIUM"
                decision = "REVIEW"
            else:
                risk_level = "HIGH"
                decision = "REJECTED"

            return CreditAnalysisResponse(
                credit_score=float(score),
                default_probability=max(0, min(1, (700 - score) / 700)),
                suggested_limit=float(request.requested_amount * score / 1000),
                risk_level=risk_level,
                decision=decision,
                justification=f"Score {score:.0f} — {decision}",
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )

        except Exception as e:
            logger.exception("Credit analysis failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return CreditAnalysisResponse(
                credit_score=500.0, risk_level="ERROR",
                decision="ERROR", decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )


class ComplianceCheckServicer(ComplianceCheckServiceServicer):

    def CheckCompliance(self, request, context):
        start = time.time()
        REQUEST_COUNT.labels(service="compliance").inc()
        decision_id = str(uuid.uuid4())

        try:
            regulations = []
            if request.amount > 10000:
                regulations.append("COAF – Art. 11, Lei 9.613/98")
            if request.country and request.country.upper() != "BR":
                regulations.append("BACEN – Circular 3.978/2020")
            regulations.append("LGPD – Lei 13.709/2018")
            regulations.append("CMN – Resolução 4.966/2021")

            is_compliant = True
            status = "CONFORME"
            risk_level = "LOW"

            if request.amount > 50000:
                is_compliant = False
                status = "NAO_CONFORME"
                risk_level = "HIGH"

            return ComplianceCheckResponse(
                is_compliant=is_compliant,
                status=status,
                applicable_regulations=regulations,
                justification="Verificação regulatória concluída" if is_compliant else "Requere análise adicional",
                risk_level=risk_level,
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )

        except Exception as e:
            logger.exception("Compliance check failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return ComplianceCheckResponse(
                is_compliant=False, status="ERROR",
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )


class GovernanceServicer(GovernanceServiceServicer):

    def Decide(self, request, context):
        start = time.time()
        REQUEST_COUNT.labels(service="governance").inc()
        decision_id = str(uuid.uuid4())

        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from aurix_ml.governance import R2Mechanical, R3Adaptive, R1TextOnly, BankingCase
            from aurix_ml.governance import OperationType
            from aurix_ml.llm import create_llm

            op_map = {
                "CREDIT_APPROVAL": OperationType.CREDIT_APPROVAL,
                "PIX_SUSPICIOUS": OperationType.PIX_SUSPICIOUS,
                "KYC_REVIEW": OperationType.KYC_REVIEW,
                "AML_INVESTIGATION": OperationType.AML_INVESTIGATION,
                "COMPLIANCE_CHECK": OperationType.COMPLIANCE_CHECK,
                "LIMIT_INCREASE": OperationType.LIMIT_INCREASE,
                "ACCOUNT_BLOCK": OperationType.ACCOUNT_BLOCK,
            }
            op_type = op_map.get(request.operation_type, OperationType.COMPLIANCE_CHECK)

            llm = create_llm({
                "provider": os.environ.get("AURIX_LLM_PROVIDER", "mock"),
                "model": os.environ.get("AURIX_LLM_MODEL", "governance"),
            })

            banking_case = BankingCase(
                case_id=request.case_id or decision_id,
                operation_type=op_type,
                amount=request.amount,
                currency=request.currency or "BRL",
                risk_score=request.risk_score,
                credit_score=request.credit_score,
                client_id=request.client_id,
                metadata=dict(request.context),
            )

            selected = request.regime or "R3"
            if selected == "R1":
                regime = R1TextOnly(llm)
            elif selected == "R2":
                regime = R2Mechanical(llm)
            else:
                regime = R3Adaptive(llm)

            result = regime.decide(banking_case)

            return GovernanceDecisionResponse(
                decision=result.decision.value,
                confidence=result.confidence,
                regime_used=result.regime,
                justification=str(result.justification) if result.justification else "",
                audit_nonce=result.audit_nonce or "",
                applied_gates=list(result.applied_gates) if result.applied_gates else [],
                audit_verified=result.audit_verified if hasattr(result, 'audit_verified') else False,
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )

        except ImportError:
            logger.warning("Governance package not available, using rule-based fallback")

            score = request.credit_score or 500
            if request.amount > 500000:
                decision = "ESCALATE"
            elif score < 300:
                decision = "REJECT"
            elif score < 500:
                decision = "REVIEW"
            else:
                decision = "APPROVE"

            return GovernanceDecisionResponse(
                decision=decision,
                confidence=0.8,
                regime_used="FALLBACK",
                justification=f"Fallback: score={score}, amount={request.amount}",
                decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )

        except Exception as e:
            logger.exception("Governance decision failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return GovernanceDecisionResponse(
                decision="ERROR", decision_id=decision_id,
                processing_time_ms=int((time.time() - start) * 1000),
            )


def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_FraudDetectionServiceServicer_to_server(FraudDetectionServicer(), server)
    add_CreditAnalysisServiceServicer_to_server(CreditAnalysisServicer(), server)
    add_ComplianceCheckServiceServicer_to_server(ComplianceCheckServicer(), server)
    add_GovernanceServiceServicer_to_server(GovernanceServicer(), server)

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("gRPC server ready on port %d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("GRPC_PORT", 50051))
    serve(port)
