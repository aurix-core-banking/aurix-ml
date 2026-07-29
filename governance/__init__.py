# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""aurix-ml Governance — Decisões LLM auditáveis para o contexto bancário brasileiro.

Implementa governança mecânica inspirada no SantanderAI/mech-gov-framework,
adaptada para as regulamentações BACEN/CMN e casos de uso do Aurix.

Regimes disponíveis
-------------------
``R1TextOnly``
    LLM decide livremente com system prompt de governança. Baixo risco.

``R2Mechanical``
    Pipeline de 6 etapas com hard gates, candidatos CEFL, verificação I6Q,
    ambiguity gate e entropy commit-reveal para trilha de auditoria BACEN.
    Indicado para: aprovação de crédito, KYC, Pix com suspeita de fraude.

``R3Adaptive``
    Seleciona R1 ou R2 dinamicamente conforme o risco calculado do caso.

Uso rápido::

    from aurix_ml.llm import create_llm
    from aurix_ml.governance import R2Mechanical, BankingCase, Decision

    llm = create_llm({"provider": "ollama", "model": "mistral:7b"})
    regime = R2Mechanical(llm)

    case = BankingCase(
        case_id="CRED-2025-001",
        operation_type="credit_approval",
        amount=50_000.00,
        client_score=680,
        income=8_000.00,
        context={"historico_inadimplencia": 0, "tempo_relacionamento_meses": 36},
    )

    result = regime.decide(case)
    print(result.decision)       # Decision.APPROVE | REJECT | DEFER | ESCALATE
    print(result.rationale)      # justificativa estruturada
    print(result.audit_nonce)    # hash para trilha de auditoria BACEN
"""

from aurix_ml.governance.case import BankingCase, Decision, OperationType
from aurix_ml.governance.result import DecisionResult
from aurix_ml.governance.regime import GovernanceRegime
from aurix_ml.governance.r1_text_only import R1TextOnly
from aurix_ml.governance.r2_mechanical import R2Mechanical
from aurix_ml.governance.r3_adaptive import R3Adaptive

__all__ = [
    "BankingCase",
    "Decision",
    "OperationType",
    "DecisionResult",
    "GovernanceRegime",
    "R1TextOnly",
    "R2Mechanical",
    "R3Adaptive",
]
