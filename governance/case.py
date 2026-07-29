# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Modelo de caso bancário — a unidade de entrada do pipeline de governança.

Representa qualquer operação que requer decisão LLM governada:
crédito, KYC, Pix suspeito, compliance, onboarding, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class Decision(str, Enum):
    """Decisões possíveis do pipeline de governança."""
    APPROVE  = "APPROVE"    # Aprovar a operação
    REJECT   = "REJECT"     # Rejeitar a operação
    DEFER    = "DEFER"      # Adiar — aguardar mais informação
    ESCALATE = "ESCALATE"   # Escalar para analista humano


class OperationType(str, Enum):
    """Tipos de operação bancária suportados."""
    CREDIT_APPROVAL    = "credit_approval"       # Aprovação de crédito
    PIX_SUSPICIOUS     = "pix_suspicious"        # Pix com suspeita de fraude
    KYC_REVIEW         = "kyc_review"            # Revisão KYC / onboarding
    AML_INVESTIGATION  = "aml_investigation"     # Investigação AML/COAF
    COMPLIANCE_CHECK   = "compliance_check"      # Verificação de compliance geral
    LIMIT_INCREASE     = "limit_increase"        # Aumento de limite
    ACCOUNT_BLOCK      = "account_block"         # Bloqueio de conta


@dataclass
class BankingCase:
    """Caso bancário para avaliação governada pelo pipeline R1/R2/R3.

    Attributes:
        case_id: Identificador único do caso (para rastreabilidade).
        operation_type: Tipo da operação (``OperationType``).
        amount: Valor em BRL da operação (0 se não aplicável).
        client_score: Score de crédito do cliente (0–1000 Serasa/SCR).
        income: Renda mensal declarada em BRL.
        risk_score: Score de risco interno (0.0–1.0, calculado por aurix-ml).
        context: Dados adicionais específicos da operação (dict livre).
        client_id: ID do cliente (opcional — não logar em prod com PII).
        description: Descrição livre do caso para o LLM.
    """

    case_id: str
    operation_type: OperationType | str = OperationType.CREDIT_APPROVAL
    amount: float = 0.0
    client_score: int = 0          # 0–1000 (Serasa/SCR)
    income: float = 0.0            # renda mensal BRL
    risk_score: float = 0.0        # 0.0–1.0 (score interno aurix-ml)
    context: Dict[str, Any] = field(default_factory=dict)
    client_id: Optional[str] = None
    description: Optional[str] = None

    # Métricas derivadas — preenchidas automaticamente
    debt_to_income: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if isinstance(self.operation_type, str):
            try:
                self.operation_type = OperationType(self.operation_type)
            except ValueError:
                pass  # mantém como string para tipos customizados
        if self.income > 0 and self.amount > 0:
            self.debt_to_income = round(self.amount / self.income, 4)

    def to_prompt(self) -> str:
        """Serializa o caso para texto estruturado enviado ao LLM."""
        lines = [
            f"CASO ID: {self.case_id}",
            f"TIPO DE OPERAÇÃO: {self.operation_type}",
            f"VALOR: R$ {self.amount:,.2f}",
            f"SCORE DE CRÉDITO: {self.client_score}/1000",
            f"RENDA MENSAL: R$ {self.income:,.2f}",
            f"COMPROMETIMENTO DE RENDA: {self.debt_to_income:.1%}",
            f"SCORE DE RISCO INTERNO: {self.risk_score:.2f}/1.00",
        ]
        if self.description:
            lines.append(f"DESCRIÇÃO: {self.description}")
        if self.context:
            lines.append("CONTEXTO ADICIONAL:")
            for k, v in self.context.items():
                lines.append(f"  - {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "operation_type": str(self.operation_type),
            "amount": self.amount,
            "client_score": self.client_score,
            "income": self.income,
            "risk_score": self.risk_score,
            "debt_to_income": self.debt_to_income,
            "context": self.context,
        }
