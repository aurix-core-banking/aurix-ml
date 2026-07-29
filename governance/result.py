# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Resultado de uma decisão do pipeline de governança.

Estrutura rica com trilha de auditoria para conformidade BACEN.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from aurix_ml.governance.case import Decision


@dataclass
class DecisionResult:
    """Resultado estruturado de uma decisão governada.

    Todos os campos são imutáveis após criação — garante integridade
    da trilha de auditoria exigida pelo BACEN para decisões automatizadas.

    Attributes:
        case_id: ID do caso avaliado.
        regime: Nome do regime usado (R1, R2, R3).
        decision: Decisão final (APPROVE/REJECT/DEFER/ESCALATE).
        rationale: Justificativa detalhada gerada pelo LLM.
        pro_arguments: Lista de argumentos favoráveis.
        con_arguments: Lista de argumentos contrários.
        gates_triggered: Hard gates que foram acionados.
        i6q_passed: Se a verificação de qualidade de argumentos passou.
        entropy_nonce: Nonce de entropia para commit-reveal (auditoria).
        audit_nonce: Hash SHA-256 do nonce + case_id (trilha BACEN).
        processing_time_ms: Latência total do pipeline.
        tokens_used: Total de tokens consumidos.
        timestamp: Timestamp UTC da decisão.
        metadata: Metadados extras do pipeline.
    """

    case_id: str
    regime: str
    decision: Decision
    rationale: str
    pro_arguments: List[str] = field(default_factory=list)
    con_arguments: List[str] = field(default_factory=list)
    gates_triggered: List[str] = field(default_factory=list)
    i6q_passed: bool = True
    entropy_nonce: Optional[str] = None
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Preenchido automaticamente
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    audit_nonce: str = field(init=False, default="")

    def __post_init__(self) -> None:
        # Gerar hash de auditoria (case_id + nonce + decisão)
        audit_input = f"{self.case_id}:{self.entropy_nonce or ''}:{self.decision.value}"
        self.audit_nonce = hashlib.sha256(audit_input.encode()).hexdigest()[:32]

    @property
    def hard_gate_override(self) -> bool:
        """True se a decisão foi forçada por um hard gate (sem LLM)."""
        return bool(self.gates_triggered) and self.metadata.get("hard_gate_override", False)

    def to_audit_record(self) -> Dict[str, Any]:
        """Serializa para registro de auditoria — salvar no aurix-audit."""
        return {
            "case_id": self.case_id,
            "regime": self.regime,
            "decision": self.decision.value,
            "audit_nonce": self.audit_nonce,
            "gates_triggered": self.gates_triggered,
            "hard_gate_override": self.hard_gate_override,
            "i6q_passed": self.i6q_passed,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.to_audit_record(),
            "rationale": self.rationale,
            "pro_arguments": self.pro_arguments,
            "con_arguments": self.con_arguments,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"DecisionResult(case={self.case_id!r}, regime={self.regime!r}, "
            f"decision={self.decision.value!r}, gates={self.gates_triggered})"
        )
