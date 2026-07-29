# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Regime R1 — Text-Only: LLM decide com system prompt de governança.

O regime mais simples: uma única chamada LLM com prompt de governança.
Adequado para: consultas informativas, análises de baixo risco, triagem.

Não deve ser usado para: aprovação de crédito > R$10k, investigações AML,
decisões que exigem trilha de auditoria BACEN obrigatória (usar R2).
"""

from __future__ import annotations

import logging
import time

from aurix_ml.governance.case import BankingCase, Decision
from aurix_ml.governance.primitives import _parse_llm_json, _score_candidate
from aurix_ml.governance.regime import GovernanceRegime
from aurix_ml.governance.result import DecisionResult
from aurix_ml.llm.base import LLMClient

logger = logging.getLogger("aurix_ml.governance.r1")

_R1_SYSTEM = """Você é analista de risco do Banco Aurix.
Avalie o caso e retorne JSON: {"decision": "APPROVE"|"REJECT"|"DEFER"|"ESCALATE",
"rationale": "...", "pro_arguments": [...], "con_arguments": [...], "confidence": 0.0}"""


class R1TextOnly(GovernanceRegime):
    """Regime R1 — Decisão LLM com system prompt governado.

    Sem hard gates, sem multi-candidatos, sem entropy. Ideal para casos
    de baixo risco onde velocidade é prioritária.
    """

    def __init__(self, llm: LLMClient, temperature: float = 0.5) -> None:
        super().__init__(llm)
        self._temperature = temperature

    @property
    def regime_name(self) -> str:
        return "R1"

    def decide(self, case: BankingCase) -> DecisionResult:
        start = time.perf_counter() * 1000
        messages = [
            {"role": "system", "content": _R1_SYSTEM},
            {"role": "user", "content": case.to_prompt()},
        ]
        resp = self._llm.chat(messages, temperature=self._temperature, max_tokens=600)
        elapsed = time.perf_counter() * 1000 - start

        parsed = _parse_llm_json(resp.content)
        decision_str = str(parsed.get("decision", "DEFER")).upper()
        try:
            decision = Decision(decision_str)
        except ValueError:
            decision = Decision.DEFER

        return DecisionResult(
            case_id=case.case_id,
            regime=self.regime_name,
            decision=decision,
            rationale=parsed.get("rationale", resp.content),
            pro_arguments=parsed.get("pro_arguments", []),
            con_arguments=parsed.get("con_arguments", []),
            i6q_passed=True,  # R1 não verifica I6Q
            processing_time_ms=elapsed,
            tokens_used=resp.total_tokens,
        )
