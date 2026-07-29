# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Regime R3 — Adaptive: Roteamento dinâmico entre R1 (baixo custo) e R2 (alta governança).

Evita o custo e latência desnecessários do R2 para transações triviais,
enquanto garante conformidade e rigor do R2 para casos de maior risco ou valor.
"""

from __future__ import annotations

import logging
from typing import Optional

from aurix_ml.governance.case import BankingCase
from aurix_ml.governance.r1_text_only import R1TextOnly
from aurix_ml.governance.r2_mechanical import R2Mechanical
from aurix_ml.governance.regime import GovernanceRegime
from aurix_ml.governance.result import DecisionResult
from aurix_ml.llm.base import LLMClient

logger = logging.getLogger("aurix_ml.governance.r3")


class R3Adaptive(GovernanceRegime):
    """Regime R3 — Governança Adaptativa baseada em risco e alçada de valor."""

    def __init__(
        self,
        llm: LLMClient,
        r1_regime: Optional[R1TextOnly] = None,
        r2_regime: Optional[R2Mechanical] = None,
        amount_threshold: float = 50_000.0,
        risk_score_threshold: float = 0.50,
        client_score_threshold: int = 500,
    ) -> None:
        """Inicializa o regime adaptativo.

        Args:
            llm: Cliente de LLM compartilhado.
            r1_regime: Instância customizada do R1 (opcional).
            r2_regime: Instância customizada do R2 (opcional).
            amount_threshold: Valor da transação em BRL a partir do qual R2 é obrigatório.
            risk_score_threshold: Score de risco a partir do qual R2 é obrigatório.
            client_score_threshold: Score de crédito abaixo do qual R2 é obrigatório.
        """
        super().__init__(llm)
        self._r1 = r1_regime or R1TextOnly(llm)
        self._r2 = r2_regime or R2Mechanical(llm)
        self._amount_threshold = amount_threshold
        self._risk_score_threshold = risk_score_threshold
        self._client_score_threshold = client_score_threshold

    @property
    def regime_name(self) -> str:
        return "R3"

    def decide(self, case: BankingCase) -> DecisionResult:
        # Regras de roteamento dinâmico
        use_r2 = False
        reasons = []

        if case.amount >= self._amount_threshold:
            use_r2 = True
            reasons.append(f"valor R$ {case.amount:,.2f} >= limite R$ {self._amount_threshold:,.2f}")

        if case.risk_score >= self._risk_score_threshold:
            use_r2 = True
            reasons.append(f"score de risco {case.risk_score:.2f} >= limite {self._risk_score_threshold:.2f}")

        if 0 < case.client_score < self._client_score_threshold:
            use_r2 = True
            reasons.append(f"score de crédito {case.client_score} < limite {self._client_score_threshold}")

        if use_r2:
            logger.info("R3: Roteando para R2 devido a: %s", ", ".join(reasons))
            res = self._r2.decide(case)
        else:
            logger.info("R3: Roteando para R1 (baixo risco/valor)")
            res = self._r1.decide(case)

        # Retorna o resultado envelopado no regime R3
        new_metadata = dict(res.metadata)
        new_metadata["adaptive_routed_to"] = res.regime
        new_metadata["adaptive_reasons"] = reasons

        return DecisionResult(
            case_id=res.case_id,
            regime=self.regime_name,
            decision=res.decision,
            rationale=res.rationale,
            pro_arguments=res.pro_arguments,
            con_arguments=res.con_arguments,
            gates_triggered=res.gates_triggered,
            i6q_passed=res.i6q_passed,
            entropy_nonce=res.entropy_nonce,
            processing_time_ms=res.processing_time_ms,
            tokens_used=res.tokens_used,
            metadata=new_metadata,
        )
