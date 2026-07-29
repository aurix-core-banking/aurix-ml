# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Regime R2 — Mechanical: Pipeline determinístico de 6 etapas de governança.

Implementa um pipeline de tomada de decisão com gates de segurança:
1. Hard Gates (pré-LLM) — regras determinísticas que podem rejeitar ou escalar direto.
2. Entropy Commit (E3) — geração de nonce para provar não-manipulação.
3. CEFL (Candidate Expansion) — gera N decisões candidatas e seleciona a melhor.
4. I6Q Quality Gate — garante riqueza mínima de argumentação com retries se necessário.
5. Ambiguity Gate (K0_11) — força adiamento ou escalada se houver pouca info ou risco alto.
6. Entropy Reveal (E3) — revela e verifica o nonce na trilha de auditoria.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.governance.case import BankingCase, Decision
from aurix_ml.governance.primitives import (
    evaluate_hard_gates,
    e3_commit,
    e3_reveal,
    generate_cefl_candidates,
    select_best_candidate,
    check_i6q,
    ambiguity_gate,
    _parse_llm_json,
    _SYSTEM_PROMPT_R2
)
from aurix_ml.governance.regime import GovernanceRegime
from aurix_ml.governance.result import DecisionResult
from aurix_ml.llm.base import LLMClient, Message

logger = logging.getLogger("aurix_ml.governance.r2")


class R2Mechanical(GovernanceRegime):
    """Regime R2 — Governança Mecânica rigorosa para transações financeiras de alto risco."""

    def __init__(
        self,
        llm: LLMClient,
        n_candidates: int = 3,
        temperature: float = 0.7,
        theta_iota: float = 0.30,
        risk_escalation_threshold: float = 0.75,
        max_i6q_retries: int = 2,
    ) -> None:
        super().__init__(llm)
        self._n_candidates = n_candidates
        self._temperature = temperature
        self._theta_iota = theta_iota
        self._risk_escalation_threshold = risk_escalation_threshold
        self._max_i6q_retries = max_i6q_retries

    @property
    def regime_name(self) -> str:
        return "R2"

    def decide(self, case: BankingCase) -> DecisionResult:
        start_time = time.perf_counter() * 1000
        tokens_used = 0
        gates_triggered: List[str] = []
        metadata: Dict[str, Any] = {}

        # 1. Hard Gates (Pré-LLM)
        hard_gate_res = evaluate_hard_gates(case)
        if hard_gate_res is not None:
            elapsed = time.perf_counter() * 1000 - start_time
            metadata["hard_gate_override"] = True
            metadata["hard_gate_id"] = hard_gate_res.gate_id
            return DecisionResult(
                case_id=case.case_id,
                regime=self.regime_name,
                decision=hard_gate_res.forced_decision,
                rationale=hard_gate_res.rationale,
                gates_triggered=[hard_gate_res.gate_id],
                i6q_passed=True,
                entropy_nonce=None,
                processing_time_ms=elapsed,
                tokens_used=0,
                metadata=metadata,
            )

        # 2. Entropy Commit (E3)
        commit = e3_commit()

        # 3. CEFL (Candidate Expansion, Freezing and Leveling)
        candidates = generate_cefl_candidates(
            case, self._llm, n_candidates=self._n_candidates, temperature=self._temperature
        )
        for c in candidates:
            tokens_used += c.get("tokens", 0)

        best_cand = select_best_candidate(candidates)
        parsed = best_cand.get("parsed") or {}
        decision_str = str(parsed.get("decision", "DEFER")).upper()
        try:
            decision = Decision(decision_str)
        except ValueError:
            decision = Decision.DEFER

        rationale = parsed.get("rationale", "Nenhuma justificativa válida produzida.")
        pro_arguments = parsed.get("pro_arguments", [])
        con_arguments = parsed.get("con_arguments", [])
        confidence = parsed.get("confidence", 0.0)

        metadata["cefl_candidates_generated"] = len(candidates)
        metadata["best_candidate_score"] = best_cand.get("score", 0.0)
        metadata["confidence"] = confidence

        # 4. I6Q Quality Gate + Retries
        i6q_res = check_i6q(pro_arguments, con_arguments)
        retries = 0
        current_raw = best_cand.get("raw", "")

        while not i6q_res.passed and retries < self._max_i6q_retries:
            retries += 1
            feedback_msg = (
                f"Sua resposta anterior falhou no portão de qualidade de informação (I6Q):\n"
                f"Erro: {i6q_res.details}\n"
                f"Por favor, reformule sua resposta fornecendo argumentos mais longos, "
                f"claros e detalhados em português (mínimo 10 palavras por argumento, "
                f"pelo menos 2 argumentos pró e 2 contra). Retorne APENAS o JSON estruturado."
            )

            retry_messages: List[Message] = [
                {"role": "system", "content": _SYSTEM_PROMPT_R2},
                {"role": "user", "content": case.to_prompt()},
                {"role": "assistant", "content": current_raw},
                {"role": "user", "content": feedback_msg},
            ]

            try:
                resp = self._llm.chat(retry_messages, temperature=self._temperature, max_tokens=800)
                tokens_used += resp.total_tokens
                current_raw = resp.content
                parsed_retry = _parse_llm_json(resp.content)
                if parsed_retry:
                    pro_retry = parsed_retry.get("pro_arguments", [])
                    con_retry = parsed_retry.get("con_arguments", [])
                    i6q_res = check_i6q(pro_retry, con_retry)
                    if i6q_res.passed or parsed_retry.get("decision"):
                        parsed = parsed_retry
                        decision_str = str(parsed.get("decision", "DEFER")).upper()
                        try:
                            decision = Decision(decision_str)
                        except ValueError:
                            decision = Decision.DEFER
                        rationale = parsed.get("rationale", rationale)
                        pro_arguments = pro_retry
                        con_arguments = con_retry
                        confidence = parsed.get("confidence", confidence)
            except Exception as exc:
                logger.warning("Erro durante tentativa %d de retry I6Q: %s", retries, exc)

        metadata["i6q_retries"] = retries
        i6q_passed = i6q_res.passed

        if not i6q_passed:
            logger.warning("Decisão para caso %s falhou no portão I6Q após %d retries. Forçando ESCALATE.", case.case_id, retries)
            decision = Decision.ESCALATE
            rationale = f"[FORÇADO ESCALATE — FALHA I6Q] {rationale}"

        # 5. Ambiguity Gate K0_11 (Pós-LLM)
        ambiguity_override = ambiguity_gate(
            case, theta_iota=self._theta_iota, risk_escalation_threshold=self._risk_escalation_threshold
        )
        if ambiguity_override is not None:
            gates_triggered.append("K0_11_AMBIGUITY_GATE")
            decision = ambiguity_override
            rationale = f"[FORÇADO {decision.value} — AMBIGUITY GATE] {rationale}"

        # 6. Entropy Reveal (E3)
        reveal = e3_reveal(commit)
        metadata["e3_verified"] = reveal.verified

        elapsed = time.perf_counter() * 1000 - start_time

        return DecisionResult(
            case_id=case.case_id,
            regime=self.regime_name,
            decision=decision,
            rationale=rationale,
            pro_arguments=pro_arguments,
            con_arguments=con_arguments,
            gates_triggered=gates_triggered,
            i6q_passed=i6q_passed,
            entropy_nonce=commit.nonce,
            processing_time_ms=elapsed,
            tokens_used=tokens_used,
            metadata=metadata,
        )
