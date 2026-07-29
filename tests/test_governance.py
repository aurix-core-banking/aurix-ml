# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitários do pipeline de governança de IA — R1, R2, R3 e primitivas."""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from aurix_ml.governance.case import BankingCase, Decision, OperationType
from aurix_ml.governance.result import DecisionResult
from aurix_ml.governance.r1_text_only import R1TextOnly
from aurix_ml.governance.r2_mechanical import R2Mechanical
from aurix_ml.governance.r3_adaptive import R3Adaptive
from aurix_ml.governance.primitives import (
    evaluate_hard_gates,
    e3_commit,
    e3_reveal,
    check_i6q,
    ambiguity_gate,
)
from aurix_ml.llm.base import LLMResponse
from aurix_ml.llm.providers.mock import MockClient


# ===========================================================================
# 1. Testes de Primitivas
# ===========================================================================

class TestGovernancePrimitives:
    def test_evaluate_hard_gates_score_minimo(self):
        case = BankingCase(case_id="C1", client_score=250)
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.REJECT
        assert "G1_SCORE_MINIMO" in res.gate_id

    def test_evaluate_hard_gates_comprometimento_renda(self):
        # amount=9000, income=10000 -> debt_to_income = 0.90 (> 80%)
        case = BankingCase(case_id="C2", amount=9000, income=10000, client_score=600)
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.REJECT
        assert "G2_COMPROMETIMENTO_RENDA" in res.gate_id

    def test_evaluate_hard_gates_valor_alcada_superior(self):
        case = BankingCase(case_id="C3", amount=600_000, income=1_000_000, client_score=700)
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.ESCALATE
        assert "G3_VALOR_ALCADA_SUPERIOR" in res.gate_id

    def test_evaluate_hard_gates_risk_score(self):
        case = BankingCase(case_id="C4", risk_score=0.95, client_score=650)
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.ESCALATE
        assert "G4_RISCO_CRITICO" in res.gate_id

    def test_evaluate_hard_gates_inadimplencia_grave(self):
        case = BankingCase(case_id="C5", client_score=550, context={"meses_inadimplencia_recente": 8})
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.REJECT
        assert "G5_INADIMPLENCIA_GRAVE" in res.gate_id

    def test_evaluate_hard_gates_lista_coaf(self):
        case = BankingCase(case_id="C6", client_score=600, context={"lista_coaf": True})
        res = evaluate_hard_gates(case)
        assert res is not None
        assert res.forced_decision == Decision.ESCALATE
        assert "G6_LISTA_RESTRICAO_REGULATORIA" in res.gate_id

    def test_evaluate_hard_gates_pass(self):
        case = BankingCase(case_id="C7", amount=5000, income=10000, client_score=700, risk_score=0.2)
        res = evaluate_hard_gates(case)
        assert res is None

    def test_e3_commit_reveal(self):
        commit = e3_commit()
        assert commit.nonce is not None
        assert len(commit.nonce_hash) == 64
        
        reveal = e3_reveal(commit)
        assert reveal.verified is True
        assert reveal.nonce == commit.nonce

    def test_check_i6q(self):
        # Argumentos válidos (pelo menos 2 pro e 2 con, >= 8 palavras cada)
        pro = [
            "O cliente possui renda mensal regular comprovada e suficiente.",
            "O histórico de pagamento do cliente é excelente no SCR."
        ]
        con = [
            "O valor da operação solicitado é ligeiramente acima da média.",
            "O tempo de relacionamento do cliente com o banco é curto."
        ]
        res = check_i6q(pro, con)
        assert res.passed is True

        # Falha: poucos argumentos
        assert check_i6q(["Um argumento curto"], con).passed is False

        # Falha: argumento muito curto (menos de 8 palavras)
        pro_curto = ["Renda boa", "Histórico de pagamento excelente no SCR do cliente."]
        assert check_i6q(pro_curto, con).passed is False

    def test_ambiguity_gate(self):
        # Caso completo
        case_completo = BankingCase(
            case_id="C_COMPLETO",
            amount=5000.0,
            client_score=600,
            income=8000.0,
            risk_score=0.2,
            context={"foo": "bar"},
            description="Um caso bancario de teste completo."
        )
        assert ambiguity_gate(case_completo) is None

        # Caso incompleto (completeness < 0.3)
        case_incompleto = BankingCase(case_id="C_INCOMPLETO")
        assert ambiguity_gate(case_incompleto) == Decision.DEFER

        # Caso com risco muito alto
        case_risco = BankingCase(
            case_id="C_RISCO",
            amount=5000.0,
            client_score=600,
            income=8000.0,
            risk_score=0.85,
            context={"foo": "bar"},
            description="Caso com alto score de risco interno."
        )
        assert ambiguity_gate(case_risco) == Decision.ESCALATE


# ===========================================================================
# 2. Testes de Regimes (R1, R2, R3)
# ===========================================================================

class TestGovernanceRegimes:
    def test_r1_text_only(self):
        valid_json = {
            "decision": "APPROVE",
            "rationale": "Renda e score excelentes.",
            "pro_arguments": ["Renda boa", "Score bom"],
            "con_arguments": ["Risco baixo", "Nenhum"],
            "confidence": 0.9
        }
        llm = MockClient(response=json.dumps(valid_json))
        regime = R1TextOnly(llm)
        case = BankingCase(case_id="R1_TEST", client_score=750)
        res = regime.decide(case)
        
        assert res.decision == Decision.APPROVE
        assert res.regime == "R1"
        assert "excelentes" in res.rationale
        assert res.tokens_used > 0
        assert res.audit_nonce is not None

    def test_r2_mechanical_success(self):
        valid_json = {
            "decision": "APPROVE",
            "rationale": "Justificativa detalhada com mais de vinte palavras para passar no score do candidato e na validação padrão.",
            "pro_arguments": [
                "O cliente tem renda mensal excelente e comprovada em carteira.",
                "O score de crédito é extremamente elevado e sem apontamentos."
            ],
            "con_arguments": [
                "O montante solicitado é considerável para o primeiro empréstimo.",
                "O tempo de conta corrente no banco ainda é menor que um ano."
            ],
            "confidence": 0.95
        }
        llm = MockClient(response=json.dumps(valid_json))
        regime = R2Mechanical(llm, n_candidates=2)
        case = BankingCase(
            case_id="R2_TEST_OK",
            amount=10_000,
            income=20_000,
            client_score=750,
            risk_score=0.1,
            description="Descrição do caso bancário do cliente."
        )
        res = regime.decide(case)

        assert res.decision == Decision.APPROVE
        assert res.regime == "R2"
        assert res.i6q_passed is True
        assert res.entropy_nonce is not None
        assert res.metadata["e3_verified"] is True
        assert res.metadata["cefl_candidates_generated"] == 2

    def test_r2_mechanical_hard_gate_override(self):
        llm = MockClient(response="{}")
        regime = R2Mechanical(llm)
        # Score 200 dispara hard gate G1_SCORE_MINIMO
        case = BankingCase(case_id="R2_HG", client_score=200)
        res = regime.decide(case)

        assert res.decision == Decision.REJECT
        assert res.hard_gate_override is True
        assert "G1_SCORE_MINIMO" in res.gates_triggered
        assert res.tokens_used == 0

    def test_r2_mechanical_i6q_failure_escapes(self):
        # Retorna JSON inválido para I6Q (argumentos curtos)
        invalid_i6q_json = {
            "decision": "APPROVE",
            "rationale": "Razoavel",
            "pro_arguments": ["Renda ok"],
            "con_arguments": ["Nenhum"]
        }
        llm = MockClient(response=json.dumps(invalid_i6q_json))
        regime = R2Mechanical(llm, n_candidates=1, max_i6q_retries=1)
        case = BankingCase(
            case_id="R2_FAIL_I6Q",
            amount=10_000,
            income=20_000,
            client_score=750,
            risk_score=0.1,
            description="Descrição do caso bancário do cliente."
        )
        res = regime.decide(case)

        # Deve ser forçado a ESCALATE devido à falha do portão I6Q
        assert res.decision == Decision.ESCALATE
        assert res.i6q_passed is False
        assert "FALHA I6Q" in res.rationale

    def test_r3_adaptive_routing(self):
        # Mock para R1 e R2
        valid_json = {
            "decision": "APPROVE",
            "rationale": "Justificativa detalhada com mais de vinte palavras para passar no score do candidato e na validação padrão.",
            "pro_arguments": [
                "O cliente tem renda mensal excelente e comprovada em carteira.",
                "O score de crédito é extremamente elevado e sem apontamentos."
            ],
            "con_arguments": [
                "O montante solicitado é considerável para o primeiro empréstimo.",
                "O tempo de conta corrente no banco ainda é menor que um ano."
            ],
            "confidence": 0.95
        }
        llm = MockClient(response=json.dumps(valid_json))
        regime = R3Adaptive(llm, amount_threshold=50_000.0)

        # Caso de baixo valor (deve usar R1)
        case_r1 = BankingCase(
            case_id="CASE_LOW",
            amount=5000.0,
            client_score=600,
            income=8000.0,
            risk_score=0.2,
            description="Descrição do caso."
        )
        res_r1 = regime.decide(case_r1)
        assert res_r1.metadata["adaptive_routed_to"] == "R1"
        assert res_r1.regime == "R3"

        # Caso de alto valor (deve usar R2)
        case_r2 = BankingCase(
            case_id="CASE_HIGH",
            amount=100_000.0,
            client_score=600,
            income=200_000.0,
            risk_score=0.2,
            description="Descrição do caso."
        )
        res_r2 = regime.decide(case_r2)
        assert res_r2.metadata["adaptive_routed_to"] == "R2"
        assert res_r2.regime == "R3"
