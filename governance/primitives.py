# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Primitivas do pipeline R2 — hard gates, CEFL, I6Q, entropy, ambiguity gate.

Cada primitiva é independente e testável isoladamente.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from aurix_ml.governance.case import BankingCase, Decision
from aurix_ml.llm.base import LLMClient, Message

logger = logging.getLogger("aurix_ml.governance.primitives")


# ===========================================================================
# 1. HARD GATES — Verificações determinísticas pré-LLM
# ===========================================================================

@dataclass
class HardGateResult:
    triggered: bool
    gate_id: str
    forced_decision: Decision
    rationale: str


def evaluate_hard_gates(case: BankingCase) -> Optional[HardGateResult]:
    """Avalia regras determinísticas que forçam decisão sem chamar o LLM.

    Ordem de avaliação: score mínimo → limite de comprometimento →
    valor máximo → risco crítico → inadimplência grave.

    Returns:
        ``HardGateResult`` se algum gate disparou, ``None`` caso contrário.
    """
    ctx = case.context

    # G1: Score de crédito abaixo do mínimo absoluto → REJECT imediato
    if case.client_score < 300:
        return HardGateResult(
            triggered=True,
            gate_id="G1_SCORE_MINIMO",
            forced_decision=Decision.REJECT,
            rationale=(
                f"Score de crédito {case.client_score} abaixo do mínimo operacional "
                f"(300). Operação rejeitada automaticamente conforme política de crédito."
            ),
        )

    # G2: Comprometimento de renda > 80% → REJECT
    if case.debt_to_income > 0.80:
        return HardGateResult(
            triggered=True,
            gate_id="G2_COMPROMETIMENTO_RENDA",
            forced_decision=Decision.REJECT,
            rationale=(
                f"Comprometimento de renda {case.debt_to_income:.1%} excede limite "
                f"máximo de 80% (Resolução CMN nº 4.966/2021)."
            ),
        )

    # G3: Valor > R$500k → ESCALATE obrigatório (alçada superior)
    if case.amount > 500_000:
        return HardGateResult(
            triggered=True,
            gate_id="G3_VALOR_ALCADA_SUPERIOR",
            forced_decision=Decision.ESCALATE,
            rationale=(
                f"Valor R$ {case.amount:,.2f} excede alçada automática (R$ 500.000). "
                f"Requer aprovação de comitê de crédito."
            ),
        )

    # G4: Score de risco interno crítico → ESCALATE
    if case.risk_score >= 0.90:
        return HardGateResult(
            triggered=True,
            gate_id="G4_RISCO_CRITICO",
            forced_decision=Decision.ESCALATE,
            rationale=(
                f"Score de risco interno {case.risk_score:.2f} em nível crítico (≥0.90). "
                f"Operação sinalizada para revisão manual."
            ),
        )

    # G5: Inadimplência grave recente → REJECT
    inadimplencia_meses = ctx.get("meses_inadimplencia_recente", 0)
    if int(inadimplencia_meses) >= 6:
        return HardGateResult(
            triggered=True,
            gate_id="G5_INADIMPLENCIA_GRAVE",
            forced_decision=Decision.REJECT,
            rationale=(
                f"Inadimplência recente de {inadimplencia_meses} meses. "
                f"Política de crédito requer período de regularização mínimo de 12 meses."
            ),
        )

    # G6: Cliente em lista COAF/BACEN → ESCALATE obrigatório
    if ctx.get("lista_coaf", False) or ctx.get("lista_bacen_impedidos", False):
        return HardGateResult(
            triggered=True,
            gate_id="G6_LISTA_RESTRICAO_REGULATORIA",
            forced_decision=Decision.ESCALATE,
            rationale=(
                "Cliente identificado em lista regulatória (COAF/BACEN). "
                "Operação bloqueada e escalada para compliance (COAF Resolução 36/2021)."
            ),
        )

    return None  # Nenhum gate disparou — seguir para LLM


# ===========================================================================
# 2. ENTROPY E3 — Commit-reveal para trilha de auditoria
# ===========================================================================

@dataclass
class EntropyCommit:
    nonce: str
    nonce_hash: str
    timestamp_ns: int


@dataclass
class EntropyReveal:
    verified: bool
    nonce: str


def e3_commit() -> EntropyCommit:
    """Gera e commita um nonce criptográfico ANTES da chamada LLM.

    O hash do nonce é registrado nos metadados. Após a decisão, o nonce
    original é revelado e verificado — provando que não foi manipulado
    com base no output do LLM (resistência a cherry-picking).
    """
    nonce = secrets.token_hex(16)
    nonce_hash = hashlib.sha256(nonce.encode()).hexdigest()
    return EntropyCommit(
        nonce=nonce,
        nonce_hash=nonce_hash,
        timestamp_ns=time.time_ns(),
    )


def e3_reveal(commit: EntropyCommit) -> EntropyReveal:
    """Verifica que o nonce original corresponde ao hash commitado."""
    expected = hashlib.sha256(commit.nonce.encode()).hexdigest()
    verified = expected == commit.nonce_hash
    if not verified:
        logger.error("E3 commit-reveal FALHOU! Possível adulteração do nonce.")
    return EntropyReveal(verified=verified, nonce=commit.nonce)


# ===========================================================================
# 3. CEFL — Candidate Expansion, Freezing and Leveling
# ===========================================================================

_SYSTEM_PROMPT_R2 = """Você é um sistema de análise de risco bancário do Banco Aurix.
Avalie o caso bancário apresentado e retorne EXCLUSIVAMENTE um JSON válido com esta estrutura:

{
  "decision": "APPROVE" | "REJECT" | "DEFER" | "ESCALATE",
  "rationale": "justificativa detalhada em português (mínimo 50 palavras)",
  "pro_arguments": ["argumento favorável 1", "argumento favorável 2"],
  "con_arguments": ["argumento contrário 1", "argumento contrário 2"],
  "confidence": 0.0
}

Regras:
- APPROVE: operação dentro dos parâmetros de risco aceitável
- REJECT: operação viola política de crédito ou representa risco excessivo
- DEFER: informações insuficientes para decisão — especificar o que falta
- ESCALATE: requer análise humana por complexidade ou valor elevado
- confidence: 0.0 (incerto) a 1.0 (certeza total)
- pro_arguments: ao menos 2 argumentos com mínimo 10 palavras cada
- con_arguments: ao menos 2 argumentos com mínimo 10 palavras cada
Não inclua nenhum texto fora do JSON."""


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Extrai JSON da resposta do LLM com tolerância a formatação."""
    text = text.strip()
    # Remover blocos markdown ```json ... ```
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    # Remover vírgulas antes de } ou ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Tentar extrair apenas o bloco JSON
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _score_candidate(parsed: Dict[str, Any]) -> float:
    """Pontua a qualidade de um candidato CEFL (0.0–1.0)."""
    score = 0.0
    # Decisão válida
    valid_decisions = {"APPROVE", "REJECT", "DEFER", "ESCALATE"}
    if parsed.get("decision", "").upper() in valid_decisions:
        score += 0.30
    # Rationale com conteúdo suficiente
    rationale = parsed.get("rationale", "")
    if len(rationale.split()) >= 20:
        score += 0.25
    # Argumentos com qualidade mínima
    pro = parsed.get("pro_arguments", [])
    con = parsed.get("con_arguments", [])
    if isinstance(pro, list) and len(pro) >= 2:
        if all(len(a.split()) >= 5 for a in pro):
            score += 0.20
    if isinstance(con, list) and len(con) >= 2:
        if all(len(a.split()) >= 5 for a in con):
            score += 0.20
    # Confidence presente
    if isinstance(parsed.get("confidence"), (int, float)):
        score += 0.05
    return min(score, 1.0)


def generate_cefl_candidates(
    case: BankingCase,
    llm: LLMClient,
    n_candidates: int = 3,
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """Gera N candidatos de decisão e pontua cada um.

    CEFL = Candidate Expansion, Freezing and Leveling.
    Múltiplos candidatos com temperatura > 0 aumentam cobertura
    do espaço de decisão e reduzem viés de uma única amostragem.
    """
    user_msg = (
        "Avalie o seguinte caso bancário e retorne sua decisão em JSON:\n\n"
        + case.to_prompt()
    )
    messages: List[Message] = [
        {"role": "system", "content": _SYSTEM_PROMPT_R2},
        {"role": "user", "content": user_msg},
    ]

    candidates = []
    for i in range(n_candidates):
        try:
            resp = llm.chat(messages, temperature=temperature, max_tokens=800)
            parsed = _parse_llm_json(resp.content)
            score = _score_candidate(parsed)
            candidates.append({
                "index": i,
                "raw": resp.content,
                "parsed": parsed,
                "score": score,
                "tokens": resp.total_tokens,
            })
            logger.debug("CEFL candidato %d: decision=%s score=%.2f",
                         i, parsed.get("decision"), score)
        except Exception as exc:
            logger.warning("CEFL candidato %d falhou: %s", i, exc)
            candidates.append({"index": i, "raw": "", "parsed": {}, "score": 0.0, "tokens": 0})

    return candidates


def select_best_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Seleciona o candidato com maior score (Leveling)."""
    valid = [c for c in candidates if c["score"] > 0]
    if not valid:
        return candidates[0] if candidates else {"parsed": {}, "score": 0.0}
    return max(valid, key=lambda c: c["score"])


# ===========================================================================
# 4. I6Q — Verificação de Qualidade dos Argumentos
# ===========================================================================

@dataclass
class I6QResult:
    passed: bool
    details: str


def check_i6q(
    pro_args: List[str],
    con_args: List[str],
    min_args: int = 2,
    min_words_per_arg: int = 8,
) -> I6QResult:
    """Verifica qualidade mínima dos argumentos (I6Q gate).

    I6Q = Information Quality Gate.
    Garante que o LLM produziu argumentação fundamentada,
    não apenas decisões vazias ou argumentos de uma palavra.

    Returns:
        I6QResult com ``passed=True`` se todos os critérios foram atingidos.
    """
    if not isinstance(pro_args, list) or len(pro_args) < min_args:
        return I6QResult(False, f"Mínimo {min_args} argumentos favoráveis necessários.")
    if not isinstance(con_args, list) or len(con_args) < min_args:
        return I6QResult(False, f"Mínimo {min_args} argumentos contrários necessários.")

    for i, arg in enumerate(pro_args):
        if len(str(arg).split()) < min_words_per_arg:
            return I6QResult(
                False,
                f"Argumento favorável {i+1} muito curto "
                f"({len(str(arg).split())} palavras, mínimo {min_words_per_arg})."
            )
    for i, arg in enumerate(con_args):
        if len(str(arg).split()) < min_words_per_arg:
            return I6QResult(
                False,
                f"Argumento contrário {i+1} muito curto "
                f"({len(str(arg).split())} palavras, mínimo {min_words_per_arg})."
            )

    return I6QResult(True, "Qualidade de argumentos OK.")


# ===========================================================================
# 5. AMBIGUITY GATE K0_11 — Verificação pós-LLM
# ===========================================================================

def ambiguity_gate(
    case: BankingCase,
    theta_iota: float = 0.30,
    risk_escalation_threshold: float = 0.75,
) -> Optional[Decision]:
    """Gate de ambiguidade pós-LLM (K0_11).

    Força DEFER ou ESCALATE se o caso apresenta incompletude de informação
    ou risco elevado após análise do LLM.

    Args:
        case: Caso bancário avaliado.
        theta_iota: Limiar de completeness (abaixo → DEFER).
        risk_escalation_threshold: Limiar de risco para ESCALATE.

    Returns:
        ``Decision`` forçada ou ``None`` se o LLM pode decidir livremente.
    """
    # Calcular completeness do dossiê (campos preenchidos)
    total_fields = 6
    filled = sum([
        1 if case.amount > 0 else 0,
        1 if case.client_score > 0 else 0,
        1 if case.income > 0 else 0,
        1 if case.risk_score > 0 else 0,
        1 if bool(case.context) else 0,
        1 if bool(case.description) else 0,
    ])
    completeness = filled / total_fields

    if completeness < theta_iota:
        logger.info("Ambiguity gate K0_11: completeness=%.2f < %.2f → DEFER",
                    completeness, theta_iota)
        return Decision.DEFER

    if case.risk_score >= risk_escalation_threshold:
        logger.info("Ambiguity gate K0_11: risk_score=%.2f >= %.2f → ESCALATE",
                    case.risk_score, risk_escalation_threshold)
        return Decision.ESCALATE

    return None
