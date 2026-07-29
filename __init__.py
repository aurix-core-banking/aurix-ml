# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""aurix-ml — Camada de Machine Learning e IA da Aurix Platform.

Pacotes disponíveis
-------------------
``aurix_ml.llm``
    Interface vendor-neutral para LLMs com 13 providers (Ollama, HuggingFace,
    OpenAI, Bedrock, Gemini, Mock, etc.).

``aurix_ml.agents``
    Acoplamento com frameworks open-source: LangChain, LlamaIndex, CrewAI,
    AutoGen, Haystack. Inclui ``AgentRunner`` com retry e observabilidade.

``aurix_ml.governance``
    Governança mecânica para decisões LLM de alto risco (crédito, compliance,
    fraude). Pipeline R2: hard gates → CEFL → I6Q → ambiguity gate → auditoria.

Uso rápido::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import AgentRunner, AgentTask
    from aurix_ml.governance import R2Mechanical, BankingCase

    # 1. Provider — troca sem alterar código de negócio
    llm = create_llm({"provider": "ollama", "model": "mistral:7b"})

    # 2. Tarefa simples via AgentRunner
    runner = AgentRunner(llm)
    result = runner.run(AgentTask(name="análise", prompt="Avalie o risco..."))

    # 3. Decisão governada (crédito, compliance)
    regime = R2Mechanical(llm)
    decision = regime.decide(BankingCase(case_id="C001", ...))
"""

from aurix_ml.llm import create_llm, available_providers, LLMClient, LLMResponse
from aurix_ml.agents import AgentRunner, AgentTask, AgentResult
from aurix_ml.governance import (
    BankingCase,
    Decision,
    OperationType,
    DecisionResult,
    GovernanceRegime,
    R1TextOnly,
    R2Mechanical,
    R3Adaptive,
)

__version__ = "1.0.0"

__all__ = [
    # LLM
    "create_llm",
    "available_providers",
    "LLMClient",
    "LLMResponse",
    # Agents
    "AgentRunner",
    "AgentTask",
    "AgentResult",
    # Governance
    "BankingCase",
    "Decision",
    "OperationType",
    "DecisionResult",
    "GovernanceRegime",
    "R1TextOnly",
    "R2Mechanical",
    "R3Adaptive",
]
