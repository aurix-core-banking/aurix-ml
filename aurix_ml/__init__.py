# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Pacote raiz ``aurix_ml`` (shim de namespace).

O repositório é armazenado no diretório ``aurix-ml`` (hífen), mas todo o
código (governance, llm, agents, testes, serving gRPC) importa via
``aurix_ml.*``. Este módulo expõe o namespace ``aurix_ml`` sem exigir
renomeação do diretório do monorepo: o caminho do pacote é estendido para a
raiz do repositório, de modo que ``aurix_ml.governance``, ``aurix_ml.llm`` e
``aurix_ml.agents`` resolvem para as pastas ``governance/``, ``llm/`` e
``agents/`` existentes.
"""

from __future__ import annotations

from pathlib import Path

_RAIZ = Path(__file__).resolve().parent.parent

# Garante que o diretório raiz do repositório esteja acessível para imports
# absolutos de subpacotes (ex.: ``from aurix_ml.governance.case import ...``).
import sys  # noqa: E402

if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

# Expande o caminho do pacote para a raiz do repositório: os subpacotes
# existentes (governance, llm, agents) passam a ser encontrados sob o
# namespace ``aurix_ml``.
if str(_RAIZ) not in __path__:
    __path__.append(str(_RAIZ))

from aurix_ml.llm import create_llm, available_providers, LLMClient, LLMResponse  # noqa: E402
from aurix_ml.agents import AgentRunner, AgentTask, AgentResult  # noqa: E402
from aurix_ml.governance import (  # noqa: E402
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
