# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Interface base dos regimes de governança."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aurix_ml.governance.case import BankingCase
from aurix_ml.governance.result import DecisionResult
from aurix_ml.llm.base import LLMClient


class GovernanceRegime(ABC):
    """Interface base para todos os regimes de governança.

    Todo regime recebe um ``BankingCase`` e um ``LLMClient`` e retorna
    um ``DecisionResult`` com decisão, rationale e trilha de auditoria.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    @abstractmethod
    def regime_name(self) -> str:
        """Identificador do regime (R1, R2, R3)."""

    @abstractmethod
    def decide(self, case: BankingCase) -> DecisionResult:
        """Processa o caso e retorna a decisão governada."""
