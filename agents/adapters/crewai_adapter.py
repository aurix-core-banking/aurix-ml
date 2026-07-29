# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Adapter CrewAI — expõe LLMClient Aurix como LLM do CrewAI.

CrewAI é um framework para orquestrar times de agentes AI com papéis
definidos (Roles). Útil no Aurix para workflows de:
- Análise multi-perspectiva de crédito (analista + jurídico + compliance)
- Geração e revisão de relatórios regulatórios
- Investigação de fraude (investigador + validador + decisor)

Requires:
    pip install crewai

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_crewai
    from crewai import Agent, Task, Crew, Process

    llm = create_llm({"provider": "ollama", "model": "llama3.2"})
    crew_llm = as_crewai(llm)

    analista = Agent(
        role="Analista de Crédito Sênior",
        goal="Avaliar o risco de crédito com base em dados financeiros",
        backstory="Especialista com 15 anos no setor bancário brasileiro",
        llm=crew_llm,
        verbose=True,
    )
    compliance = Agent(
        role="Especialista em Compliance",
        goal="Verificar conformidade com normas BACEN e LGPD",
        backstory="Especialista em regulamentação financeira brasileira",
        llm=crew_llm,
    )

    tarefa = Task(
        description="Analise o pedido de crédito do cliente ID 12345",
        agent=analista,
        expected_output="Parecer de risco com score e justificativa",
    )

    crew = Crew(agents=[analista, compliance], tasks=[tarefa], process=Process.sequential)
    resultado = crew.kickoff()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient


class AurixCrewAILLM:
    """Wrapper que adapta LLMClient Aurix para o framework CrewAI.

    CrewAI espera um objeto com método ``call()`` ou compatível com
    a interface LiteLLM. Este wrapper implementa ambos os padrões.
    """

    def __init__(self, client: LLMClient, **defaults: Any):
        self._client = client
        self._defaults = defaults
        self.model = f"{client.provider}/{client.model}"  # formato LiteLLM
        self.temperature = defaults.get("temperature", 0.7)
        self.max_tokens = defaults.get("max_tokens", 1024)

    def call(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Interface principal usada pelo CrewAI internamente."""
        merged = {**self._defaults, **kwargs}
        resp = self._client.chat(
            messages,
            temperature=merged.get("temperature", self.temperature),
            max_tokens=merged.get("max_tokens", self.max_tokens),
        )
        return resp.content

    def __call__(self, messages: Any, **kwargs: Any) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.call(messages, **kwargs)

    # Compatibilidade com duck-typing do CrewAI
    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True


def as_crewai(client: LLMClient, **kwargs: Any) -> AurixCrewAILLM:
    """Adapta um LLMClient Aurix para uso em times de agentes CrewAI.

    Args:
        client: Qualquer LLMClient criado via ``create_llm()``.
        **kwargs: Defaults para temperature, max_tokens, etc.

    Returns:
        Wrapper compatível com a interface LLM do CrewAI.
    """
    return AurixCrewAILLM(client, **kwargs)
