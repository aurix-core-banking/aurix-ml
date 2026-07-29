# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Contratos base do cliente LLM da Aurix Platform.

Toda a plataforma depende apenas desta interface — nunca de um SDK de vendor
específico. Isso permite trocar o provider (OpenAI, Bedrock, Gemini, gateway
interno) sem alterar código de negócio.

Padrão arquitetural originalmente publicado pelo Santander AI Lab (llm_bridge,
Apache-2.0). Adaptado para o contexto da Aurix Platform com extensões
para rastreabilidade regulatória (BACEN/LGPD).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Mensagem de chat: {"role": "system|user|assistant", "content": "..."}
Message = Dict[str, str]


@dataclass
class LLMResponse:
    """Resposta normalizada retornada por qualquer provider.

    Todos os campos de custo e latência são obrigatórios para permitir
    rastreamento de custos por operação (requisito de governança interno).
    """

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    provider: str = "unknown"
    # Resposta bruta do SDK — nunca logar, pode conter PII
    raw: Any = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Total de tokens consumidos (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimativa de custo em USD. Override no provider para valores reais."""
        return 0.0


def split_system_messages(
    messages: List[Message],
) -> Tuple[Optional[str], List[Message]]:
    """Separa mensagens de sistema das de usuário/assistente.

    Útil para providers (Bedrock, Gemini) que recebem o system prompt
    como campo separado ao invés de inline na lista de mensagens.

    Returns:
        Tupla (system_text, turns) onde system_text concatena todos os
        blocos ``role=system`` e turns mantém apenas user/assistant.
    """
    system_parts: List[str] = []
    turns: List[Message] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            turns.append({"role": role, "content": content})
    system_text = "\n\n".join(system_parts) if system_parts else None
    return system_text, turns


class LLMClient(ABC):
    """Cliente LLM vendor-neutral da Aurix Platform.

    Implementações devem fornecer apenas :meth:`chat` e a property
    :attr:`model`. O método :meth:`complete` é um wrapper de conveniência
    para chamadas single-turn.

    Exemplo de uso em serviços de negócio::

        from aurix_ml.llm import create_llm

        llm = create_llm(config)  # config vem do application.yml

        resp = llm.complete(
            "Avalie o risco da operação...",
            system="Você é analista de risco sênior do banco Aurix."
        )
        logger.info("Decisão LLM: %s (%d tokens)", resp.content[:100], resp.total_tokens)
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Identificador do modelo (ex: ``gpt-4o-mini``, ``amazon.nova-lite-v1:0``)."""

    @property
    def provider(self) -> str:
        """Nome curto do provider (ex: ``openai``, ``bedrock``, ``mock``)."""
        return "unknown"

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Envia lista de mensagens de chat e retorna resposta normalizada."""

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        """Wrapper single-turn: constrói messages a partir de prompt/system."""
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
