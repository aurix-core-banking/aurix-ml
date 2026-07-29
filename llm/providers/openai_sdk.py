# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider OpenAI SDK — OpenAI, Azure OpenAI e qualquer API compatível.

Compatível com:
- OpenAI (api.openai.com)
- Azure OpenAI Service
- LM Studio (http://localhost:1234)
- vLLM (self-hosted, OpenAI-compatible)
- Together AI, Groq, Perplexity, etc.

Requires:
    pip install openai>=1.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message

logger = logging.getLogger("aurix_ml.llm.providers.openai_sdk")

# Custo estimado em USD por 1M tokens (atualizar conforme pricing OpenAI)
_COST_PER_MILLION = {
    "gpt-4o": (2.50, 10.00),          # (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
}


class OpenAIClient(LLMClient):
    """Cliente para OpenAI e APIs OpenAI-compatíveis.

    Args:
        model: Identificador do modelo (ex: ``gpt-4o-mini``).
        api_key: API key. Se None, usa ``OPENAI_API_KEY`` do ambiente.
        base_url: URL base. Override para Azure, LM Studio, vLLM, etc.
        organization: Organização OpenAI (opcional).
        timeout: Timeout em segundos.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 60,
        **kwargs: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Provider 'openai' requer: pip install openai") from exc

        self._model = model
        client_kwargs: Dict[str, Any] = {"timeout": timeout}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        self._client = OpenAI(**client_kwargs)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "openai"

    @property
    def estimated_cost_usd(self) -> float:
        return 0.0  # calculado por chamada em chat()

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
        except Exception as exc:
            logger.error("Erro na chamada OpenAI (%s): %s", self._model, exc)
            raise

        return LLMResponse(
            content=content,
            model=self._model,
            provider="openai",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            latency_ms=time.perf_counter() * 1000 - start,
            raw=None,
        )


def build(config: Dict[str, Any]) -> OpenAIClient:
    return OpenAIClient(
        model=config.get("model", "gpt-4o-mini"),
        api_key=config.get("api_key"),
        base_url=config.get("base_url"),
        organization=config.get("organization"),
        timeout=config.get("timeout", 60),
    )
