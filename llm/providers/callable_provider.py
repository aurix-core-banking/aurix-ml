# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider callable — wraps qualquer função Python como cliente LLM.

Útil para integrar modelos customizados, gateways internos do banco,
ou qualquer callable que receba mensagens e retorne texto.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

from aurix_ml.llm.base import LLMClient, LLMResponse, Message


class CallableClient(LLMClient):
    """Adapta qualquer função ``(messages: list) -> str`` como LLMClient."""

    def __init__(self, fn: Callable[[List[Message]], str], model: str = "callable"):
        if not callable(fn):
            raise TypeError(f"'callable' provider requer um callable, recebeu: {type(fn)}")
        self._fn = fn
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "callable"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        content = self._fn(messages)
        latency = time.perf_counter() * 1000 - start

        return LLMResponse(
            content=str(content),
            model=self._model,
            provider="callable",
            prompt_tokens=0,
            completion_tokens=0,
            finish_reason="stop",
            latency_ms=latency,
        )


def build(config: Dict[str, Any]) -> CallableClient:
    fn = config.get("callable")
    if fn is None:
        raise ValueError("Provider 'callable' requer a chave 'callable' no config.")
    return CallableClient(fn=fn, model=config.get("model", "callable"))
