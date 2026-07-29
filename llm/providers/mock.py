# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider mock — offline, determinístico, sem credenciais.

Ideal para testes unitários, CI/CD e desenvolvimento local sem
necessidade de credenciais de cloud ou modelos instalados.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _last_user(messages: List[Message]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[-1].get("content", "") if messages else ""


class MockClient(LLMClient):
    """Cliente determinístico in-process, sem rede e sem credenciais.

    Args:
        response: Texto fixo retornado para todas as chamadas.
        responder: Callable ``(messages) -> str`` — tem precedência.
        model: Identificador reportado pelo cliente.
    """

    def __init__(
        self,
        response: Optional[str] = None,
        responder: Optional[Callable[[List[Message]], str]] = None,
        model: str = "mock",
    ):
        self._response = response
        self._responder = responder
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "mock"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        start = time.perf_counter() * 1000
        if self._responder is not None:
            content = self._responder(messages)
        elif self._response is not None:
            content = self._response
        else:
            content = f"[mock] {_last_user(messages)}"
        latency = time.perf_counter() * 1000 - start

        prompt_text = " ".join(m.get("content", "") for m in messages)
        return LLMResponse(
            content=content,
            model=self._model,
            provider="mock",
            prompt_tokens=_estimate_tokens(prompt_text),
            completion_tokens=_estimate_tokens(content),
            finish_reason="stop",
            latency_ms=latency,
            raw=None,
        )


def build(config: Dict[str, Any]) -> MockClient:
    return MockClient(
        response=config.get("response"),
        responder=config.get("responder"),
        model=config.get("model", "mock"),
    )
