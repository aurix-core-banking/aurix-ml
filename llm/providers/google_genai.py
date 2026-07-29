# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider Google Gemini — via SDK google-genai.

Suporta modelos Gemini:
- gemini-2.0-flash       (rápido, custo baixo)
- gemini-2.5-pro         (máxima qualidade)
- gemini-1.5-flash       (contexto longo, 1M tokens)

Requires:
    pip install google-genai>=1.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message, split_system_messages

logger = logging.getLogger("aurix_ml.llm.providers.google_genai")


class GoogleGenAIClient(LLMClient):
    """Cliente para Google Gemini via SDK google-genai.

    Args:
        model: Identificador do modelo (ex: ``gemini-2.0-flash``).
        api_key: API key do Google AI Studio. None = usa ``GOOGLE_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise ImportError(
                "Provider 'google' requer: pip install google-genai"
            ) from exc

        self._model = model
        self._genai_types = genai_types
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = genai.Client(**client_kwargs)

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        system_text, turns = split_system_messages(messages)

        # Converter para formato google-genai
        contents = []
        for m in turns:
            role = "user" if m["role"] == "user" else "model"
            contents.append(self._genai_types.Content(
                role=role,
                parts=[self._genai_types.Part(text=m["content"])],
            ))

        config = self._genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_text,
        )

        start = time.perf_counter() * 1000
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            logger.error("Erro no Google Gemini (%s): %s", self._model, exc)
            raise

        content = response.text or ""
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage else 0
        completion_tokens = usage.candidates_token_count if usage else 0

        return LLMResponse(
            content=content,
            model=self._model,
            provider="google",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="stop",
            latency_ms=time.perf_counter() * 1000 - start,
            raw=None,
        )


def build(config: Dict[str, Any]) -> GoogleGenAIClient:
    return GoogleGenAIClient(
        model=config.get("model", "gemini-2.0-flash"),
        api_key=config.get("api_key"),
    )
