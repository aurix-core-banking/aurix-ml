# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider Ollama — modelos LLM locais via servidor Ollama.

Ollama (https://ollama.com) serve modelos open-source (Llama 3, Mistral,
Gemma, Qwen, etc.) com API compatível com OpenAI. Ideal para:
- Desenvolvimento local sem custo de API
- Ambientes air-gapped (sem acesso à internet)
- Dados sensíveis que não podem sair da infraestrutura

Instalação do Ollama:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2          # 3B params, rápido
    ollama pull mistral           # 7B, bom equilíbrio
    ollama pull qwen2.5:14b       # 14B, alta qualidade

Requires:
    pip install openai>=1.0       # Ollama expõe API OpenAI-compatível
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message

logger = logging.getLogger("aurix_ml.llm.providers.ollama")

# Modelos recomendados para contexto bancário (equilíbrio qualidade/velocidade)
RECOMMENDED_MODELS = {
    "fast": "llama3.2:3b",           # ~2GB, ideal para classificação rápida
    "balanced": "mistral:7b",         # ~4GB, bom para análise de risco
    "quality": "qwen2.5:14b",         # ~9GB, melhor para compliance/redação
    "code": "deepseek-coder-v2:16b",  # ~9GB, para geração de código/SQL
}


class OllamaClient(LLMClient):
    """Cliente para modelos locais servidos pelo Ollama.

    Args:
        model: Nome do modelo Ollama (ex: ``llama3.2``, ``mistral:7b``).
        base_url: URL do servidor Ollama. Default: ``http://localhost:11434``.
        timeout: Timeout em segundos para geração. Default: 120s.
        keep_alive: Tempo de keep-alive do modelo em memória (ex: ``5m``, ``-1``).
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        keep_alive: str = "5m",
        **kwargs: Any,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Provider 'ollama' requer o pacote openai: pip install openai"
            ) from exc

        self._model = model
        self._base_url = base_url.rstrip("/")
        self._keep_alive = keep_alive
        self._client = OpenAI(
            base_url=f"{self._base_url}/v1",
            api_key="ollama",  # Ollama não exige API key
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Verifica se o servidor Ollama está acessível."""
        try:
            import urllib.request
            urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lista modelos disponíveis localmente no Ollama."""
        try:
            import json
            import urllib.request
            with urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=5) as r:
                data = json.loads(r.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning("Não foi possível listar modelos Ollama: %s", e)
            return []

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
                extra_body={"keep_alive": self._keep_alive},
            )
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
        except Exception as exc:
            logger.error("Erro na chamada Ollama (%s): %s", self._model, exc)
            raise

        return LLMResponse(
            content=content,
            model=self._model,
            provider="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            latency_ms=time.perf_counter() * 1000 - start,
            raw=None,
        )


def build(config: Dict[str, Any]) -> OllamaClient:
    return OllamaClient(
        model=config.get("model", "llama3.2"),
        base_url=config.get("base_url", "http://localhost:11434"),
        timeout=config.get("timeout", 120),
        keep_alive=config.get("keep_alive", "5m"),
    )
