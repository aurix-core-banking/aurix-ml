# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""aurix-ml LLM layer — vendor-neutral LLM client, padrão inspirado no SantanderAI/llm_bridge.

Uso básico::

    from aurix_ml.llm import create_llm

    # Desenvolvimento / CI sem credenciais
    llm = create_llm({"provider": "mock"})

    # Produção com OpenAI
    llm = create_llm({"provider": "openai", "model": "gpt-4o-mini"})

    # Bedrock (AWS corporativo)
    llm = create_llm({"provider": "bedrock", "model": "amazon.nova-lite-v1:0"})

    resp = llm.complete("Analise o risco desta operação: ...")
    print(resp.content, resp.total_tokens)
"""

from aurix_ml.llm.base import LLMClient, LLMResponse, Message
from aurix_ml.llm.registry import create_llm, register_provider, available_providers, load_config

__all__ = [
    "LLMClient",
    "LLMResponse",
    "Message",
    "create_llm",
    "register_provider",
    "available_providers",
    "load_config",
]
