# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Registry de providers LLM — mapeamento nome → builder.

Providers sem dependência externa (mock, callable) são registrados
de forma eager. Providers que dependem de SDK de vendor ou runtime
local são registrados de forma lazy — o SDK só é importado quando
``create_llm`` é chamado com aquele provider específico.

Isso garante que importar ``aurix_ml.llm`` nunca quebre o ambiente
se um SDK não estiver instalado.

Providers disponíveis
---------------------
Cloud / API:
  openai      — OpenAI API (gpt-4o, gpt-4o-mini, ...)
  azure       — Azure OpenAI Service (mesmo SDK openai)
  bedrock     — AWS Bedrock (amazon.nova, anthropic.claude, ...)
  google      — Google Gemini API
  gemini      — alias de google

Local / Open-source:
  ollama      — Ollama (llama3, mistral, qwen, gemma, ...)
  lmstudio    — LM Studio (OpenAI-compatível, porta 1234)
  llamacpp    — llama.cpp server (OpenAI-compatível)
  vllm        — vLLM self-hosted (OpenAI-compatível)
  huggingface — HuggingFace Transformers (inferência direta)

Utilitários:
  mock        — cliente offline para CI/CD
  callable    — wraps qualquer função Python
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

from aurix_ml.llm.base import LLMClient

logger = logging.getLogger("aurix_ml.llm.registry")

ProviderBuilder = Callable[[Dict[str, Any]], LLMClient]

_PROVIDERS: Dict[str, ProviderBuilder] = {}


def register_provider(name: str, builder: ProviderBuilder) -> None:
    """Registra (ou sobrescreve) um builder de provider pelo nome."""
    _PROVIDERS[name] = builder
    logger.debug("Provider '%s' registrado.", name)


def available_providers() -> List[str]:
    """Retorna lista ordenada de providers registrados."""
    return sorted(_PROVIDERS)


def create_llm(config: Dict[str, Any], **overrides: Any) -> LLMClient:
    """Constrói um cliente LLM a partir de um dicionário de configuração.

    O dict deve conter ao menos a chave ``provider``. Kwargs extras em
    ``overrides`` sobrescrevem entradas do config (útil para injeção de
    configuração via variáveis de ambiente ou propriedades do Spring).

    Exemplos::

        # CI/CD offline
        llm = create_llm({"provider": "mock"})

        # Ollama local
        llm = create_llm({"provider": "ollama", "model": "mistral:7b"})

        # LM Studio
        llm = create_llm({"provider": "lmstudio", "model": "lmstudio-community/Llama-3.2-3B"})

        # HuggingFace com quantização 4-bit
        llm = create_llm({"provider": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct",
                           "load_in_4bit": True})

        # OpenAI
        llm = create_llm({"provider": "openai", "model": "gpt-4o-mini"})

        # Override de env var
        llm = create_llm(cfg, model=os.environ.get("AUREUS_LLM_MODEL"))

    Raises:
        ValueError: Se o provider não estiver registrado.
    """
    cfg = dict(config)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    provider = cfg.get("provider")
    if not provider:
        raise ValueError("config deve conter a chave 'provider'.")

    builder = _PROVIDERS.get(str(provider))
    if builder is None:
        raise ValueError(
            f"Provider desconhecido: '{provider}'. "
            f"Disponíveis: {available_providers()}."
        )
    logger.debug("Construindo cliente LLM via provider '%s'", provider)
    return builder(cfg)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Carrega configuração de LLM de arquivo JSON ou YAML.

    JSON usa stdlib. YAML requer PyYAML (importado de forma lazy).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config não encontrado: {p}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Config YAML requer PyYAML: pip install pyyaml") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


# ---------------------------------------------------------------------------
# Registro dos providers builtin
# ---------------------------------------------------------------------------

def _openai_compat_builder(
    base_url: str | None,
    provider_name: str,
    default_model: str = "default",
) -> ProviderBuilder:
    """Factory para providers que usam API OpenAI-compatível."""
    def _build(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.openai_sdk import OpenAIClient
        return OpenAIClient(
            model=cfg.get("model", default_model),
            api_key=cfg.get("api_key", "local"),
            base_url=cfg.get("base_url", base_url),
            timeout=cfg.get("timeout", 120),
        )
    _build.__name__ = f"build_{provider_name}"
    return _build


def _register_builtins() -> None:
    # ------------------------------------------------------------------
    # Providers sem dependência (eager)
    # ------------------------------------------------------------------
    from aurix_ml.llm.providers import mock, callable_provider
    register_provider("mock", mock.build)
    register_provider("callable", callable_provider.build)

    # ------------------------------------------------------------------
    # Providers locais / open-source (lazy)
    # ------------------------------------------------------------------

    def _ollama(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.ollama import build
        return build(cfg)

    def _huggingface(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.huggingface import build
        return build(cfg)

    # LM Studio — servidor local com API OpenAI-compatível (porta 1234)
    _lmstudio = _openai_compat_builder(
        base_url="http://localhost:1234/v1",
        provider_name="lmstudio",
    )

    # llama.cpp server — API OpenAI-compatível (porta 8080 por padrão)
    _llamacpp = _openai_compat_builder(
        base_url="http://localhost:8080/v1",
        provider_name="llamacpp",
    )

    # vLLM — serving de alta performance (porta 8000 por padrão)
    _vllm = _openai_compat_builder(
        base_url="http://localhost:8000/v1",
        provider_name="vllm",
    )

    register_provider("ollama", _ollama)
    register_provider("huggingface", _huggingface)
    register_provider("hf", _huggingface)          # alias
    register_provider("lmstudio", _lmstudio)
    register_provider("llamacpp", _llamacpp)
    register_provider("llama_cpp", _llamacpp)      # alias
    register_provider("vllm", _vllm)

    # ------------------------------------------------------------------
    # Providers cloud (lazy — SDK importado apenas quando usado)
    # ------------------------------------------------------------------

    def _openai(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.openai_sdk import build
        return build(cfg)

    def _bedrock(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.bedrock import build
        return build(cfg)

    def _google(cfg: Dict[str, Any]) -> LLMClient:
        from aurix_ml.llm.providers.google_genai import build
        return build(cfg)

    register_provider("openai", _openai)
    register_provider("azure", _openai)       # Azure OpenAI — mesmo SDK
    register_provider("bedrock", _bedrock)
    register_provider("aws", _bedrock)        # alias
    register_provider("google", _google)
    register_provider("gemini", _google)      # alias


_register_builtins()
