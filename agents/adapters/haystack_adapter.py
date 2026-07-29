# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Adapter Haystack — expõe LLMClient Aurix como Generator do Haystack.

Haystack (deepset) é um framework declarativo para pipelines de NLP e RAG
baseado em componentes. Útil no Aurix para:
- Pipelines de extração de informação de documentos regulatórios (PDF → dados)
- Question-answering sobre contratos e normativos
- Geração de sumários de relatórios BACEN

Requires:
    pip install haystack-ai

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_haystack
    from haystack import Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    llm = create_llm({"provider": "ollama", "model": "mistral:7b"})
    generator = as_haystack(llm)

    # Pipeline RAG simples
    pipeline = Pipeline()
    pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
    pipeline.add_component("prompt", PromptBuilder(template="..."))
    pipeline.add_component("llm", generator)

    pipeline.connect("retriever", "prompt.documents")
    pipeline.connect("prompt", "llm.prompt")
    result = pipeline.run({"retriever": {"query": "limite PIX BACEN"}})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient


class AurixHaystackLLM:
    """Componente Generator do Haystack usando LLMClient Aurix.

    Implementa a interface de Generator esperada pelos pipelines Haystack,
    com suporte às anotações de input/output do Haystack v2+.
    """

    def __init__(self, client: LLMClient, **defaults: Any):
        self._client = client
        self._defaults = defaults

    def warm_up(self) -> None:
        """Haystack chama isso antes de usar o componente — sem operação necessária."""
        pass

    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Interface principal do Haystack Generator.

        Args:
            prompt: O prompt gerado pelo PromptBuilder.
            generation_kwargs: Parâmetros de geração (temperature, max_tokens).

        Returns:
            Dict com chaves ``replies`` e ``meta``.
        """
        kwargs = {**self._defaults, **(generation_kwargs or {})}
        resp = self._client.complete(
            prompt,
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return {
            "replies": [resp.content],
            "meta": [
                {
                    "model": resp.model,
                    "provider": resp.provider,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "finish_reason": resp.finish_reason,
                    "latency_ms": resp.latency_ms,
                }
            ],
        }

    # Compatibilidade com Haystack v1 (legado)
    def predict(self, prompt: str, **kwargs: Any) -> str:
        result = self.run(prompt, generation_kwargs=kwargs)
        return result["replies"][0]


def as_haystack(client: LLMClient, **kwargs: Any) -> AurixHaystackLLM:
    """Adapta um LLMClient Aurix para uso em pipelines Haystack.

    Args:
        client: Qualquer LLMClient criado via ``create_llm()``.
        **kwargs: Defaults para temperature, max_tokens, etc.

    Returns:
        Componente Generator compatível com pipelines Haystack v2.
    """
    return AurixHaystackLLM(client, **kwargs)
