# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Adapter LlamaIndex — expõe LLMClient Aurix como CustomLLM do LlamaIndex.

Permite usar qualquer provider Aurix em pipelines RAG (Retrieval-Augmented
Generation) do LlamaIndex — ideal para o knowledge vault do Aurix e para
consultas sobre normativos BACEN/CMN.

Requires:
    pip install llama-index llama-index-core

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_llamaindex
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

    llm = create_llm({"provider": "ollama", "model": "mistral:7b"})
    li_llm = as_llamaindex(llm)

    # RAG sobre documentação regulatória
    docs = SimpleDirectoryReader("docs/regulatorio/bacen").load_data()
    index = VectorStoreIndex.from_documents(docs, llm=li_llm)
    query_engine = index.as_query_engine(llm=li_llm)

    resp = query_engine.query("Qual o limite de valor para TED sem rastreamento?")
    print(resp.response)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from aurix_ml.llm.base import LLMClient


class AurixLlamaIndexLLM:
    """Wrapper que expõe LLMClient Aurix como CustomLLM do LlamaIndex.

    Implementa a interface necessária para ser usado como ``llm=`` em
    VectorStoreIndex, QueryEngine, e outros componentes LlamaIndex.
    """

    def __init__(self, client: LLMClient, context_window: int = 4096, **kwargs: Any):
        self._client = client
        self._context_window = context_window
        self._defaults = kwargs

        try:
            from llama_index.core.llms import (
                CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
            )
            self._CustomLLM = CustomLLM
            self._CompletionResponse = CompletionResponse
            self._LLMMetadata = LLMMetadata
            self._has_llamaindex = True
        except ImportError:
            self._has_llamaindex = False

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Completion síncrono — interface principal do LlamaIndex."""
        merged = {**self._defaults, **kwargs}
        resp = self._client.complete(
            prompt,
            temperature=merged.get("temperature", 0.7),
            max_tokens=merged.get("max_tokens", 1024),
        )
        if self._has_llamaindex:
            return self._CompletionResponse(text=resp.content)
        return resp.content

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        """Chat interface — LlamaIndex repassa lista de ChatMessage."""
        merged = {**self._defaults, **kwargs}
        if hasattr(messages, "__iter__"):
            aurix_msgs = []
            for m in messages:
                role = getattr(m, "role", "user")
                content = getattr(m, "content", str(m))
                aurix_msgs.append({"role": str(role), "content": content})
        else:
            aurix_msgs = [{"role": "user", "content": str(messages)}]

        resp = self._client.chat(
            aurix_msgs,
            temperature=merged.get("temperature", 0.7),
            max_tokens=merged.get("max_tokens", 1024),
        )
        return resp.content

    @property
    def metadata(self) -> Any:
        if self._has_llamaindex:
            return self._LLMMetadata(
                context_window=self._context_window,
                num_output=512,
                model_name=self._client.model,
            )
        return {"model": self._client.model}

    # Duck-typing para LlamaIndex BaseLLM
    def predict(self, prompt: Any, **kwargs: Any) -> str:
        text = getattr(prompt, "format", None)
        if callable(text):
            text = text(**kwargs)
        return self.complete(str(text)).text if hasattr(self.complete(str(text)), "text") else str(self.complete(str(text)))

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.complete(prompt, **kwargs)


def as_llamaindex(client: LLMClient, context_window: int = 4096, **kwargs: Any) -> AurixLlamaIndexLLM:
    """Adapta um LLMClient Aurix para uso em pipelines LlamaIndex/RAG.

    Args:
        client: Qualquer LLMClient criado via ``create_llm()``.
        context_window: Tamanho do contexto do modelo (tokens).
        **kwargs: Defaults para temperature, max_tokens, etc.

    Returns:
        Wrapper compatível com a interface LlamaIndex CustomLLM.
    """
    return AurixLlamaIndexLLM(client, context_window=context_window, **kwargs)
