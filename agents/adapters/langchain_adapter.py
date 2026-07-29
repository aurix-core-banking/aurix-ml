# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Adapter LangChain — expõe LLMClient Aurix como ChatModel do LangChain.

Permite usar qualquer provider Aurix (Ollama, HuggingFace, OpenAI, Mock...)
em qualquer chain, agent ou pipeline LangChain sem alteração de código.

Requires:
    pip install langchain langchain-core

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_langchain
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    llm = create_llm({"provider": "ollama", "model": "llama3.2"})
    lc = as_langchain(llm)

    # LCEL chain
    prompt = ChatPromptTemplate.from_template("Analise o risco: {input}")
    chain = prompt | lc | StrOutputParser()
    resultado = chain.invoke({"input": "PIX de R$50.000 às 3h da manhã"})

    # Com agente ReAct
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain.tools import Tool

    tools = [Tool(name="consultar_bacen", func=..., description="...")]
    agent = create_react_agent(lc, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    executor.invoke({"input": "Verifique conformidade da operação X"})
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from aurix_ml.llm.base import LLMClient, Message


class AurixLangChainLLM:
    """Wrapper que expõe LLMClient Aurix como BaseChatModel do LangChain.

    Implementa a interface mínima necessária para ser usada em chains LCEL,
    agents ReAct, e qualquer componente que aceite um BaseChatModel.
    """

    def __init__(self, client: LLMClient, **default_kwargs: Any):
        self._client = client
        self._defaults = default_kwargs

        # Importação lazy — não quebra o módulo se LangChain não estiver instalado
        try:
            from langchain_core.language_models.chat_models import BaseChatModel
            from langchain_core.messages import (
                AIMessage, HumanMessage, SystemMessage, BaseMessage
            )
            from langchain_core.outputs import ChatResult, ChatGeneration
            self._BaseChatModel = BaseChatModel
            self._AIMessage = AIMessage
            self._ChatResult = ChatResult
            self._ChatGeneration = ChatGeneration
            self._msg_classes = {
                "human": HumanMessage,
                "user": HumanMessage,
                "ai": AIMessage,
                "assistant": AIMessage,
                "system": SystemMessage,
            }
            self._has_langchain = True
        except ImportError:
            self._has_langchain = False

    def _to_aurix_messages(self, messages: List[Any]) -> List[Message]:
        """Converte mensagens LangChain para o formato Aurix."""
        result: List[Message] = []
        for m in messages:
            if hasattr(m, "type"):
                role = {"human": "user", "ai": "assistant", "system": "system"}.get(
                    m.type, "user"
                )
            elif hasattr(m, "role"):
                role = m.role
            else:
                role = "user"
            content = m.content if hasattr(m, "content") else str(m)
            result.append({"role": role, "content": content})
        return result

    def invoke(self, messages: Any, **kwargs: Any) -> Any:
        """Chamada síncrona — compatível com interface LCEL."""
        if isinstance(messages, str):
            aurix_msgs = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Pode ser lista de BaseMessage ou lista de dicts
            if messages and hasattr(messages[0], "content"):
                aurix_msgs = self._to_aurix_messages(messages)
            else:
                aurix_msgs = messages
        else:
            aurix_msgs = [{"role": "user", "content": str(messages)}]

        merged = {**self._defaults, **kwargs}
        response = self._client.chat(
            aurix_msgs,
            temperature=merged.get("temperature", 0.7),
            max_tokens=merged.get("max_tokens", 1024),
        )

        if self._has_langchain:
            return self._AIMessage(content=response.content)
        return response.content

    def __call__(self, messages: Any, **kwargs: Any) -> Any:
        return self.invoke(messages, **kwargs)

    # Compatibilidade com LangChain BaseChatModel.predict()
    def predict(self, text: str, **kwargs: Any) -> str:
        resp = self._client.complete(text, **{**self._defaults, **kwargs})
        return resp.content

    # Propriedades exigidas pelo LangChain
    @property
    def _llm_type(self) -> str:
        return f"aurix-{self._client.provider}"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model": self._client.model, "provider": self._client.provider}

    def as_langchain_model(self) -> Any:
        """Retorna instância compatível com BaseChatModel via duck typing."""
        return self


def as_langchain(client: LLMClient, **kwargs: Any) -> AurixLangChainLLM:
    """Adapta um LLMClient Aurix para uso em pipelines LangChain.

    Args:
        client: Qualquer LLMClient criado via ``create_llm()``.
        **kwargs: Defaults para temperature, max_tokens, etc.

    Returns:
        Wrapper compatível com a interface LangChain ChatModel.
    """
    return AurixLangChainLLM(client, **kwargs)
