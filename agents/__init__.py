# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Camada de acoplamento com frameworks de agentes open-source.

Adapta o ``LLMClient`` da Aurix para ser usado como backend
dos principais frameworks de agentes:

- **LangChain** (``langchain-community``)
- **LlamaIndex** (``llama-index``)
- **CrewAI** (``crewai``)
- **AutoGen / AG2** (``pyautogen``)
- **Haystack** (``haystack-ai``)

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_langchain, as_llamaindex, as_crewai

    # Qualquer provider Aurix → LangChain ChatModel
    llm = create_llm({"provider": "ollama", "model": "mistral:7b"})
    lc_llm = as_langchain(llm)

    # Usar em chains / agents LangChain normalmente
    from langchain.chains import LLMChain
    from langchain.prompts import ChatPromptTemplate
    chain = LLMChain(llm=lc_llm, prompt=ChatPromptTemplate.from_template("{input}"))
    result = chain.run("Analise o risco da operação...")
"""

from aurix_ml.agents.adapters.langchain_adapter import as_langchain, AurixLangChainLLM
from aurix_ml.agents.adapters.llamaindex_adapter import as_llamaindex, AurixLlamaIndexLLM
from aurix_ml.agents.adapters.crewai_adapter import as_crewai, AurixCrewAILLM
from aurix_ml.agents.adapters.autogen_adapter import as_autogen, AurixAutoGenClient
from aurix_ml.agents.adapters.haystack_adapter import as_haystack, AurixHaystackLLM
from aurix_ml.agents.runner import AgentRunner, AgentTask, AgentResult

__all__ = [
    # LangChain
    "as_langchain",
    "AurixLangChainLLM",
    # LlamaIndex
    "as_llamaindex",
    "AurixLlamaIndexLLM",
    # CrewAI
    "as_crewai",
    "AurixCrewAILLM",
    # AutoGen / AG2
    "as_autogen",
    "AurixAutoGenClient",
    # Haystack
    "as_haystack",
    "AurixHaystackLLM",
    # Runner genérico
    "AgentRunner",
    "AgentTask",
    "AgentResult",
]
