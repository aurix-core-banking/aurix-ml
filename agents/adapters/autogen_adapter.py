# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Adapter AutoGen / AG2 — expõe LLMClient Aurix como ModelClient do AutoGen.

AutoGen (Microsoft) é um framework para conversas multi-agente onde agentes
se comunicam entre si para resolver tarefas complexas. Útil no Aurix para:
- Revisão cruzada de decisões críticas (agente propõe, agente revisa)
- Geração automatizada de código SQL para relatórios regulatórios
- Simulação de mesa de compliance para validação de processos

Requires:
    pip install pyautogen  # ou: pip install ag2

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents import as_autogen
    from autogen import AssistantAgent, UserProxyAgent

    llm = create_llm({"provider": "ollama", "model": "qwen2.5:14b"})
    config = as_autogen(llm)

    # Agente assistente com provider Aurix
    assistente = AssistantAgent(
        name="analista_credito",
        system_message="Você é especialista em análise de crédito bancário brasileiro.",
        llm_config=config,
    )

    # Agente proxy (representa o usuário/sistema)
    proxy = UserProxyAgent(
        name="sistema_aurix",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config={"work_dir": "workspace"},
    )

    proxy.initiate_chat(
        assistente,
        message="Analise o risco do cliente ID 9876 e gere relatório BACEN",
    )
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Union

from aurix_ml.llm.base import LLMClient, Message


class AurixAutoGenClient:
    """ModelClient compatível com AutoGen que usa um LLMClient Aurix.

    AutoGen usa uma interface de ``model_client_cls`` customizado.
    Esta classe implementa o protocolo esperado pelo AutoGen v0.2+.
    """

    def __init__(self, client: LLMClient, **defaults: Any):
        self._client = client
        self._defaults = defaults
        # Configuração no formato que AutoGen espera
        self._config = {
            "model": client.model,
            "api_type": "aurix",
        }

    def create(self, params: Dict[str, Any]) -> Any:
        """Interface principal chamada pelo AutoGen para gerar resposta."""
        messages: List[Message] = params.get("messages", [])
        temperature = params.get("temperature", self._defaults.get("temperature", 0.7))
        max_tokens = params.get("max_tokens", self._defaults.get("max_tokens", 1024))

        resp = self._client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Retornar no formato OpenAI que o AutoGen espera
        return _MockOpenAIResponse(
            content=resp.content,
            model=resp.model,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
        )

    def message_retrieval(self, response: Any) -> List[str]:
        """Extrai lista de mensagens da resposta — interface AutoGen."""
        if hasattr(response, "choices"):
            return [c.message.content for c in response.choices]
        if hasattr(response, "content"):
            return [response.content]
        return [str(response)]

    def cost(self, response: Any) -> float:
        """Custo estimado da chamada — AutoGen usa para rastreamento."""
        return 0.0

    @staticmethod
    def get_usage(response: Any) -> Dict[str, Any]:
        """Retorna dict de usage — AutoGen usa para rastreamento."""
        if hasattr(response, "usage"):
            return {
                "prompt_tokens": response.usage.get("prompt_tokens", 0),
                "completion_tokens": response.usage.get("completion_tokens", 0),
                "total_tokens": response.usage.get("total_tokens", 0),
            }
        return {}


class _MockOpenAIResponse:
    """Simula a estrutura de resposta OpenAI que o AutoGen espera."""

    def __init__(self, content: str, model: str, prompt_tokens: int, completion_tokens: int):
        self.choices = [_Choice(content)]
        self.model = model
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Message:
    def __init__(self, content: str):
        self.content = content
        self.role = "assistant"
        self.function_call = None
        self.tool_calls = None


def as_autogen(client: LLMClient, **kwargs: Any) -> Dict[str, Any]:
    """Gera configuração AutoGen usando um LLMClient Aurix.

    Args:
        client: Qualquer LLMClient criado via ``create_llm()``.
        **kwargs: Parâmetros adicionais (temperature, max_tokens, etc.).

    Returns:
        Dict ``llm_config`` pronto para passar ao AssistantAgent/UserProxyAgent.

    Exemplo::

        config = as_autogen(create_llm({"provider": "ollama", "model": "llama3.2"}))
        agent = AssistantAgent("analista", llm_config=config)
    """
    autogen_client = AurixAutoGenClient(client, **kwargs)
    return {
        "config_list": [{"model": client.model, "api_type": "aurix"}],
        "model_client_cls": type(autogen_client),
        # Instância disponível via config para acesso manual
        "_aurix_client": autogen_client,
        **kwargs,
    }
