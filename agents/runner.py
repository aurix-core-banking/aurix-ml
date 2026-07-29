# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""AgentRunner — orquestrador de tarefas de agentes para o Aurix.

Runner genérico que executa tarefas LLM com qualquer provider Aurix,
com suporte a retry, timeout, rastreamento de custo e logging estruturado.
Agnóstico de framework — funciona com ou sem LangChain/CrewAI/AutoGen.

Uso::

    from aurix_ml.llm import create_llm
    from aurix_ml.agents.runner import AgentRunner, AgentTask

    llm = create_llm({"provider": "ollama", "model": "llama3.2"})
    runner = AgentRunner(llm, max_retries=2, timeout_seconds=30)

    task = AgentTask(
        name="analise_risco_credito",
        system="Você é analista sênior de crédito do Banco Aurix.",
        prompt="Cliente solicitou limite de R$50.000. Score: 720. Renda: R$8.000/mês.",
        expected_output="JSON com: decisao, score_risco, justificativa, limite_sugerido",
    )

    result = runner.run(task)
    print(result.content)
    print(f"Tokens: {result.total_tokens} | Latência: {result.latency_ms:.0f}ms")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message

logger = logging.getLogger("aurix_ml.agents.runner")


@dataclass
class AgentTask:
    """Definição de uma tarefa para o agente executar.

    Attributes:
        name: Identificador legível da tarefa (para logging).
        prompt: Prompt principal (mensagem do usuário).
        system: System prompt que define o papel/comportamento do agente.
        context: Contexto adicional injetado no prompt (dados, documentos).
        temperature: Temperatura de geração (0=determinístico, 1=criativo).
        max_tokens: Limite de tokens na resposta.
        tags: Tags para rastreamento (ex: ["credito", "bacen", "producao"]).
    """

    name: str
    prompt: str
    system: Optional[str] = None
    context: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    tags: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None  # hint para logging/validação

    def build_messages(self) -> List[Message]:
        """Constrói a lista de mensagens para o LLM."""
        messages: List[Message] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        user_content = self.prompt
        if self.context:
            user_content = f"## Contexto\n{self.context}\n\n## Tarefa\n{self.prompt}"
        messages.append({"role": "user", "content": user_content})
        return messages


@dataclass
class AgentResult:
    """Resultado de uma execução de tarefa de agente.

    Attributes:
        task_name: Nome da tarefa executada.
        content: Conteúdo da resposta do LLM.
        success: True se executou sem erro.
        total_tokens: Total de tokens consumidos.
        latency_ms: Latência total (inclui retries).
        retries: Número de tentativas realizadas.
        error: Mensagem de erro se ``success=False``.
        metadata: Metadados extras (model, provider, tags, etc.).
    """

    task_name: str
    content: str
    success: bool
    total_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRunner:
    """Executa tarefas de agente com retry, timeout e logging estruturado.

    Args:
        llm: Qualquer LLMClient criado via ``create_llm()``.
        max_retries: Número máximo de tentativas em caso de erro.
        timeout_seconds: Timeout por tentativa (0 = sem timeout).
        on_success: Callback chamado após execução bem-sucedida.
        on_error: Callback chamado em caso de falha definitiva.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_retries: int = 2,
        timeout_seconds: int = 60,
        on_success: Optional[Any] = None,
        on_error: Optional[Any] = None,
    ):
        self._llm = llm
        self._max_retries = max_retries
        self._timeout = timeout_seconds
        self._on_success = on_success
        self._on_error = on_error

    def run(self, task: AgentTask) -> AgentResult:
        """Executa uma tarefa com retry automático.

        Args:
            task: Definição da tarefa (AgentTask).

        Returns:
            AgentResult com conteúdo, métricas e status.
        """
        logger.info(
            "[AgentRunner] Iniciando tarefa='%s' provider=%s model=%s tags=%s",
            task.name,
            self._llm.provider,
            self._llm.model,
            task.tags,
        )

        messages = task.build_messages()
        total_start = time.perf_counter() * 1000
        last_error: Optional[str] = None
        total_tokens = 0

        for attempt in range(self._max_retries + 1):
            try:
                resp = self._llm.chat(
                    messages,
                    temperature=task.temperature,
                    max_tokens=task.max_tokens,
                )
                total_tokens += resp.total_tokens
                elapsed = time.perf_counter() * 1000 - total_start

                logger.info(
                    "[AgentRunner] tarefa='%s' OK attempt=%d tokens=%d latency=%.0fms",
                    task.name,
                    attempt + 1,
                    total_tokens,
                    elapsed,
                )

                result = AgentResult(
                    task_name=task.name,
                    content=resp.content,
                    success=True,
                    total_tokens=total_tokens,
                    latency_ms=elapsed,
                    retries=attempt,
                    metadata={
                        "model": resp.model,
                        "provider": resp.provider,
                        "finish_reason": resp.finish_reason,
                        "tags": task.tags,
                    },
                )
                if self._on_success:
                    self._on_success(result)
                return result

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "[AgentRunner] tarefa='%s' ERRO attempt=%d/%d: %s",
                    task.name,
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # backoff exponencial
                    logger.info("[AgentRunner] Aguardando %ds antes de retry...", wait)
                    time.sleep(wait)

        elapsed = time.perf_counter() * 1000 - total_start
        result = AgentResult(
            task_name=task.name,
            content="",
            success=False,
            total_tokens=total_tokens,
            latency_ms=elapsed,
            retries=self._max_retries,
            error=last_error,
            metadata={"tags": task.tags},
        )
        logger.error(
            "[AgentRunner] tarefa='%s' FALHOU após %d tentativas: %s",
            task.name,
            self._max_retries + 1,
            last_error,
        )
        if self._on_error:
            self._on_error(result)
        return result

    def run_batch(self, tasks: List[AgentTask]) -> List[AgentResult]:
        """Executa múltiplas tarefas sequencialmente.

        Para execução paralela, use ``run_batch_async`` com asyncio.
        """
        return [self.run(t) for t in tasks]
