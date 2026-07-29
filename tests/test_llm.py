# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Testes unitários da camada LLM — sem dependências externas, CI-safe.

Todos os testes usam o MockClient (sem rede, sem credenciais, sem GPU).
Cobrem: interface base, registry, providers locais, adapters de agentes,
e o AgentRunner com retry.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from aurix_ml.llm import create_llm, available_providers, register_provider
from aurix_ml.llm.base import LLMClient, LLMResponse, Message, split_system_messages
from aurix_ml.llm.providers.mock import MockClient, build as mock_build
from aurix_ml.llm.providers.callable_provider import CallableClient, build as callable_build
from aurix_ml.agents.runner import AgentRunner, AgentTask, AgentResult
from aurix_ml.agents.adapters.langchain_adapter import AurixLangChainLLM, as_langchain
from aurix_ml.agents.adapters.llamaindex_adapter import AurixLlamaIndexLLM, as_llamaindex
from aurix_ml.agents.adapters.crewai_adapter import AurixCrewAILLM, as_crewai
from aurix_ml.agents.adapters.haystack_adapter import AurixHaystackLLM, as_haystack


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def mock_llm() -> MockClient:
    return MockClient(response="Risco BAIXO. Score 720. Aprovado.")

@pytest.fixture
def echo_llm() -> MockClient:
    """Retorna o último user message — útil para testar prompt building."""
    return MockClient()  # modo echo

@pytest.fixture
def callable_llm() -> CallableClient:
    def meu_modelo(messages):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[modelo-interno] {last[:50]}"
    return CallableClient(fn=meu_modelo, model="modelo-interno-v1")


# ===========================================================================
# 1. LLMResponse
# ===========================================================================

class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(content="ok", model="m", prompt_tokens=10, completion_tokens=5)
        assert r.total_tokens == 15

    def test_estimated_cost_default(self):
        r = LLMResponse(content="ok", model="m")
        assert r.estimated_cost_usd == 0.0

    def test_defaults(self):
        r = LLMResponse(content="texto", model="mock")
        assert r.finish_reason == "stop"
        assert r.latency_ms == 0.0
        assert r.provider == "unknown"


# ===========================================================================
# 2. split_system_messages
# ===========================================================================

class TestSplitSystemMessages:
    def test_extrai_system(self):
        msgs = [
            {"role": "system", "content": "Você é analista."},
            {"role": "user", "content": "Analise isso."},
        ]
        system, turns = split_system_messages(msgs)
        assert system == "Você é analista."
        assert len(turns) == 1
        assert turns[0]["role"] == "user"

    def test_sem_system(self):
        msgs = [{"role": "user", "content": "oi"}]
        system, turns = split_system_messages(msgs)
        assert system is None
        assert len(turns) == 1

    def test_multiplos_system(self):
        msgs = [
            {"role": "system", "content": "Parte 1."},
            {"role": "system", "content": "Parte 2."},
            {"role": "user", "content": "Pergunta."},
        ]
        system, turns = split_system_messages(msgs)
        assert "Parte 1." in system
        assert "Parte 2." in system
        assert len(turns) == 1


# ===========================================================================
# 3. MockClient
# ===========================================================================

class TestMockClient:
    def test_resposta_fixa(self, mock_llm):
        resp = mock_llm.chat([{"role": "user", "content": "oi"}])
        assert resp.content == "Risco BAIXO. Score 720. Aprovado."
        assert resp.provider == "mock"
        assert resp.total_tokens > 0

    def test_echo_mode(self, echo_llm):
        resp = echo_llm.chat([{"role": "user", "content": "teste"}])
        assert "teste" in resp.content

    def test_responder_callable(self):
        llm = MockClient(responder=lambda msgs: "resposta-customizada")
        resp = llm.chat([{"role": "user", "content": "x"}])
        assert resp.content == "resposta-customizada"

    def test_complete_wrapper(self, mock_llm):
        resp = mock_llm.complete("qual o risco?", system="Você é analista.")
        assert resp.content == "Risco BAIXO. Score 720. Aprovado."

    def test_model_property(self, mock_llm):
        assert mock_llm.model == "mock"

    def test_latency_measured(self, mock_llm):
        resp = mock_llm.chat([{"role": "user", "content": "x"}])
        assert resp.latency_ms >= 0


# ===========================================================================
# 4. CallableClient
# ===========================================================================

class TestCallableClient:
    def test_wraps_function(self, callable_llm):
        resp = callable_llm.chat([{"role": "user", "content": "analise o PIX"}])
        assert "[modelo-interno]" in resp.content
        assert resp.provider == "callable"

    def test_requer_callable(self):
        with pytest.raises(TypeError):
            CallableClient(fn="nao-e-callable")

    def test_build_sem_callable_raises(self):
        with pytest.raises(ValueError, match="callable"):
            callable_build({})


# ===========================================================================
# 5. Registry
# ===========================================================================

class TestRegistry:
    def test_providers_disponiveis(self):
        providers = available_providers()
        assert "mock" in providers
        assert "callable" in providers
        assert "ollama" in providers
        assert "huggingface" in providers
        assert "lmstudio" in providers
        assert "llamacpp" in providers
        assert "vllm" in providers
        assert "openai" in providers
        assert "azure" in providers
        assert "bedrock" in providers
        assert "google" in providers
        assert "gemini" in providers

    def test_create_mock(self):
        llm = create_llm({"provider": "mock", "response": "ok"})
        assert isinstance(llm, MockClient)
        assert llm.complete("x").content == "ok"

    def test_create_callable(self):
        llm = create_llm({"provider": "callable", "callable": lambda msgs: "resp"})
        assert isinstance(llm, CallableClient)

    def test_provider_desconhecido(self):
        with pytest.raises(ValueError, match="desconhecido"):
            create_llm({"provider": "nao-existe"})

    def test_sem_provider_key(self):
        with pytest.raises(ValueError, match="provider"):
            create_llm({"model": "x"})

    def test_override_config(self):
        llm = create_llm({"provider": "mock", "model": "base"}, model="override")
        assert llm.model == "override"

    def test_register_custom_provider(self):
        class MeuLLM(LLMClient):
            @property
            def model(self): return "meu-modelo"
            @property
            def provider(self): return "meu-provider"
            def chat(self, messages, **kwargs):
                return LLMResponse(content="custom", model="meu-modelo", provider="meu-provider")

        register_provider("meu-provider", lambda cfg: MeuLLM())
        llm = create_llm({"provider": "meu-provider"})
        assert llm.provider == "meu-provider"
        assert llm.complete("x").content == "custom"


# ===========================================================================
# 6. AgentTask
# ===========================================================================

class TestAgentTask:
    def test_build_messages_simples(self):
        task = AgentTask(name="teste", prompt="analise isso")
        msgs = task.build_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_build_messages_com_system(self):
        task = AgentTask(name="t", prompt="p", system="Você é analista.")
        msgs = task.build_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_build_messages_com_contexto(self):
        task = AgentTask(name="t", prompt="avalie", context="Dados: score=720")
        msgs = task.build_messages()
        assert "Dados: score=720" in msgs[-1]["content"]
        assert "avalie" in msgs[-1]["content"]


# ===========================================================================
# 7. AgentRunner
# ===========================================================================

class TestAgentRunner:
    def test_executa_tarefa_simples(self, mock_llm):
        runner = AgentRunner(mock_llm)
        task = AgentTask(name="credito", prompt="analise o cliente")
        result = runner.run(task)

        assert result.success is True
        assert "Risco BAIXO" in result.content
        assert result.retries == 0
        assert result.latency_ms > 0

    def test_retry_em_falha(self):
        """Runner deve tentar max_retries+1 vezes e registrar o erro."""
        chamadas = {"n": 0}
        def falha(messages, **kwargs):
            chamadas["n"] += 1
            raise RuntimeError("timeout simulado")

        llm = MockClient(responder=lambda _: (_ for _ in ()).throw(RuntimeError("x")))
        # Usar callable que levanta exceção
        llm_falho = CallableClient(fn=lambda _: (_ for _ in ()).throw(RuntimeError("timeout")))

        runner = AgentRunner(llm_falho, max_retries=1)

        # Patch sleep para não esperar nos testes
        with patch("time.sleep"):
            result = runner.run(AgentTask(name="t", prompt="x"))

        assert result.success is False
        assert result.error is not None
        assert result.retries == 1

    def test_callback_on_success(self, mock_llm):
        resultados = []
        runner = AgentRunner(mock_llm, on_success=resultados.append)
        runner.run(AgentTask(name="t", prompt="x"))
        assert len(resultados) == 1
        assert resultados[0].success

    def test_run_batch(self, mock_llm):
        runner = AgentRunner(mock_llm)
        tasks = [AgentTask(name=f"t{i}", prompt=f"tarefa {i}") for i in range(3)]
        results = runner.run_batch(tasks)
        assert len(results) == 3
        assert all(r.success for r in results)

    def test_metadata_preenchida(self, mock_llm):
        runner = AgentRunner(mock_llm)
        result = runner.run(AgentTask(name="t", prompt="x", tags=["credito", "prod"]))
        assert "model" in result.metadata
        assert "provider" in result.metadata
        assert "credito" in result.metadata["tags"]


# ===========================================================================
# 8. Adapters de Frameworks
# ===========================================================================

class TestAdapters:
    def test_langchain_invoke(self, mock_llm):
        lc = as_langchain(mock_llm)
        result = lc.invoke([{"role": "user", "content": "analise"}])
        # Retorna AIMessage ou string
        content = result.content if hasattr(result, "content") else result
        assert "Risco BAIXO" in content

    def test_langchain_predict(self, mock_llm):
        lc = as_langchain(mock_llm)
        resp = lc.predict("analise o risco")
        assert isinstance(resp, str)
        assert "Risco BAIXO" in resp

    def test_langchain_llm_type(self, mock_llm):
        lc = as_langchain(mock_llm)
        assert "mock" in lc._llm_type

    def test_llamaindex_complete(self, mock_llm):
        li = as_llamaindex(mock_llm)
        result = li.complete("analise o crédito")
        content = result.text if hasattr(result, "text") else result
        assert "Risco BAIXO" in content

    def test_llamaindex_metadata(self, mock_llm):
        li = as_llamaindex(mock_llm, context_window=8192)
        meta = li.metadata
        assert meta is not None

    def test_crewai_call(self, mock_llm):
        crew = as_crewai(mock_llm)
        result = crew.call([{"role": "user", "content": "analise"}])
        assert "Risco BAIXO" in result

    def test_crewai_model_property(self, mock_llm):
        crew = as_crewai(mock_llm)
        assert "mock" in crew.model

    def test_haystack_run(self, mock_llm):
        hs = as_haystack(mock_llm)
        result = hs.run("analise o risco do cliente")
        assert "replies" in result
        assert "meta" in result
        assert "Risco BAIXO" in result["replies"][0]
        assert result["meta"][0]["provider"] == "mock"

    def test_haystack_warm_up(self, mock_llm):
        """warm_up não deve levantar exceção."""
        hs = as_haystack(mock_llm)
        hs.warm_up()  # deve ser no-op


# ===========================================================================
# 9. Testes de Integração (smoke tests com Ollama — pulados se não disponível)
# ===========================================================================

@pytest.mark.integration
class TestOllamaIntegration:
    """Smoke tests com Ollama local. Requer: ollama pull llama3.2"""

    def test_ollama_disponivel(self):
        try:
            llm = create_llm({"provider": "ollama", "model": "llama3.2"})
        except ImportError:
            pytest.skip("Ollama/openai SDK não instalado.")
        if not llm.is_available():
            pytest.skip("Ollama não está rodando localmente.")

    def test_ollama_listagem_modelos(self):
        try:
            llm = create_llm({"provider": "ollama", "model": "llama3.2"})
        except ImportError:
            pytest.skip("Ollama/openai SDK não instalado.")
        if not llm.is_available():
            pytest.skip("Ollama não disponível.")
        models = llm.list_models()
        assert isinstance(models, list)

    def test_ollama_complete(self):
        try:
            llm = create_llm({"provider": "ollama", "model": "llama3.2"})
        except ImportError:
            pytest.skip("Ollama/openai SDK não instalado.")
        if not llm.is_available():
            pytest.skip("Ollama não disponível.")
        resp = llm.complete("Responda apenas: 'ok'", max_tokens=10)
        assert len(resp.content) > 0
        assert resp.provider == "ollama"
