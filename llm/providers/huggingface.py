# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider HuggingFace Transformers — inferência local direta de modelos HF.

Carrega modelos diretamente do HuggingFace Hub ou de um diretório local.
Ideal para modelos fine-tuned no domínio bancário ou quando se quer
inferência sem servidor intermediário (llama.cpp, Ollama, etc.).

Modelos recomendados para fintech/português:
- ``maritaca-ai/sabia-7b``               — modelo PT-BR especializado
- ``meta-llama/Meta-Llama-3.2-3B-Instruct``   — leve, multilingual
- ``microsoft/Phi-3.5-mini-instruct``    — 3.8B, excelente qualidade
- ``Qwen/Qwen2.5-7B-Instruct``          — 7B, forte em raciocínio

Requires:
    pip install transformers>=4.40 torch accelerate bitsandbytes
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message, split_system_messages

logger = logging.getLogger("aurix_ml.llm.providers.huggingface")


class HuggingFaceClient(LLMClient):
    """Inferência local via HuggingFace Transformers pipeline.

    Args:
        model: ID do modelo no HF Hub ou caminho local.
        device: ``cpu``, ``cuda``, ``mps`` ou ``auto``.
        load_in_4bit: Quantização 4-bit via bitsandbytes (reduz VRAM ~4x).
        load_in_8bit: Quantização 8-bit via bitsandbytes (reduz VRAM ~2x).
        trust_remote_code: Permitir código customizado do repo do modelo.
    """

    def __init__(
        self,
        model: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        torch_dtype: str = "auto",
        **kwargs: Any,
    ):
        try:
            import transformers
            import torch
        except ImportError as exc:
            raise ImportError(
                "Provider 'huggingface' requer: pip install transformers torch accelerate"
            ) from exc

        self._model_id = model
        logger.info("Carregando modelo HuggingFace: %s (device=%s)", model, device)

        quant_kwargs: Dict[str, Any] = {}
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                )
            except ImportError:
                logger.warning("bitsandbytes não instalado — ignorando quantização.")

        _dtype = getattr(torch, torch_dtype, "auto") if torch_dtype != "auto" else "auto"

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            device_map=device,
            torch_dtype=_dtype,
            trust_remote_code=trust_remote_code,
            **quant_kwargs,
        )
        logger.info("Modelo %s carregado com sucesso.", model)

    @property
    def model(self) -> str:
        return self._model_id

    @property
    def provider(self) -> str:
        return "huggingface"

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
            outputs = self._pipeline(
                messages,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
                **kwargs,
            )
            content = outputs[0]["generated_text"]
            # Alguns pipelines retornam lista de mensagens
            if isinstance(content, list):
                content = content[-1].get("content", "")
        except Exception as exc:
            logger.error("Erro na inferência HuggingFace (%s): %s", self._model_id, exc)
            raise

        return LLMResponse(
            content=str(content).strip(),
            model=self._model_id,
            provider="huggingface",
            prompt_tokens=0,   # Transformers não expõe usage facilmente
            completion_tokens=0,
            finish_reason="stop",
            latency_ms=time.perf_counter() * 1000 - start,
        )


def build(config: Dict[str, Any]) -> HuggingFaceClient:
    model = config.get("model")
    if not model:
        raise ValueError("Provider 'huggingface' requer a chave 'model'.")
    return HuggingFaceClient(
        model=model,
        device=config.get("device", "auto"),
        load_in_4bit=config.get("load_in_4bit", False),
        load_in_8bit=config.get("load_in_8bit", False),
        trust_remote_code=config.get("trust_remote_code", False),
        torch_dtype=config.get("torch_dtype", "auto"),
    )
