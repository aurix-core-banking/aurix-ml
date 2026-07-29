# Copyright (c) 2025 Aurix Platform
# SPDX-License-Identifier: Apache-2.0
"""Provider AWS Bedrock — modelos gerenciados na nuvem AWS.

Suporta os principais families de modelos disponíveis no Bedrock:
- Amazon Nova (nova-lite, nova-micro, nova-pro)
- Anthropic Claude (claude-3-5-sonnet, claude-3-haiku, ...)
- Meta Llama (llama3-70b, llama3-8b, ...)
- Mistral AI (mistral-large, mistral-small, ...)
- Cohere Command R

Requires:
    pip install boto3>=1.34
    # Credenciais via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
    # ou IAM Role (recomendado em produção)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from aurix_ml.llm.base import LLMClient, LLMResponse, Message, split_system_messages

logger = logging.getLogger("aurix_ml.llm.providers.bedrock")

# Mapeamento de famílias de modelos → formato de request/response
_CONVERSE_SUPPORTED = True  # Bedrock Converse API — formato unificado (recomendado)


class BedrockClient(LLMClient):
    """Cliente para AWS Bedrock usando a API Converse (formato unificado).

    A API Converse abstrai as diferenças de formato entre providers
    (Anthropic, Amazon, Meta, Mistral) — usamos ela sempre que possível.

    Args:
        model: Model ID do Bedrock (ex: ``amazon.nova-lite-v1:0``).
        region: Região AWS (ex: ``us-east-1``, ``sa-east-1`` para Brasil).
        profile: Perfil AWS (~/.aws/credentials). None = default.
        assume_role_arn: ARN de role para assume-role (para cross-account).
    """

    def __init__(
        self,
        model: str = "amazon.nova-lite-v1:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        assume_role_arn: Optional[str] = None,
        timeout: int = 60,
        **kwargs: Any,
    ):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("Provider 'bedrock' requer: pip install boto3") from exc

        self._model = model
        self._region = region

        session_kwargs: Dict[str, Any] = {}
        if profile:
            session_kwargs["profile_name"] = profile

        session = boto3.Session(**session_kwargs)

        if assume_role_arn:
            sts = session.client("sts")
            creds = sts.assume_role(
                RoleArn=assume_role_arn,
                RoleSessionName="aurix-ml-llm",
            )["Credentials"]
            session = boto3.Session(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            )

        self._client = session.client(
            "bedrock-runtime",
            region_name=region,
            config=__import__("botocore").config.Config(
                read_timeout=timeout,
                connect_timeout=10,
            ),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "bedrock"

    def chat(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LLMResponse:
        system_text, turns = split_system_messages(messages)

        # Montar mensagens no formato Converse
        converse_messages = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in turns
        ]

        request: Dict[str, Any] = {
            "modelId": self._model,
            "messages": converse_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_text:
            request["system"] = [{"text": system_text}]

        start = time.perf_counter() * 1000
        try:
            response = self._client.converse(**request)
        except Exception as exc:
            logger.error("Erro no Bedrock Converse (%s): %s", self._model, exc)
            raise

        content = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})
        stop_reason = response.get("stopReason", "stop")

        return LLMResponse(
            content=content,
            model=self._model,
            provider="bedrock",
            prompt_tokens=usage.get("inputTokens", 0),
            completion_tokens=usage.get("outputTokens", 0),
            finish_reason=stop_reason,
            latency_ms=time.perf_counter() * 1000 - start,
            raw=None,
        )


def build(config: Dict[str, Any]) -> BedrockClient:
    return BedrockClient(
        model=config.get("model", "amazon.nova-lite-v1:0"),
        region=config.get("region", "us-east-1"),
        profile=config.get("profile"),
        assume_role_arn=config.get("assume_role_arn"),
        timeout=config.get("timeout", 60),
    )
