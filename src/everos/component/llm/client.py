"""Process-wide LLM client accessor.

Lazy singleton — first call reads settings and builds the algo LLM
client; subsequent calls return the cached instance. Raises
:class:`LLMNotConfiguredError` when no credentials are present so
misconfiguration surfaces at app startup (via the LLM lifespan
provider) instead of silently failing per-request downstream.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from everalgo.llm import ChatMessage, ChatResponse, build_client
from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient
from pydantic import BaseModel

from everos.config import load_settings
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_PROMPT_PREVIEW_LIMIT = 400
_RESPONSE_PREVIEW_LIMIT = 240


def _normalize_preview_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _content_preview(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_content_preview(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return _content_preview(content["text"])
        return ", ".join(
            f"{key}={_content_preview(value)}" for key, value in content.items()
        )
    return str(content)


def _message_preview(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages[:4]):
        role = getattr(message, "role", None)
        if role is None and isinstance(message, dict):
            role = message.get("role")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        parts.append(f"{index}:{role or 'unknown'}={_content_preview(content)}")
    if len(messages) > 4:
        parts.append(f"...(+{len(messages) - 4} more messages)")
    return _normalize_preview_text(" | ".join(parts), limit=_PROMPT_PREVIEW_LIMIT)


def _response_preview(response: ChatResponse) -> str:
    return _normalize_preview_text(response.content, limit=_RESPONSE_PREVIEW_LIMIT)


class _LoggingLLMClient:
    """Transparent proxy that adds per-request chat logs."""

    def __init__(self, inner: LLMClient, *, client_name: str, default_model: str):
        self._inner = inner
        self._client_name = client_name
        self._default_model = default_model

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        request_model = model or self._default_model
        started_at = perf_counter()
        logger.debug(
            "llm_chat_started",
            client_name=self._client_name,
            model=request_model,
            message_count=len(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=(
                response_format.__name__ if response_format is not None else None
            ),
            extra_keys=sorted(extra),
            prompt_preview=_message_preview(messages),
        )
        try:
            response = await self._inner.chat(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **extra,
            )
        except Exception:
            logger.exception(
                "llm_chat_failed",
                client_name=self._client_name,
                model=request_model,
                message_count=len(messages),
                duration_ms=round((perf_counter() - started_at) * 1000, 1),
                prompt_preview=_message_preview(messages),
            )
            raise

        usage = response.usage
        logger.debug(
            "llm_chat_completed",
            client_name=self._client_name,
            model=request_model,
            provider_model=response.model,
            message_count=len(messages),
            duration_ms=round((perf_counter() - started_at) * 1000, 1),
            finish_reason=response.finish_reason,
            prompt_tokens=(usage.prompt_tokens if usage is not None else None),
            completion_tokens=(usage.completion_tokens if usage is not None else None),
            response_preview=_response_preview(response),
        )
        return response


def _wrap_llm_client(
    client: LLMClient, *, client_name: str, default_model: str
) -> LLMClient:
    return _LoggingLLMClient(
        client,
        client_name=client_name,
        default_model=default_model,
    )


class LLMNotConfiguredError(RuntimeError):
    """Raised when ``settings.llm`` is missing ``api_key`` or ``base_url``."""


_llm_client: LLMClient | None = None
_multimodal_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton algo LLM client.

    Raises:
        LLMNotConfiguredError: When ``settings.llm.api_key`` or
            ``settings.llm.base_url`` is unset.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    llm_cfg = load_settings().llm
    api_key = (
        llm_cfg.api_key.get_secret_value() if llm_cfg.api_key is not None else None
    )
    if not api_key or not llm_cfg.base_url:
        raise LLMNotConfiguredError(
            "LLM is required; set EVEROS_LLM__API_KEY + EVEROS_LLM__BASE_URL"
        )
    _llm_client = _wrap_llm_client(
        build_client(
            LLMConfig(
                model=llm_cfg.model,
                api_key=api_key,
                base_url=llm_cfg.base_url,
                temperature=llm_cfg.temperature,
                timeout=llm_cfg.timeout_seconds,
            )
        ),
        client_name="default",
        default_model=llm_cfg.model,
    )
    logger.info("llm_client_built", model=llm_cfg.model)
    return _llm_client


def get_multimodal_llm_client() -> LLMClient:
    """Return the singleton multimodal LLM client (for everalgo.parser).

    Reads the flat ``[multimodal]`` config — kept separate from the main
    ``[llm]`` so parsing can target a vision/audio-capable endpoint.

    Raises:
        LLMNotConfiguredError: When ``settings.multimodal.api_key`` or
            ``settings.multimodal.base_url`` is unset.
    """
    global _multimodal_client
    if _multimodal_client is not None:
        return _multimodal_client

    cfg = load_settings().multimodal
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            "Multimodal LLM is required for parsing; set "
            "EVEROS_MULTIMODAL__API_KEY + EVEROS_MULTIMODAL__BASE_URL"
        )
    _multimodal_client = _wrap_llm_client(
        build_client(
            LLMConfig(
                model=cfg.model,
                api_key=api_key,
                base_url=cfg.base_url,
            )
        ),
        client_name="multimodal",
        default_model=cfg.model,
    )
    logger.info("multimodal_llm_client_built", model=cfg.model)
    return _multimodal_client
