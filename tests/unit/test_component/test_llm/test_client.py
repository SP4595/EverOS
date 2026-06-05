"""get_llm_client — raises on missing credentials, caches on success."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
from pydantic import SecretStr

from everos.component.llm import ChatResponse, Usage
from everos.component.llm import LLMNotConfiguredError
from everos.config import Settings
from everos.config.settings import LLMSettings

_client_mod = importlib.import_module("everos.component.llm.client")


def _reset_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_client_mod, "_llm_client", None, raising=False)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str | None,
    base_url: str | None,
    temperature: float = 0.0,
) -> None:
    """Stub the ``load_settings`` reference bound inside the client module."""
    cfg = Settings(
        llm=LLMSettings(
            model="gpt-4o-mini",
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=base_url,
            temperature=temperature,
        )
    )
    monkeypatch.setattr(_client_mod, "load_settings", lambda: cfg)


def test_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key=None, base_url="https://example.test")

    with pytest.raises(LLMNotConfiguredError, match="EVEROS_LLM__API_KEY"):
        _client_mod.get_llm_client()


def test_raises_when_base_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url=None)

    with pytest.raises(LLMNotConfiguredError, match="EVEROS_LLM__BASE_URL"):
        _client_mod.get_llm_client()


def test_returns_singleton_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(
        monkeypatch,
        api_key="sk-test",
        base_url="https://example.test",
        temperature=0.6,
    )
    sentinel = object()
    captured = {}

    def _build_client(cfg: Any) -> object:
        captured["cfg"] = cfg
        return sentinel

    monkeypatch.setattr(_client_mod, "build_client", _build_client)

    first = _client_mod.get_llm_client()
    second = _client_mod.get_llm_client()

    assert first is not sentinel
    assert first is second
    assert captured["cfg"].temperature == 0.6


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[str, dict[str, Any]]] = []
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.exception_calls: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **kwargs: Any) -> None:
        self.debug_calls.append((event, kwargs))

    def info(self, event: str, **kwargs: Any) -> None:
        self.info_calls.append((event, kwargs))

    def exception(self, event: str, **kwargs: Any) -> None:
        self.exception_calls.append((event, kwargs))


class _FakeInnerClient:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> ChatResponse:
        self.calls.append((messages, kwargs))
        return ChatResponse(
            content="You like climbing at the indoor bouldering gym.",
            model="qwen3:30b-instruct",
            usage=Usage(prompt_tokens=123, completion_tokens=17),
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_chat_proxy_logs_and_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    _reset_singleton(monkeypatch)
    _patch_settings(monkeypatch, api_key="sk-test", base_url="https://example.test")
    inner = _FakeInnerClient()
    fake_logger = _FakeLogger()
    monkeypatch.setattr(_client_mod, "build_client", lambda cfg: inner)
    monkeypatch.setattr(_client_mod, "logger", fake_logger)

    client = _client_mod.get_llm_client()
    response = await client.chat(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Where do I like to climb?"},
        ],
        temperature=0.2,
        max_tokens=64,
    )

    assert response.content == "You like climbing at the indoor bouldering gym."
    assert inner.calls == [
        (
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Where do I like to climb?"},
            ],
            {
                "model": None,
                "temperature": 0.2,
                "max_tokens": 64,
                "response_format": None,
            },
        )
    ]

    started_event, started_fields = fake_logger.debug_calls[0]
    completed_event, completed_fields = fake_logger.debug_calls[1]

    assert started_event == "llm_chat_started"
    assert started_fields["client_name"] == "default"
    assert started_fields["message_count"] == 2
    assert "Where do I like to climb?" in started_fields["prompt_preview"]

    assert completed_event == "llm_chat_completed"
    assert completed_fields["provider_model"] == "qwen3:30b-instruct"
    assert completed_fields["prompt_tokens"] == 123
    assert completed_fields["completion_tokens"] == 17
    assert "indoor bouldering gym" in completed_fields["response_preview"]
