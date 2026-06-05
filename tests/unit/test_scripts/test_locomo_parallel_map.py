"""Focused tests for serial progress visibility in ``tests/test_locomo.py``."""

from __future__ import annotations

import importlib
from typing import Any


def test_parallel_map_uses_tqdm_in_serial_mode(monkeypatch) -> None:
    locomo = importlib.import_module("tests.test_locomo")
    calls: list[dict[str, Any]] = []

    def fake_tqdm(iterable, **kwargs):
        calls.append(kwargs)
        return iterable

    monkeypatch.setattr(locomo, "_tqdm", fake_tqdm)

    results = locomo._parallel_map(
        [10, 20, 30],
        lambda i, item: (i, item * 2),
        desc="Answer",
        total=3,
        quiet=False,
        concurrency=1,
    )

    assert results == [(0, 20), (1, 40), (2, 60)]
    assert calls == [
        {
            "total": 3,
            "desc": "Answer",
            "unit": "item",
            "dynamic_ncols": True,
        }
    ]


def test_build_chat_completion_kwargs_adds_ollama_safeguards() -> None:
    locomo = importlib.import_module("tests.test_locomo")

    kwargs = locomo._build_chat_completion_kwargs(
        model="qwen3:30b-instruct",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.6,
        timeout=900,
        max_tokens=256,
        base_url="http://127.0.0.1:11434/v1",
        ollama_repeat_penalty=1.1,
    )

    assert kwargs["max_tokens"] == 256
    assert kwargs["extra_body"] == {"options": {"repeat_penalty": 1.1}}


def test_build_chat_completion_kwargs_skips_ollama_options_for_other_backends() -> None:
    locomo = importlib.import_module("tests.test_locomo")

    kwargs = locomo._build_chat_completion_kwargs(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        timeout=60,
        max_tokens=None,
        base_url="https://api.openai.com/v1",
        ollama_repeat_penalty=1.1,
    )

    assert "max_tokens" not in kwargs
    assert "extra_body" not in kwargs