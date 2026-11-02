"""Thin, typed LLM abstraction: swap real and mock clients without touching workflow code."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings, get_settings

log = logging.getLogger(__name__)


class TransientError(Exception):
    """Retryable failure (rate limit, timeout, 5xx). Real SDK errors are matched by name in is_transient."""


_TRANSIENT_NAMES = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
                    "OverloadedError", "TransientError"}


def is_transient(exc: BaseException) -> bool:
    return any(c.__name__ in _TRANSIENT_NAMES for c in type(exc).__mro__)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    id: str = ""


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""


class LLMClient(Protocol):
    total_usage: Usage

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        force_tool: str | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Async Anthropic client with built-in retries/timeouts and usage accounting."""

    def __init__(self, settings: Settings | None = None) -> None:
        from anthropic import AsyncAnthropic  # lazy: keeps offline use dependency-light

        s = settings or get_settings()
        if not s.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Use MockClient for offline runs.")
        self._client = AsyncAnthropic(
            api_key=s.api_key, max_retries=s.max_retries, timeout=s.timeout_s
        )
        self.total_usage = Usage()

    async def complete(self, *, system, messages, model, max_tokens=1024, temperature=0.0,
                       tools=None, force_tool=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=model, system=system, messages=messages,
                                      max_tokens=max_tokens, temperature=temperature)
        if tools:
            kwargs["tools"] = tools
            if force_tool:
                kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        resp = await self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [ToolCall(b.name, dict(b.input), b.id) for b in resp.content if b.type == "tool_use"]
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        self.total_usage.add(usage)
        log.debug("llm call model=%s in=%d out=%d", model, usage.input_tokens, usage.output_tokens)
        return LLMResponse(text=text, tool_calls=calls, usage=usage, model=model)
