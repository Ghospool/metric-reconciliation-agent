"""Deterministic offline client for tests, demos and CI. Takes a handler callable."""
from __future__ import annotations

from typing import Any, Callable

from .llm import LLMResponse, Usage

Handler = Callable[[dict[str, Any]], LLMResponse]


class MockClient:
    def __init__(self, handler: Handler) -> None:
        self._handler = handler
        self.calls: list[dict[str, Any]] = []
        self.total_usage = Usage()

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        resp = self._handler(kwargs)
        resp.model = kwargs.get("model", "mock")
        if not resp.usage.input_tokens:  # rough token estimate for cost tracking
            n_in = len(kwargs.get("system", "")) + sum(len(str(m["content"])) for m in kwargs["messages"])
            resp.usage = Usage(n_in // 4, max(1, len(resp.text) // 4))
        self.total_usage.add(resp.usage)
        return resp
