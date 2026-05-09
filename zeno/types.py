from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


Message = dict[str, object]
Tool = Callable[..., object]


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]


class ChatModel(Protocol):
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        """Return the assistant response for a list of chat messages."""
