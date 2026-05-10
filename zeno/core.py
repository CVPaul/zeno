from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .logging import VerboseLogger
from .models import OpenAICompatibleChatModel
from .types import ChatModel, Message, Tool, ToolCall
from .vllm_family import DEFAULT_STARTUP_TIMEOUT, VllmFamilyManager, default_backend, default_model_name as platform_default_model_name


_THOUGHT_PATTERNS = (
    re.compile(r"<\|channel>thought(?P<thinking>[\s\S]*?)<channel\|>"),
    re.compile(r"<think>(?P<thinking>[\s\S]*?)</think>", re.IGNORECASE),
)


@dataclass(frozen=True)
class AgentResult:
    answer: str
    thinking: str = ""


def clean_model_output(content: str) -> str:
    return split_model_output(content).answer


def split_model_output(content: str) -> AgentResult:
    thinking: list[str] = []
    cleaned = content
    for pattern in _THOUGHT_PATTERNS:
        thinking.extend(match.group("thinking").strip() for match in pattern.finditer(cleaned) if match.group("thinking").strip())
        cleaned = pattern.sub("", cleaned)
    return AgentResult(answer=cleaned.strip(), thinking="\n\n".join(thinking))


def _json_type(annotation: object) -> str:
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    return "string"


def tool_schema(name: str, tool: Tool) -> dict[str, object]:
    signature = inspect.signature(tool)
    properties: dict[str, object] = {}
    required: list[str] = []
    for parameter_name, parameter in signature.parameters.items():
        properties[parameter_name] = {"type": _json_type(parameter.annotation)}
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter_name)

    description = inspect.getdoc(tool) or f"Call the {name} tool."
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


@dataclass
class Agent:
    model: ChatModel
    system: str = "You are a helpful, concise assistant."
    tools: Mapping[str, Tool] | None = None
    max_steps: int = 5

    def run(self, prompt: str) -> str:
        messages: list[Message] = [
            {"role": "system", "content": self.system},
            {"role": "user", "content": prompt},
        ]
        return self.run_messages(messages)

    def run_messages(self, messages: list[Message]) -> str:
        return self.run_messages_with_result(messages).answer

    def run_messages_with_result(self, messages: list[Message]) -> AgentResult:
        schemas = self._tool_schemas()

        for _ in range(self.max_steps):
            response = self.model.chat(messages, schemas)
            if not response.tool_calls:
                return split_model_output(response.content)

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [self._tool_call_message(call, index) for index, call in enumerate(response.tool_calls)],
                }
            )
            for call in response.tool_calls:
                result = self._call_tool(call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        raise RuntimeError("Agent reached max_steps before producing a final answer")

    def _tool_schemas(self) -> list[dict[str, object]] | None:
        if not self.tools:
            return None
        return [tool_schema(name, tool) for name, tool in self.tools.items()]

    def _call_tool(self, call: ToolCall) -> object:
        if not self.tools or call.name not in self.tools:
            raise RuntimeError(f"Unknown tool: {call.name}")
        return self.tools[call.name](**call.arguments)

    def _tool_call_message(self, call: ToolCall, index: int) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "index": index,
                "name": call.name,
                "arguments": call.arguments,
            },
        }


def default_model_name(model: str | None = None, backend: str | None = None, log: VerboseLogger | None = None) -> str:
    return platform_default_model_name(model or os.environ.get("ZENO_MODEL"), backend, log=log)


def default_local_model(
    model: str | None = None,
    backend: str | None = None,
    log: VerboseLogger | None = None,
    device: str | None = None,
    startup_timeout: float | None = None,
) -> OpenAICompatibleChatModel:
    selected_backend = backend or os.environ.get("ZENO_BACKEND") or default_backend()
    if log is not None:
        log(f"selected backend: {selected_backend}")
    selected_model = default_model_name(model, selected_backend, log=log)
    return OpenAICompatibleChatModel(model=selected_model, base_url="http://localhost:8000/v1")


def ensure_default_local_model(
    model: str | None = None,
    backend: str | None = None,
    log: VerboseLogger | None = None,
    device: str | None = None,
    startup_timeout: float | None = None,
) -> OpenAICompatibleChatModel:
    selected_backend = backend or os.environ.get("ZENO_BACKEND") or default_backend()
    if log is not None:
        log(f"selected backend: {selected_backend}")
    selected_model = default_model_name(model, selected_backend, log=log)
    selected_device = device or os.environ.get("ZENO_DEVICE")
    if log is not None and selected_device is not None:
        log(f"selected device override: {selected_device}")
    selected_timeout = startup_timeout or _float_env("ZENO_STARTUP_TIMEOUT")
    if log is not None and selected_timeout is not None:
        log(f"selected startup timeout: {selected_timeout:.0f}s")
    manager = VllmFamilyManager(
        model=selected_model,
        backend=selected_backend,
        log=log,
        device=selected_device,
        startup_timeout=selected_timeout if selected_timeout is not None else DEFAULT_STARTUP_TIMEOUT,
    )
    manager.ensure_ready()
    if log is not None:
        log(f"using OpenAI-compatible endpoint: {manager.openai_base_url()}")
    return OpenAICompatibleChatModel(model=selected_model, base_url=manager.openai_base_url())


def _float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number of seconds") from exc
