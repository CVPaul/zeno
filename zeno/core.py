from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from .models import OpenAICompatibleChatModel
from .types import ChatModel, Message, Tool, ToolCall
from .vllm_family import VllmFamilyManager, default_backend, default_model_name as platform_default_model_name


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
        schemas = self._tool_schemas()

        for _ in range(self.max_steps):
            response = self.model.chat(messages, schemas)
            if not response.tool_calls:
                return response.content.strip()

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


def default_model_name(model: str | None = None, backend: str | None = None) -> str:
    return platform_default_model_name(model or os.environ.get("ZENO_MODEL"), backend)


def default_local_model(model: str | None = None, backend: str | None = None) -> OpenAICompatibleChatModel:
    selected_backend = backend or os.environ.get("ZENO_BACKEND") or default_backend()
    selected_model = default_model_name(model, selected_backend)
    return OpenAICompatibleChatModel(model=selected_model, base_url="http://localhost:8000/v1")


def ensure_default_local_model(model: str | None = None, backend: str | None = None) -> OpenAICompatibleChatModel:
    selected_backend = backend or os.environ.get("ZENO_BACKEND") or default_backend()
    selected_model = default_model_name(model, selected_backend)
    manager = VllmFamilyManager(model=selected_model, backend=selected_backend)
    manager.ensure_ready()
    return OpenAICompatibleChatModel(model=selected_model, base_url=manager.openai_base_url())
