from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field

from .types import ChatResponse, Message, ToolCall


DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


def _post_json(url: str, payload: Mapping[str, object], timeout: float) -> dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach local model endpoint: {url}") from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model endpoint returned invalid JSON: {body[:200]}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("Model endpoint returned a non-object JSON response")
    return parsed


def _parse_tool_calls(raw_calls: object) -> list[ToolCall]:
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise RuntimeError("Model response tool_calls must be a list")

    calls: list[ToolCall] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            raise RuntimeError("Model response tool_call must be an object")
        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("Model response tool_call is missing function")
        name = function.get("name")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError("Model response tool_call arguments are invalid JSON") from exc
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise RuntimeError("Model response tool_call has invalid function shape")
        calls.append(ToolCall(name=name, arguments=arguments))
    return calls


@dataclass
class MLXChatModel:
    model: str = DEFAULT_MLX_MODEL
    max_tokens: int = 4096
    _runtime: object | None = field(default=None, init=False, repr=False)

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        if tools:
            raise RuntimeError("MLX backend does not support native tool calls yet")
        runtime = self._load_runtime()
        try:
            response = runtime.chat(messages, max_tokens=self.max_tokens)
        except TypeError:
            response = runtime.chat(messages)
        text = getattr(response, "text", response)
        if not isinstance(text, str):
            raise RuntimeError("MLX backend returned a non-text response")
        return ChatResponse(content=text, tool_calls=[])

    def _load_runtime(self) -> object:
        if self._runtime is not None:
            return self._runtime
        try:
            from vllm_mlx.models import MLXLanguageModel
        except ImportError as exc:
            raise RuntimeError("vllm-mlx is not installed. Install it with `python -m pip install vllm-mlx`.") from exc
        runtime = MLXLanguageModel(self.model)
        runtime.load()
        self._runtime = runtime
        return runtime


@dataclass(frozen=True)
class OllamaChatModel:
    model: str = "qwen3:14b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        response = _post_json(f"{self.base_url.rstrip('/')}/api/chat", payload, self.timeout)
        message = response.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama response is missing message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("Ollama response message content must be a string")
        return ChatResponse(content=content, tool_calls=_parse_tool_calls(message.get("tool_calls")))


@dataclass(frozen=True)
class OpenAICompatibleChatModel:
    model: str
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local"
    timeout: float = 120.0

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None = None,
    ) -> ChatResponse:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach local model endpoint: {self.base_url}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model endpoint returned invalid JSON: {body[:200]}") from exc

        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible response is missing choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("OpenAI-compatible choice is not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible choice is missing message")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise RuntimeError("OpenAI-compatible message content must be a string")
        return ChatResponse(content=content, tool_calls=_parse_tool_calls(message.get("tool_calls")))
