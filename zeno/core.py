from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .logging import VerboseLogger
from .models import OpenAICompatibleChatModel
from .types import ChatModel, Message, Tool, ToolCall
from .vllm_family import DEFAULT_STARTUP_TIMEOUT, VllmFamilyManager, default_backend, default_model_name as platform_default_model_name


_THOUGHT_PATTERNS = (
    re.compile(r"<\|channel>thought(?P<thinking>[\s\S]*?)<channel\|>"),
    re.compile(r"<think>(?P<thinking>[\s\S]*?)</think>", re.IGNORECASE),
)
_INLINE_TOOL_CALL_PATTERN = re.compile(r"<\|tool_call>(?P<body>[\s\S]*?)<tool_call\|>")
_INLINE_TOOL_HEADER_PATTERN = re.compile(r"^call:(?P<name>[A-Za-z_][A-Za-z0-9_]*)\{(?P<arguments>[\s\S]*)\}$")
_INLINE_STRING_START = '<|"|>'
_INLINE_STRING_END = '<|"|>'
_INLINE_TOOL_START = "<|tool_call>"
_INLINE_TOOL_END = "<tool_call|>"
_THOUGHT_STREAM_STARTS = ("<|channel>thought", "<think>")
_THOUGHT_STREAM_ENDS = ("<channel|>", "</think>")


@dataclass(frozen=True)
class AgentResult:
    answer: str
    thinking: str = ""
    streamed: bool = False


class _StreamDisplayFilter:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside_tool_call = False
        self._inside_thought = False
        self._thought_end = ""
        self._thought_buffer = ""
        self.displayed_answer = ""

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output: list[str] = []
        while self._buffer:
            if self._inside_tool_call:
                end = self._buffer.find(_INLINE_TOOL_END)
                if end == -1:
                    keep = self._pending_prefix_len((_INLINE_TOOL_END,))
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._buffer = self._buffer[end + len(_INLINE_TOOL_END) :]
                self._inside_tool_call = False
                continue

            if self._inside_thought:
                end = self._buffer.find(self._thought_end)
                if end == -1:
                    keep = self._pending_prefix_len((self._thought_end,))
                    if keep and len(self._buffer) <= keep:
                        break
                    thinking = self._buffer[:-keep] if keep else self._buffer
                    self._thought_buffer += thinking
                    self._buffer = self._buffer[-keep:] if keep else ""
                    break
                self._thought_buffer += self._buffer[:end]
                output.append(self._format_thinking_block(self._thought_buffer))
                self._buffer = self._buffer[end + len(self._thought_end) :]
                self._inside_thought = False
                self._thought_end = ""
                self._thought_buffer = ""
                continue

            start = self._buffer.find(_INLINE_TOOL_START)
            thought_start, thought_token, thought_end = self._find_thought_start()
            starts = [(start, _INLINE_TOOL_START, "tool")]
            if thought_start != -1:
                starts.append((thought_start, thought_token, thought_end))
            starts = [candidate for candidate in starts if candidate[0] != -1]
            if starts:
                first_start, token, kind = min(starts, key=lambda item: item[0])
                normal = self._buffer[:first_start]
                output.append(normal)
                self.displayed_answer += normal
                self._buffer = self._buffer[first_start + len(token) :]
                if kind == "tool":
                    self._inside_tool_call = True
                else:
                    output.append("thinking:\n")
                    self._inside_thought = True
                    self._thought_end = kind
                continue

            keep = self._pending_prefix_len((_INLINE_TOOL_START, *_THOUGHT_STREAM_STARTS))
            if keep == len(self._buffer):
                break
            normal = self._buffer[:-keep] if keep else self._buffer
            output.append(normal)
            self.displayed_answer += normal
            self._buffer = self._buffer[-keep:] if keep else ""
            break

        return "".join(output)

    def finish(self) -> str:
        if self._inside_tool_call:
            text = ""
        elif self._inside_thought:
            text = self._format_thinking_block(self._thought_buffer + self._buffer)
        else:
            text = self._buffer
            self.displayed_answer += text
        self._buffer = ""
        return text

    def _find_thought_start(self) -> tuple[int, str, str]:
        matches = [
            (self._buffer.find(start), start, end)
            for start, end in zip(_THOUGHT_STREAM_STARTS, _THOUGHT_STREAM_ENDS, strict=True)
            if self._buffer.find(start) != -1
        ]
        return min(matches, key=lambda item: item[0]) if matches else (-1, "", "")

    def _format_thinking_block(self, text: str) -> str:
        thinking = text.strip()
        if not thinking:
            return ""
        return "".join(f"  {line}\n" for line in thinking.splitlines())

    def _pending_prefix_len(self, markers: tuple[str, ...]) -> int:
        pending = 0
        for marker in markers:
            for length in range(1, min(len(marker), len(self._buffer)) + 1):
                if self._buffer.endswith(marker[:length]):
                    pending = max(pending, length)
        return pending


def clean_model_output(content: str) -> str:
    return split_model_output(content).answer


def split_model_output(content: str) -> AgentResult:
    thinking: list[str] = []
    cleaned = content
    for pattern in _THOUGHT_PATTERNS:
        thinking.extend(match.group("thinking").strip() for match in pattern.finditer(cleaned) if match.group("thinking").strip())
        cleaned = pattern.sub("", cleaned)
    return AgentResult(answer=cleaned.strip(), thinking="\n\n".join(thinking))


def parse_inline_tool_calls(content: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _INLINE_TOOL_CALL_PATTERN.finditer(content):
        header = _INLINE_TOOL_HEADER_PATTERN.match(match.group("body").strip())
        if header is None:
            continue
        calls.append(ToolCall(name=header.group("name"), arguments=_parse_inline_arguments(header.group("arguments"))))
    return calls


def strip_inline_tool_calls(content: str) -> str:
    return _INLINE_TOOL_CALL_PATTERN.sub("", content).strip()


def _parse_inline_arguments(text: str) -> dict[str, object]:
    arguments: dict[str, object] = {}
    index = 0
    while index < len(text):
        while index < len(text) and text[index] in {",", " ", "\n", "\t"}:
            index += 1
        key_start = index
        while index < len(text) and (text[index].isalnum() or text[index] == "_"):
            index += 1
        if key_start == index:
            break
        key = text[key_start:index]
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ":":
            break
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        if not text.startswith(_INLINE_STRING_START, index):
            break
        index += len(_INLINE_STRING_START)
        value_end = text.find(_INLINE_STRING_END, index)
        if value_end == -1:
            break
        arguments[key] = text[index:value_end]
        index = value_end + len(_INLINE_STRING_END)
    return arguments


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
                result = split_model_output(response.content)
                inline_calls = parse_inline_tool_calls(result.answer)
                if inline_calls:
                    return AgentResult(answer=self._run_inline_tool_calls(inline_calls, strip_inline_tool_calls(result.answer)), thinking=result.thinking)
                return result

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

    def stream_messages_with_result(self, messages: list[Message], on_chunk: Callable[[str], object] | None = None) -> AgentResult | None:
        if not _supports_streaming(self.model):
            return None
        schemas = self._tool_schemas()
        chunks: list[str] = []
        display_filter = _StreamDisplayFilter()
        for chunk in self.model.stream_chat(messages, schemas):
            chunks.append(chunk)
            if on_chunk is not None:
                display = display_filter.feed(chunk)
                if display:
                    on_chunk(display)
        if on_chunk is not None:
            display = display_filter.finish()
            if display:
                on_chunk(display)
        content = "".join(chunks)
        result = split_model_output(content)
        inline_calls = parse_inline_tool_calls(result.answer)
        if inline_calls:
            answer = self._run_inline_tool_calls(inline_calls, strip_inline_tool_calls(result.answer))
            self._stream_missing_tail(answer, display_filter.displayed_answer, on_chunk)
            return AgentResult(answer=answer, thinking=result.thinking, streamed=on_chunk is not None)
        answer = result.answer
        self._stream_missing_tail(answer, display_filter.displayed_answer, on_chunk)
        return AgentResult(answer=answer, thinking=result.thinking, streamed=on_chunk is not None)

    def _tool_schemas(self) -> list[dict[str, object]] | None:
        if not self.tools:
            return None
        return [tool_schema(name, tool) for name, tool in self.tools.items()]

    def _call_tool(self, call: ToolCall) -> object:
        if not self.tools or call.name not in self.tools:
            raise RuntimeError(f"Unknown tool: {call.name}")
        return self.tools[call.name](**call.arguments)

    def _run_inline_tool_calls(self, calls: list[ToolCall], message: str) -> str:
        lines: list[str] = []
        if message:
            lines.append(message)
        for call in calls:
            result = self._call_tool(call)
            lines.append(f"tool {call.name} completed: {json.dumps(result, ensure_ascii=False)}")
        return "\n".join(lines).strip()

    def _stream_missing_tail(self, answer: str, displayed: str, on_chunk: Callable[[str], object] | None) -> None:
        if on_chunk is None:
            return
        if not answer or answer == displayed:
            return
        if displayed and answer.startswith(displayed):
            tail = answer[len(displayed) :]
        else:
            displayed_clean = displayed.strip()
            if not displayed_clean or not answer.startswith(displayed_clean):
                displayed_clean = ""
            tail = answer
        if tail:
            on_chunk(tail)

    def _tool_call_message(self, call: ToolCall, index: int) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "index": index,
                "name": call.name,
                "arguments": call.arguments,
            },
        }


def _supports_streaming(model: ChatModel) -> bool:
    return callable(getattr(model, "stream_chat", None))


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
