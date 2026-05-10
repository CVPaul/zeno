from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterable

from .core import Agent, AgentResult, ensure_default_local_model
from .config import ConfigStore
from .logging import verbose_logger
from .vllm_family import default_backend
from .sessions import SessionStore
from .tools import default_tools
from .types import Message


COMPACT_AFTER_MESSAGES = 24
KEEP_RECENT_MESSAGES = 12
TYPEWRITER_DELAY = 0.005


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeno", description="Minimal local-first agent CLI")
    parser.add_argument("--backend", choices=["vllm-mlx", "vllm"], help="Model backend to manage for this session")
    parser.add_argument("--model", help="Model to use for this chat session")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="vLLM device override for debugging or CPU-only environments")
    parser.add_argument("--startup-timeout", type=float, help="Seconds to wait for backend startup and first-run model downloads")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show backend startup and readiness details")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue the latest task session in this workspace")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("serve", help="Start the local model server and keep it running")

    task = subparsers.add_parser("task", help="Run one-off tasks")
    task_subparsers = task.add_subparsers(dest="task_command", required=True)
    create = task_subparsers.add_parser("create", help="Run a one-off task and record it as a session")
    create.add_argument("description", help="Task description to send to the model")
    task_subparsers.add_parser("list", help="List task history")

    return parser


def run_chat(agent: Agent, store: SessionStore | None = None, session_id: str | None = None) -> int:
    session_store = store or SessionStore()
    session_id = session_id or session_store.create()
    print(f"zeno task: {session_id}")
    print("type exit / quit / empty line to leave")
    while True:
        try:
            prompt = input("zeno> ").strip()
        except EOFError:
            break
        if is_exit_prompt(prompt):
            break
        session_store.append(session_id, "user", prompt)
        try:
            result = run_with_session_history(agent, session_store, session_id)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        session_store.append(session_id, "assistant", result.answer)
        print_agent_result(result)
    return 0


def list_tasks(store: SessionStore | None = None) -> int:
    session_store = store or SessionStore()
    sessions = session_store.list()
    if not sessions:
        print("No tasks found.")
        return 0

    print("ID\tUPDATED\tMESSAGES\tTITLE")
    for session in sessions:
        print(f"{session.session_id}\t{session.updated_at}\t{session.message_count}\t{session.title}")
    return 0


def is_exit_prompt(prompt: str) -> bool:
    return prompt.strip().lower() in {"", "exit", "exit()", "quit", "quit()"}


def run_task(description: str, agent: Agent, store: SessionStore | None = None) -> int:
    session_store = store or SessionStore()
    session_id = session_store.create()
    session_store.append(session_id, "user", description)
    try:
        result = run_with_session_history(agent, session_store, session_id)
    except RuntimeError as exc:
        session_store.delete(session_id)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    session_store.append(session_id, "assistant", result.answer)
    print_agent_result(result)
    return 0


def run_with_session_history(agent: Agent, store: SessionStore, session_id: str) -> AgentResult:
    messages: list[Message] = [{"role": "system", "content": agent.system}]
    for message in compact_messages(store.messages(session_id)):
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    streaming_result = agent.stream_messages_with_result(messages, on_chunk=stream_chunk)
    if streaming_result is not None:
        return streaming_result
    return agent.run_messages_with_result(messages)


def print_agent_result(result: AgentResult) -> None:
    if result.streamed:
        if result.answer:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return
    if result.thinking:
        print_thinking(result.thinking)
    typewriter_print(result.answer)


def stream_chunk(chunk: str) -> None:
    sys.stdout.write(chunk)
    sys.stdout.flush()


def print_thinking(thinking: str) -> None:
    lines = ["thinking:"]
    lines.extend(f"  {line}" for line in thinking.strip().splitlines())
    typewriter_print("\n".join(lines), delay=TYPEWRITER_DELAY)


def compact_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(messages) <= COMPACT_AFTER_MESSAGES:
        return messages
    recent = messages[-KEEP_RECENT_MESSAGES:]
    older = messages[:-KEEP_RECENT_MESSAGES]
    summary = summarize_messages(older)
    if not summary:
        return recent
    return [{"role": "assistant", "content": summary}] + recent


def summarize_messages(messages: Iterable[dict[str, object]]) -> str:
    lines: list[str] = ["Earlier conversation summary:"]
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        compact = " ".join(content.strip().split())
        if len(compact) > 240:
            compact = f"{compact[:237]}..."
        lines.append(f"- {role}: {compact}")
    return "\n".join(lines) if len(lines) > 1 else ""


def typewriter_print(text: str, delay: float = TYPEWRITER_DELAY, write: Callable[[str], object] | None = None, flush: Callable[[], object] | None = None) -> None:
    writer = write or sys.stdout.write
    flusher = flush or sys.stdout.flush
    if not text:
        writer("\n")
        flusher()
        return
    for character in text:
        writer(character)
        flusher()
        if delay > 0:
            time.sleep(delay)
    writer("\n")
    flusher()


def main(argv: list[str] | None = None, agent: Agent | None = None, store: SessionStore | None = None, config: ConfigStore | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log = verbose_logger(args.verbose)
    config_store = config or ConfigStore()
    if log is not None:
        log(f"command={args.command or 'chat'} backend={args.backend or 'auto'} model={args.model or 'auto'}")

    if args.command is None or args.command in {"task", "serve"}:
        if args.command == "task" and args.task_command == "list":
            return list_tasks(store)
        if args.command is None and args.continue_session:
            session_store = store or SessionStore()
            session_id = session_store.latest_id()
            if session_id is None:
                print("error: no task sessions found in this workspace", file=sys.stderr)
                return 1
            try:
                selected_model = resolve_model_choice(args.model, args.backend, log, config_store)
                selected_agent = agent or default_agent(_ensure_model(selected_model, args.backend, log, args.device, args.startup_timeout))
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return run_chat(selected_agent, session_store, session_id=session_id)
        try:
            selected_model = resolve_model_choice(args.model, args.backend, log, config_store)
            selected_agent = agent or default_agent(_ensure_model(selected_model, args.backend, log, args.device, args.startup_timeout))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.command == "serve":
            print(f"serving model: {selected_agent.model.model}")
            return 0
        if args.command == "task" and args.task_command == "create":
            return run_task(args.description, selected_agent, store)
        return run_chat(selected_agent, store)

    parser.error(f"unknown command: {args.command}")
    return 2


def _ensure_model(model: str | None, backend: str | None, log: object | None, device: str | None, startup_timeout: float | None) -> object:
    if log is None:
        if device is None and startup_timeout is None:
            return ensure_default_local_model(model, backend)
        if startup_timeout is None:
            return ensure_default_local_model(model, backend, device=device)
        if device is None:
            return ensure_default_local_model(model, backend, startup_timeout=startup_timeout)
        return ensure_default_local_model(model, backend, device=device, startup_timeout=startup_timeout)
    if device is None and startup_timeout is None:
        return ensure_default_local_model(model, backend, log=log)
    if startup_timeout is None:
        return ensure_default_local_model(model, backend, log=log, device=device)
    if device is None:
        return ensure_default_local_model(model, backend, log=log, startup_timeout=startup_timeout)
    return ensure_default_local_model(model, backend, log=log, device=device, startup_timeout=startup_timeout)


def default_agent(model: object) -> Agent:
    return Agent(
        model=model,
        system=(
            "You are a helpful, concise local coding agent. "
            "When the user asks you to implement code or create a file, use the write_file tool instead of only describing the code."
        ),
        tools=default_tools(),
    )


def resolve_model_choice(model: str | None, backend: str | None, log: object | None, config: ConfigStore) -> str | None:
    selected_backend = backend or default_backend()
    if model:
        config.save_model(selected_backend, model)
        return model
    saved_model = config.model_for_backend(selected_backend)
    if saved_model:
        if log is not None:
            log(f"using saved model for {selected_backend}: {saved_model}")
        return saved_model
    return None
