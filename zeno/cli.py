from __future__ import annotations

import argparse
import sys

from .core import Agent, ensure_default_local_model
from .config import ConfigStore
from .llmfit import recommend_models
from .logging import verbose_logger
from .vllm_family import default_backend
from .sessions import SessionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeno", description="Minimal local-first agent CLI")
    parser.add_argument("--backend", choices=["vllm-mlx", "vllm"], help="Model backend to manage for this session")
    parser.add_argument("--model", help="Model to use for this chat session")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], help="vLLM device override for debugging or CPU-only environments")
    parser.add_argument("--startup-timeout", type=float, help="Seconds to wait for backend startup and first-run model downloads")
    parser.add_argument("--select-model", action="store_true", help="Interactively choose from llmfit model recommendations before starting")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show backend startup and readiness details")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue the latest task session in this workspace")
    subparsers = parser.add_subparsers(dest="command")

    models = subparsers.add_parser("models", help="List llmfit model recommendations")
    models.add_argument("--limit", type=int, default=5, help="Number of candidate models to show")

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
        if prompt in {"", "exit", "quit"}:
            break
        session_store.append(session_id, "user", prompt)
        try:
            answer = run_with_session_history(agent, session_store, session_id)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        session_store.append(session_id, "assistant", answer)
        print(answer)
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


def run_task(description: str, agent: Agent, store: SessionStore | None = None) -> int:
    session_store = store or SessionStore()
    session_id = session_store.create()
    session_store.append(session_id, "user", description)
    try:
        answer = run_with_session_history(agent, session_store, session_id)
    except RuntimeError as exc:
        session_store.delete(session_id)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    session_store.append(session_id, "assistant", answer)
    print(answer)
    return 0


def run_with_session_history(agent: Agent, store: SessionStore, session_id: str) -> str:
    messages = [{"role": "system", "content": agent.system}]
    for message in store.messages(session_id):
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return agent.run_messages(messages)


def main(argv: list[str] | None = None, agent: Agent | None = None, store: SessionStore | None = None, config: ConfigStore | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log = verbose_logger(args.verbose)
    config_store = config or ConfigStore()
    if log is not None:
        log(f"command={args.command or 'chat'} backend={args.backend or 'auto'} model={args.model or 'auto'}")

    if args.command == "models":
        return list_model_recommendations(args.backend, args.limit, log)

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
                selected_model = resolve_model_choice(args.model, args.backend, args.select_model, log, config_store)
                selected_agent = agent or Agent(model=_ensure_model(selected_model, args.backend, log, args.device, args.startup_timeout))
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return run_chat(selected_agent, session_store, session_id=session_id)
        try:
            selected_model = resolve_model_choice(args.model, args.backend, args.select_model, log, config_store)
            selected_agent = agent or Agent(model=_ensure_model(selected_model, args.backend, log, args.device, args.startup_timeout))
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


def list_model_recommendations(backend: str | None, limit: int, log: object | None) -> int:
    selected_backend = backend or default_backend()
    recommendations = recommend_models(selected_backend, limit=max(1, limit), log=log)
    if not recommendations:
        print("No llmfit recommendations available.")
        return 1
    print("INDEX\tBACKEND\tMODEL")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"{index}\t{selected_backend}\t{recommendation.model}")
    print("\nUse one with: zeno --model <MODEL>")
    return 0


def select_model(model: str | None, backend: str | None, log: object | None, limit: int = 5) -> str | None:
    if model:
        return model
    selected_backend = backend or default_backend()
    recommendations = recommend_models(selected_backend, limit=limit, log=log)
    if not recommendations:
        print("No llmfit recommendations available; using the default model.")
        return None
    print("Select a model:")
    for index, recommendation in enumerate(recommendations, start=1):
        print(f"  {index}. {recommendation.model}")
    choice = input("Model number, custom model ID, or Enter for 1: ").strip()
    if not choice:
        return recommendations[0].model
    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(recommendations):
            return recommendations[index - 1].model
        print(f"Invalid selection {choice}; using the first recommendation.")
        return recommendations[0].model
    return choice


def resolve_model_choice(model: str | None, backend: str | None, interactive: bool, log: object | None, config: ConfigStore) -> str | None:
    selected_backend = backend or default_backend()
    if model:
        config.save_model(selected_backend, model)
        return model
    if interactive:
        selected_model = select_model(None, selected_backend, log)
        if selected_model:
            config.save_model(selected_backend, selected_model)
        return selected_model
    saved_model = config.model_for_backend(selected_backend)
    if saved_model:
        if log is not None:
            log(f"using saved model for {selected_backend}: {saved_model}")
        return saved_model
    return None
