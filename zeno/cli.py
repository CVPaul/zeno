from __future__ import annotations

import argparse
import sys

from .core import Agent, ensure_default_local_model
from .sessions import SessionStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeno", description="Minimal local-first agent CLI")
    parser.add_argument("--backend", choices=["vllm-mlx", "vllm"], help="Model backend to manage for this session")
    parser.add_argument("--model", help="Model to use for this chat session")
    parser.add_argument("--continue", dest="continue_session", action="store_true", help="Continue the latest task session in this workspace")
    subparsers = parser.add_subparsers(dest="command")

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


def main(argv: list[str] | None = None, agent: Agent | None = None, store: SessionStore | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None or args.command == "task":
        if args.command == "task" and args.task_command == "list":
            return list_tasks(store)
        if args.command is None and args.continue_session:
            session_store = store or SessionStore()
            session_id = session_store.latest_id()
            if session_id is None:
                print("error: no task sessions found in this workspace", file=sys.stderr)
                return 1
            try:
                selected_agent = agent or Agent(model=ensure_default_local_model(args.model, args.backend))
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 1
            return run_chat(selected_agent, session_store, session_id=session_id)
        try:
            selected_agent = agent or Agent(model=ensure_default_local_model(args.model, args.backend))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.command == "task" and args.task_command == "create":
            return run_task(args.description, selected_agent, store)
        return run_chat(selected_agent, store)

    parser.error(f"unknown command: {args.command}")
    return 2
