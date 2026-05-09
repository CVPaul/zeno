from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ChatSession:
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    title: str


def default_session_dir() -> Path:
    return Path.cwd() / ".zeno" / "sessions"


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or default_session_dir()

    def create(self) -> str:
        self.root.mkdir(parents=True, exist_ok=True)
        created_at = _now()
        session_id = f"{created_at.replace(':', '').replace('-', '').replace('.', '')}-{uuid.uuid4().hex[:8]}"
        self._write(
            session_id,
            {
                "id": session_id,
                "created_at": created_at,
                "updated_at": created_at,
                "messages": [],
            },
        )
        return session_id

    def append(self, session_id: str, role: str, content: str) -> None:
        data = self._read(session_id)
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise RuntimeError(f"Session {session_id} has invalid messages")
        messages.append({"role": role, "content": content, "created_at": _now()})
        data["updated_at"] = _now()
        self._write(session_id, data)

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def messages(self, session_id: str) -> list[dict[str, object]]:
        data = self._read(session_id)
        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise RuntimeError(f"Session {session_id} has invalid messages")
        return [message for message in messages if isinstance(message, dict)]

    def latest_id(self) -> str | None:
        sessions = self.list()
        if not sessions:
            return None
        return sessions[0].session_id

    def list(self) -> list[ChatSession]:
        if not self.root.exists():
            return []
        sessions: list[ChatSession] = []
        for path in sorted(self.root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            data = _load_json(path)
            session_id = data.get("id")
            created_at = data.get("created_at")
            updated_at = data.get("updated_at")
            messages = data.get("messages", [])
            if not isinstance(session_id, str) or not isinstance(created_at, str) or not isinstance(updated_at, str):
                continue
            if not isinstance(messages, list):
                continue
            sessions.append(
                ChatSession(
                    session_id=session_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    message_count=len(messages),
                    title=_title(messages),
                )
            )
        return sessions

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def _read(self, session_id: str) -> dict[str, object]:
        return _load_json(self._path(session_id))

    def _write(self, session_id: str, data: dict[str, object]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(session_id).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"Session file is invalid: {path}")
    return data


def _title(messages: list[object]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip().replace("\n", " ")[:60]
    return "(empty)"
