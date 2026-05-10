from __future__ import annotations

from pathlib import Path

from .types import Tool


def default_tools(root: Path | None = None) -> dict[str, Tool]:
    workspace = (root or Path.cwd()).resolve()

    def write_file(path: str, content: str) -> dict[str, object]:
        """Create or overwrite a UTF-8 text file inside the current workspace."""
        target = (workspace / path).resolve()
        if not target.is_relative_to(workspace):
            raise RuntimeError("write_file path must stay inside the workspace")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"path": str(target.relative_to(workspace)), "bytes": len(content.encode("utf-8"))}

    return {"write_file": write_file}
